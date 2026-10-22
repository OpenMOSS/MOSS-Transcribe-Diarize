"""实时转写的本地服务：WebSocket 收音频，HTTP 看历史与导出。

**本模块不得 import torch。** ``--backend vllm`` 下它必须能在一个没装 torch 的进程里
跑起来——这正是阶段一第一件事做的那层隔离的用途。转写器由 ``transcriber_factory``
注入，本模块不知道它是 vLLM 还是本地模型。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from moss_transcribe_diarize.prompts import DEFAULT_PROMPT

try:  # pragma: no cover - 只是为了给路由的形参注解一个可解析的名字
    # FastAPI 用 ``get_type_hints`` 解析处理函数的注解，而它只看**模块全局**——所以
    # ``websocket: WebSocket`` 这个注解必须在这里绑定，写进 ``create_realtime_app``
    # 的函数体里是不够的（那样 FastAPI 会把 ``websocket`` 当成查询参数，连接直接被
    # 1008 拒掉）。其余 fastapi 名字仍然在函数里惰性导入。
    from fastapi import WebSocket
except ImportError:                                   # 轻量部署没装 fastapi
    WebSocket = None  # type: ignore[assignment]

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.export import EXPORT_FORMATS, export_session
from moss_transcribe_diarize.realtime.session import RealtimeSession
from moss_transcribe_diarize.realtime.store import SessionStore

STATIC_DIR = Path(__file__).with_name("static")

_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".html": "text/html",
    ".ico": "image/x-icon",
    ".png": "image/png",
}

MAX_PENDING_SECONDS = 60.0
"""``start`` 之前最多代管多少秒音频。

浏览器开麦往往早于它发 ``start``，那些音频不能丢；但一个从不发 ``start`` 的客户端
（或一个坏掉的前端）不该能把服务端的内存堆满，所以超出上限就丢最旧的——与环形
缓冲同一套语义：保新不保旧。
"""


def _speaker_names(meta: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["id"]): str(item.get("name") or item["id"])
        for item in meta.get("speakers") or []
        if isinstance(item, dict) and item.get("id")
    }


def _trim_pending(pending: list[np.ndarray], limit_samples: int) -> int:
    """把 ``start`` 之前攒下的帧裁到 ``limit_samples`` 个样本，丢弃最旧的。

    返回裁剪后剩下的样本数。只保留最后一帧是**有意**的兜底：单帧就超上限时也要让
    会话收到东西，而不是把整个待处理列表清空。
    """
    total = sum(int(frame.size) for frame in pending)
    while total > limit_samples and len(pending) > 1:
        total -= int(pending.pop(0).size)
    return total


def _with_hotwords(prompt: str, hotwords: Any) -> str:
    """把热词拼进 prompt——模型只认 prompt，没有单独的热词参数。

    ``_start`` 与 ``set_hotwords`` 都走这里，免得两处各写一份拼接逻辑。
    """
    words = [str(word).strip() for word in (hotwords or []) if str(word).strip()]
    if not words:
        return prompt.strip()
    return "{} 热词提示：{}".format(prompt.strip(), ", ".join(words)).strip()


class SessionRegistry:
    """活跃会话的登记处。每个 WebSocket 连接一个会话，互不共享状态。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}

    def add(self, session_id: str, session: Any) -> None:
        self._sessions[session_id] = session

    def get(self, session_id: str) -> Any | None:
        return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active(self) -> list[str]:
        return sorted(self._sessions)


def create_realtime_app(
    *,
    config: RealtimeConfig,
    transcriber_factory: Callable[[], Any],
    embedder: Any = None,
    runs_dir: str | Path = "runs/realtime",
    static_dir: str | Path | None = None,
    probe: Callable[[], dict[str, Any]] | None = None,
    record_audio: bool = True,
    retranscribe: Callable[[Path, str], str] | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    except ImportError as exc:
        raise RuntimeError("Install fastapi and uvicorn to run the realtime service.") from exc

    app = FastAPI(title="MOSS Realtime")
    runs = Path(runs_dir).expanduser()
    runs.mkdir(parents=True, exist_ok=True)
    assets = Path(static_dir).expanduser() if static_dir else STATIC_DIR
    registry = SessionRegistry()
    app.state.registry = registry
    app.state.config = config
    app.state.runs_dir = runs

    def error(code: str, detail: str, status: int = 400):
        return JSONResponse({"detail": detail, "code": code}, status_code=status)

    def load_rows(session_id: str) -> list[dict]:
        try:
            return SessionStore.load_committed(runs, session_id)
        except ValueError as exc:                      # session_id 非法（含路径穿越）
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def load_meta(session_id: str) -> dict:
        try:
            return SessionStore.load_meta(runs, session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = assets / "realtime.html"
        if not page.exists():
            return HTMLResponse("<h1>MOSS Realtime</h1><p>realtime.html 尚未提供。</p>")
        return HTMLResponse(page.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/assets/{asset_path:path}")
    def asset(asset_path: str):
        """前端的 js/css/词条出口。

        ``resolve()`` 之后再比对父目录是这里的守卫：URL 段原样进来，直接拼路径就是
        任意文件读取。解析后再比，``..%2F..%2Fsecret.txt`` 落不到 assets 里。
        """
        root = assets.resolve()
        target = (root / asset_path).resolve()
        if not target.is_file() or root not in target.parents:
            return JSONResponse({"detail": "asset not found"}, status_code=404)
        media = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
        # 和 `/` 一样不缓存：这是个本地服务，页面与脚本都在快速迭代，缓存只会让"改了代码
        # 却没生效"变成一次玄学排查；对升级了包的人来说还会一直跑旧 JS。
        return FileResponse(target, media_type=media, headers={"Cache-Control": "no-store"})

    @app.get("/api/runtime")
    def runtime():
        return {
            "config": {
                "window": config.window,
                "hop": config.hop,
                "tail": config.tail,
                "silence_gate": config.silence_gate,
                "speaker_threshold": config.speaker_threshold,
                "poll_interval": config.poll_interval,
            },
            "backend": _transcriber_info(transcriber_factory, probe),
            "speaker": {
                "enabled": embedder is not None,
                "provider": type(embedder).__name__ if embedder is not None else None,
            },
            "active_sessions": registry.active(),
            "export_formats": list(EXPORT_FORMATS),
        }

    @app.get("/api/sessions")
    def list_sessions():
        return {"sessions": SessionStore.list_sessions(runs)}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        meta = load_meta(session_id)
        if not meta:
            return error("session_not_found", f"no such session: {session_id}", 404)
        return {"session": meta, "segments": load_rows(session_id)}

    @app.get("/api/sessions/{session_id}/audio")
    def get_audio(session_id: str):
        path = SessionStore.session_dir(runs, session_id) / "audio.wav"
        if not path.exists():
            return error("audio_missing", "this session has no recording", 404)
        return FileResponse(path, filename=path.name)

    @app.get("/api/sessions/{session_id}/export")
    def export(session_id: str, format: str = Query("srt")):
        meta = load_meta(session_id)
        if not meta:
            return error("session_not_found", f"no such session: {session_id}", 404)
        rows = load_rows(session_id)
        try:
            text = export_session(rows, format, speaker_names=_speaker_names(meta))
        except ValueError as exc:
            return error("invalid_format", str(exc), 400)
        return JSONResponse({"format": format, "text": text})

    @app.post("/api/sessions/{session_id}/retranscribe")
    def retranscribe_session(session_id: str):
        """把整段录音交给当前后端重跑一次（spec §4.8 / §5.2 的"重跑本次会话"）。

        走的是文件模式，与实时那条路共用同一个转写后端，只是不做分窗、不做流式。
        实时路径为了不拖 torch 而注入 ``transcriber_factory``，这里同理注入
        ``retranscribe``——所以本模块仍然不知道后端是什么，也就不需要 import torch。
        """
        if retranscribe is None:
            return error(
                "retranscribe_unavailable",
                "this deployment cannot re-run a session: no file-mode backend was provided",
                501,
            )
        meta = load_meta(session_id)
        if not meta:
            return error("session_not_found", f"no such session: {session_id}", 404)
        path = SessionStore.session_dir(runs, session_id) / "audio.wav"
        if not path.exists():
            return error("audio_missing", "this session has no recording", 404)
        if meta.get("status") != "done":
            # 会话还在录：audio.wav 是边写边追加的，读到的是一份 WAV 头声称的长度与实际
            # 内容对不上的半成品。要么被后端 400 掉，要么真的转出半场内容而用户以为那是
            # 整场会议——两种都不该发生。
            return error("session_busy", "this session is still recording; stop it first", 409)
        try:
            text = retranscribe(path, str(meta.get("prompt") or ""))
        except Exception as exc:                                  # noqa: BLE001
            return error("retranscribe_failed", f"{type(exc).__name__}: {exc}", 502)
        return JSONResponse({"session_id": session_id, "text": text})

    @app.websocket("/ws/realtime")
    async def realtime_socket(websocket: WebSocket):
        from fastapi import WebSocketDisconnect

        await websocket.accept()
        session: RealtimeSession | None = None
        session_id: str | None = None
        store: SessionStore | None = None
        pump: asyncio.Task | None = None
        pending: list[np.ndarray] = []
        stopping = asyncio.Event()
        pending_limit = int(MAX_PENDING_SECONDS * config.sample_rate)
        # 基础 prompt 与热词**分开存**：反复 set_hotwords 时拼在每次的结果上会越叠越长
        # （"热词提示：A 热词提示：B"），而且空热词永远清不掉旧的。
        base_prompt = DEFAULT_PROMPT
        hotwords: Any = None
        # 会话有三个写者：驱动任务（在工作线程里跑窗口）、收尾、以及这条连接上的控制指令。
        # `RealtimeSession` 明确要求前两者不得并发（见它的类文档），而控制指令同样会写
        # `Stitcher` 的水位线、说话人表和 `transcript.jsonl`——`reassign_segment` 的
        # `rewrite_committed` 是**整体重写**，撞上正在追加的窗口就会把刚定稿的那段永久
        # 抹掉，且不报任何错。这把锁就是那三个写者的互斥；代价是控制指令最多等一个窗口。
        session_lock = asyncio.Lock()

        async def _send(payload: dict) -> None:
            try:
                await websocket.send_json(payload)
            except Exception:                                   # 客户端已断开
                pass

        async def pump_windows() -> None:
            while True:
                try:
                    async with session_lock:
                        events = await session.run_pending()
                except Exception as exc:                        # 兜底：绝不让驱动任务静默死掉
                    await _send({"type": "error", "code": "driver_failed", "detail": str(exc)})
                    return
                for event in events:
                    await _send(event)
                if stopping.is_set():
                    return
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=config.poll_interval)
                except asyncio.TimeoutError:
                    continue

        async def _start(source: dict) -> None:
            nonlocal session, session_id, store, base_prompt, hotwords, pump
            if session is not None:
                return                                          # 重复的 start：忽略，不另开会话
            if source.get("prompt"):
                base_prompt = str(source["prompt"])
            hotwords = source.get("hotwords")
            prompt = _with_hotwords(base_prompt, hotwords) or None
            try:
                store = SessionStore(runs, name=str(source.get("session_name") or ""),
                                     record_audio=record_audio)
                # 构造转写器可能很久（hf 那条路要加载权重），必须在工作线程里做——否则
                # 整个事件循环在这段时间里不响应，别的会话和 /api/runtime 全都挂住。
                transcriber = await asyncio.get_running_loop().run_in_executor(
                    None, transcriber_factory
                )
                session = RealtimeSession(
                    config,
                    transcriber=transcriber,
                    store=store,
                    embedder=embedder,
                    **({"prompt": prompt} if prompt else {}),
                )
            except Exception as exc:                            # noqa: BLE001
                # 后端起不来（模型路径写错、显存不够）必须是**可观测的失败**：发一条
                # error，并把刚建出来的空会话目录收掉——否则历史里会多出一条永远停在
                # "recording"、既没有录音也没有转写的幽灵会话。
                if store is not None:
                    shutil.rmtree(store.dir, ignore_errors=True)
                    store = None
                await _send({
                    "type": "error",
                    "code": "backend_unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                return
            session_id = store.session_id
            # 把 prompt 落到 session.json：spec §4.6 要求它在那里，而且"重跑本次会话"
            # 要用同一个 prompt——不然重跑出来的内容与实时那次不是一回事。
            store.write_meta(prompt=session.prompt)
            registry.add(session_id, session)
            for frame in pending:
                session.push_audio(frame)
            pending.clear()
            # 先把这个事件发出去再起驱动任务：客户端要保证它收到的第一条就是
            # session 描述符（里面有 session_id），否则它无从知道自己在跟谁说话。
            await _send({
                "type": "session",
                "session_id": session_id,
                "started_at": time.time(),
                "config": {
                    "window": config.window, "hop": config.hop, "tail": config.tail,
                    "silence_gate": config.silence_gate,
                },
            })
            pump = asyncio.create_task(pump_windows())

        async def _shutdown() -> None:
            """先让驱动停下来，再收尾。

            刻意**不是** ``pump.cancel()``：取消一个正卡在 ``run_in_executor`` 里的任务
            只会让它停止等待，那个工作线程还在改 Stitcher 与 SessionStore，而 ``close()``
            紧接着就在另一个线程里做同样的事——这正是 ``RealtimeSession`` 文档里那条
            "两者不得并发"的契约。信号 + await 让当前窗口跑完再退出，收尾因此是串行的。
            """
            nonlocal session, store, pump
            if session is None:
                return
            if pump is not None:
                stopping.set()
                try:
                    await pump
                except Exception:                               # 驱动已经自己报过错
                    pass
                pump = None
            current, current_id = session, session_id
            session = None
            async with session_lock:                            # 驱动已退出，这里不会等
                events = await current.close()
            store = None
            for event in events:
                await _send(event)
            if current_id is not None:
                registry.drop(current_id)

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    if len(data) % 4 != 0:
                        await _send({
                            "type": "error",
                            "code": "invalid_audio_frame",
                            "detail": f"expected float32 little-endian frames, got {len(data)} bytes",
                        })
                        continue
                    frame = np.frombuffer(data, dtype="<f4")
                    # 长度对得上不代表内容是 PCM：把 int16 采样重新解释成 float32 会得到
                    # 一批 NaN/次正规数，长度刚好是 4 的倍数。不拦的话它有两种都不会报错的
                    # 死法：静音门控把整窗判成静音（一帧都不推理，也没有 error），或者被
                    # 编码成满幅噪声送进模型（幻觉文本）。NaN 绝不是合法 PCM。
                    if not np.isfinite(frame).all():
                        await _send({
                            "type": "error",
                            "code": "invalid_audio_frame",
                            "detail": "frame contained non-finite samples; expected float32 little-endian PCM",
                        })
                        continue
                    if session is None:
                        pending.append(frame)                   # start 之前先攒着，不丢
                        _trim_pending(pending, pending_limit)
                    else:
                        session.push_audio(frame)
                    continue
                text = message.get("text")
                if text is None:
                    continue
                try:
                    command = json.loads(text)
                except json.JSONDecodeError:
                    await _send({"type": "error", "code": "invalid_json", "detail": text[:120]})
                    continue
                if not isinstance(command, dict):
                    await _send({"type": "error", "code": "invalid_json", "detail": text[:120]})
                    continue
                kind = str(command.get("type") or "")
                if kind == "start":
                    await _start(command)
                elif kind == "stop":
                    break
                elif kind == "rename_speaker":
                    if session is None:
                        await _send({"type": "error", "code": "no_session", "detail": kind})
                        continue
                    async with session_lock:
                        try:
                            session.rename_speaker(
                                str(command.get("speaker_id")), str(command.get("name") or "")
                            )
                        except KeyError:
                            await _send({"type": "error", "code": "unknown_speaker",
                                         "detail": str(command.get("speaker_id"))})
                            continue
                        roster = session.speakers()
                    # 立刻落盘：导出与历史详情读的是 session.json 里的说话人表，而它平时
                    # 只在 finalize 时写。不写的话，会开一半改名再导出，文件里还是旧名字
                    # ——屏幕上叫"张总"、下载下来叫"S01"。
                    if store is not None:
                        store.write_meta(speakers=roster)
                    await _send({"type": "speaker", "speakers": roster})
                elif kind in ("set_prompt", "set_hotwords"):
                    if session is None:
                        await _send({"type": "error", "code": "no_session", "detail": kind})
                        continue
                    # 两条指令是**各自独立**的旋钮：set_hotwords 不该把上一次的拼接结果当成
                    # 新的基础 prompt 再拼一遍。所以这里存的是"基础 prompt"与"热词"两份。
                    if command.get("prompt"):
                        base_prompt = str(command["prompt"])
                    if "hotwords" in command:
                        hotwords = command["hotwords"]
                    async with session_lock:
                        session.set_prompt(_with_hotwords(base_prompt, hotwords))
                elif kind == "reassign_segment":
                    if session is None:
                        await _send({"type": "error", "code": "no_session", "detail": kind})
                        continue
                    async with session_lock:
                        updated = session.reassign_speaker(
                            str(command.get("segment_id")), str(command.get("speaker_id"))
                        )
                        roster = session.speakers() if updated is not None else None
                    if updated is None:
                        await _send({"type": "error", "code": "unknown_segment",
                                     "detail": str(command.get("segment_id"))})
                        continue
                    # 客户端对 committed 是"增量追加"语义；同一个 id 再来一次即原地替换
                    await _send({"type": "committed", "segments": [updated.to_dict()]})
                    await _send({"type": "speaker", "speakers": roster})
                else:
                    await _send({"type": "error", "code": "unknown_command", "detail": kind})
        except WebSocketDisconnect:
            pass
        finally:
            await _shutdown()

    return app


def _transcriber_info(
    transcriber_factory: Callable[[], Any],
    probe: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """尽量在不构造转写器的前提下描述后端。

    ``probe`` 由 CLI 注入（``--backend vllm`` 时是一次到端点的短探测）。**它抛异常
    也只是一条信息**：忘起 vLLM 是最常见的一次失误，而"服务本身还在不在"不该因为这个
    探测失败而变成 500。
    """
    info: dict[str, Any] = {"factory": getattr(transcriber_factory, "__name__", repr(transcriber_factory))}
    if probe is None:
        return info
    try:
        info.update(probe())
    except Exception as exc:                                  # noqa: BLE001
        info["reachable"] = False
        info["detail"] = f"{type(exc).__name__}: {exc}"
    return info
