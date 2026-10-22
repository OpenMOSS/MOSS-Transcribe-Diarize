"""On-disk persistence for realtime sessions."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from .stitch import Segment

# 会话 id 会被直接拼进路径，所以只允许单层、无路径分隔符的字符集合。`.`/`..` 虽
# 匹配字符集却仍能跳出 runs_dir，单独拒掉。
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def _validate_session_id(session_id: str) -> str:
    """校验会话 id 可以安全地作为一个目录名使用。

    这是唯一的两处拼接（``__init__`` 与 ``session_dir``）共用的守卫。阶段二的
    ``GET /api/sessions/{id}`` 会把 URL 段原样传进来，不在这里拦住就是路径穿越。
    """
    text = str(session_id)
    if not _SESSION_ID_PATTERN.fullmatch(text) or text in (".", ".."):
        raise ValueError(f"invalid session id: {session_id!r}")
    return text


class SessionStore:
    """把一个会话的元信息、录音和转写落到 ``runs_dir/<session-id>/``。

    ``transcript.jsonl`` 是崩溃恢复的权威来源——它逐条追加，任何时刻被中断都只
    影响最后一条。``audio.wav`` 是流式写入的，崩溃时 WAV 头里的长度字段会过期，
    音频内容还在但需要修复头部，属于尽力而为。
    """

    def __init__(
        self,
        runs_dir: str | Path,
        session_id: str | None = None,
        *,
        sample_rate: int = 16000,
        record_audio: bool = True,
        name: str = "",
    ):
        self.runs_dir = Path(runs_dir).expanduser()
        self.session_id = _validate_session_id(
            session_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        self.dir = self.runs_dir / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.record_audio = bool(record_audio)
        self.name = name
        self.audio_path = self.dir / "audio.wav"
        self.transcript_path = self.dir / "transcript.jsonl"
        self.provisional_path = self.dir / "provisional.json"
        self.meta_path = self.dir / "session.json"
        self._audio = None
        self._started_at = time.time()
        self._closed = False
        self.write_meta(status="recording")

    def append_audio(self, pcm: np.ndarray) -> None:
        if not self.record_audio or self._closed:
            return
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        if self._audio is None:
            self._audio = sf.SoundFile(
                str(self.audio_path),
                mode="w",
                samplerate=self.sample_rate,
                channels=1,
                subtype="PCM_16",
            )
        self._audio.write(arr)

    def append_committed(self, segments: Iterable[Any]) -> None:
        """整批追加已定稿段落；要么全写进去，要么一条都不留。

        "一条都不留"是调用方的依赖：``RealtimeSession._commit`` 在定稿失败时会回滚水位线，
        让下一个窗口重新定稿同一段内容。所以这里若留下半批，重来的那次就会把同一段文字
        再写一遍（换上新 id 的重复行）——那不叫恢复，只是把丢失换成了重复。

        做法：先在内存里把整批序列化好（后面某条坏了就一条都还没写），再一次写入；写入
        本身抛异常时**尽力**把文件截回追加前的长度。截断是尽力而为，不是保证——见下面
        except 里的说明。
        """
        payload = "".join(
            json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in segments
        )
        if not payload:
            return
        path = self.transcript_path
        size_before = path.stat().st_size if path.exists() else 0
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
        except Exception:
            # 截断是尽力而为，且它自己失败时抛出去的是截断的 OSError——原始的写入失败
            # 原因会被盖住。这时盘上可能留下**完整的一行**（写入恰好停在行边界上），而
            # 调用方会把水位线回滚、下一窗重新定稿同一段内容，于是那一行变成重复行。
            # 故意不吞这个异常：吞掉就会多出一个无法测试、且会把清理失败藏起来的分支。
            with path.open("r+b") as handle:
                handle.truncate(size_before)
            raise

    def rewrite_committed(self, segments: Iterable[Any]) -> None:
        """整体重写 ``transcript.jsonl``。

        段落被改动过（例如改了说话人）时用这个，而不是再 append 一遍——append 会留下
        同 id 的第二行，``load_committed`` 读出来就是重复。

        与 ``append_committed`` 同样的纪律：先把整批序列化好，再写一次，所以中途失败不会
        留下半截文件。
        """
        lines = [json.dumps(item.to_dict(), ensure_ascii=False) for item in segments]
        self.transcript_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    def write_provisional(self, segments: Iterable[Segment]) -> None:
        payload = [
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speaker": seg.speaker,
                "text": seg.text,
            }
            for seg in segments
        ]
        self.provisional_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def write_meta(self, **fields: Any) -> None:
        data = self.read_meta()
        data.update(fields)
        data.setdefault("session_id", self.session_id)
        data.setdefault("started_at", self._started_at)
        data["name"] = data.get("name") or self.name or self.session_id
        data["sample_rate"] = self.sample_rate
        data["record_audio"] = self.record_audio
        self.meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_meta(self) -> dict[str, Any]:
        return _read_json_object(self.meta_path)

    def finalize(
        self,
        speakers: Iterable[dict],
        *,
        status: str = "done",
    ) -> None:
        if self._closed:
            return
        self._closed = True
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        self.write_meta(status=status, ended_at=time.time(), speakers=list(speakers))

    @staticmethod
    def session_dir(runs_dir: str | Path, session_id: str) -> Path:
        return Path(runs_dir).expanduser() / _validate_session_id(session_id)

    @staticmethod
    def load_meta(runs_dir: str | Path, session_id: str) -> dict:
        """读取任意已有会话的 session.json；不存在或损坏时返回空 dict。

        会先校验 session_id：这层的调用方是 HTTP 路由，URL 段直接传进来。
        """
        path = SessionStore.session_dir(runs_dir, session_id) / "session.json"
        return _read_json_object(path)

    @staticmethod
    def load_committed(runs_dir: str | Path, session_id: str) -> list[dict]:
        path = SessionStore.session_dir(runs_dir, session_id) / "transcript.jsonl"
        if not path.exists():
            return []
        rows: list[dict] = []
        # ``errors="replace"`` 而不是默认的严格解码：文件是逐条追加的，崩溃可能把最后
        # 一行切断在某个多字节字符中间。严格模式会因这一个坏字符让**整份转写**抛
        # ``UnicodeDecodeError``，而正确行为只是丢掉那一行。坏字节解成 U+FFFD 后，
        # 该行的 JSON 解析失败、被下面的 except 跳过，前面的行照常返回。
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    @staticmethod
    def list_sessions(runs_dir: str | Path) -> list[dict]:
        root = Path(runs_dir).expanduser()
        if not root.is_dir():
            return []
        sessions: list[dict] = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            meta = _read_json_object(path / "session.json")
            if not meta:
                continue
            meta.setdefault("session_id", path.name)
            sessions.append(meta)
        sessions.sort(key=lambda item: float(item.get("started_at") or 0.0), reverse=True)
        return sessions


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        # 同 ``load_committed``：被切断在字符中间的元信息解成 U+FFFD 后 JSON 解析失败，
        # 走 except 返回空字典，而不是让一个坏文件把整个 ``list_sessions`` 拖崩。
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
