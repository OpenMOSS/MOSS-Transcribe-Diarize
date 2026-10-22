"""把一段 wav 按实时速度推给 mtd-realtime，并把事件打到终端。

    python scripts/realtime_client.py runs/demo-audio/meeting.wav
    python scripts/realtime_client.py file.wav --url ws://127.0.0.1:7870/ws/realtime --fast

没有浏览器也能验证整条链路，也是这台服务的手工验证入口。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf


def _connection_closed() -> tuple[type[BaseException], ...]:
    """服务端收完尾就会主动关连接——那是**正常结束**，不是错误。

    没有 websockets 时（例如只跑 chunk/wav 那几个纯函数测试）返回空元组，
    ``contextlib.suppress()`` 收下它也不会拦任何东西。
    """
    try:
        from websockets.exceptions import ConnectionClosed
    except ImportError:
        return ()
    return (ConnectionClosed,)


def _make_output_robust() -> None:
    """输出是中文，而控制台编码跟着区域设置走（英文 Windows 是 cp1252）。

    不处理的话，第一条中文事件就会让整个客户端抛 UnicodeEncodeError——刚好在"连上了、
    开始收事件"的那一刻。降级成替换字符：字可能缺几个，程序不能因为一行提示就断。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass


def wav_to_pcm16k(path: str | Path, target_rate: int = 16000) -> np.ndarray:
    """读成 16 kHz 单声道 float32。重采样交给已有的实现（soxr）。

    多声道先混成单声道：麦克风 + 系统声两路在浏览器里已经混过了，这里是给"直接拿一个
    立体声录音来试"用的。
    """
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if rate == target_rate:
        return mono
    import soxr

    return np.asarray(soxr.resample(mono, rate, target_rate), dtype=np.float32)


def chunk_pcm(pcm: np.ndarray, chunk_samples: int) -> Iterator[np.ndarray]:
    for start in range(0, len(pcm), chunk_samples):
        yield pcm[start:start + chunk_samples]


async def _drain(pcm, *, chunk_samples: int, realtime: bool, rate: int):
    """把分块按时间节奏吐出来——不在这个函数里做网络，好让节奏可单独测。

    用 ``await asyncio.sleep``，**不能**用 ``time.sleep``：这是个按实时速度推音频的
    客户端，推 70 秒就真要占满 70 秒。``time.sleep`` 会把事件循环堵死这么久，于是
    收不到服务端事件、也回不了它的 ping——实测 20 秒后服务端按超时把连接断掉
    （客户端看到 WinError 10053 / "no close frame received"）。
    """
    interval = chunk_samples / rate
    for chunk in chunk_pcm(pcm, chunk_samples):
        yield chunk
        if realtime:
            await asyncio.sleep(interval)


async def run(url: str, wav_path: Path, *, realtime: bool = True, rate: int = 16000,
              chunk_ms: int = 100, connect=None, tail_wait: float = 1.5,
              stop_timeout: float = 60.0) -> dict:
    pcm = wav_to_pcm16k(wav_path, rate)
    chunk_samples = int(rate * chunk_ms / 1000)
    stats = {"committed": 0, "errors": 0, "events": 0}

    if connect is None:                                  # 真的连；测试注入假的
        import websockets

        connect = websockets.connect

    async with connect(url) as socket:
        await socket.send(json.dumps({"type": "start", "session_name": wav_path.stem}))
        closed = asyncio.Event()

        async def pump() -> None:
            async for chunk in _drain(pcm, chunk_samples=chunk_samples, realtime=realtime, rate=rate):
                await socket.send(chunk.tobytes())

        async def listen() -> None:
            try:
                async for raw in socket:
                    if isinstance(raw, bytes):
                        continue
                    event = json.loads(raw)
                    stats["events"] += 1
                    kind = event.get("type")
                    if kind == "session":
                        print("  · 会话  {}".format(event["session_id"]))
                    elif kind == "committed":
                        stats["committed"] += len(event["segments"])
                        for seg in event["segments"]:
                            print("  + 定稿 {:7.2f}-{:<7.2f} {}  {}".format(
                                seg["start"], seg["end"], seg["speaker_name"], seg["text"]))
                    elif kind == "provisional":
                        body = "（空）" if not event["segments"] else " | ".join(
                            "{} {:.1f}-{:.1f}".format(s["speaker"], s["start"], s["end"])
                            for s in event["segments"])
                        print("  ~ 临时  " + body)
                    elif kind == "status":
                        print("  · 状态  rtf={:.2f} 落后={:.1f}s 已跳过={}窗 已跳过={:.0f}s{}".format(
                            event["rtf"], event["lag_sec"],
                            event.get("gated_windows", 0), event.get("gated_sec", 0.0),
                            "  【降级】" if event.get("degraded") else ""))
                    elif kind == "speaker":
                        print("  · 说话人  " + ", ".join(
                            "{}={}".format(s["id"], s["name"]) for s in event["speakers"]))
                    elif kind == "error":
                        stats["errors"] += 1
                        print("  ! 错误  {} {}".format(event["code"], event["detail"]))
            finally:
                closed.set()

        listener = asyncio.create_task(listen())
        await pump()
        await asyncio.sleep(tail_wait)                   # 让最后几窗跑完
        await socket.send(json.dumps({"type": "stop"}))
        # 等**服务端自己关连接**，而不是等一个固定秒数：收尾要跑完一个完整窗口（还可能
        # 有一次声纹嵌入），慢后端上就是几秒起步。等固定秒数会把收尾那几段——往往正是
        # 会议最后几句——漏打印，而盘上的 transcript.jsonl 是完整的，于是"看到的"和
        # "存下来的"对不上。服务端跑完 `_shutdown` 就关连接，那就是结束信号。
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(closed.wait(), timeout=stop_timeout)
        listener.cancel()
        with contextlib.suppress(asyncio.CancelledError, *_connection_closed()):
            await listener
    return stats


def main(argv: list[str] | None = None) -> int:
    _make_output_robust()
    parser = argparse.ArgumentParser(prog="realtime_client")
    parser.add_argument("wav", type=Path, help="要推的音频文件")
    parser.add_argument("--url", default="ws://127.0.0.1:7870/ws/realtime")
    parser.add_argument("--fast", action="store_true", help="不按实时速度推，尽快推完")
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--settle", type=float, default=1.5,
                        help="推完之后等多久让最后几窗跑完（秒）")
    args = parser.parse_args(argv)

    stats = asyncio.run(run(args.url, args.wav, realtime=not args.fast,
                            chunk_ms=args.chunk_ms, tail_wait=args.settle))
    print("\n定稿 {} 段，事件 {} 条，错误 {} 次".format(
        stats["committed"], stats["events"], stats["errors"]))
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
