"""起一个**不用 GPU、不用模型**的 mtd-realtime：内置一个假的 OpenAI 兼容端点。

给前端开发与走查用：整条链路（WebSocket 音频帧 → 窗口推理 → Stitcher → 落盘 →
导出 → 重跑）都是真的，只有"听清说了什么"这一步是替身——替身会真的解码收到的
窗口 WAV，按能量分段、按主频判人，返回与真实模型同形的 `[t0][S01]文本[t1]`。

    .venv/Scripts/python.exe scripts/realtime_dev_server.py
    # 然后打开 http://127.0.0.1:7871

要看**真模型**跑起来什么样，用 `mtd-realtime --backend hf --model <路径>`，不要用这个。
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
PHRASES = [
    "大家早上好，我们开始吧", "好的，我先说一下进度", "这个方案我来跟进",
    "那下周我们对一下结果", "我这边没有问题", "收到，会后我发纪要",
]
COUNTER = {"n": 0}


def dominant_hz(segment: np.ndarray) -> float:
    if segment.size < 512:
        return 0.0
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    return float(np.fft.rfftfreq(segment.size, 1.0 / SR)[int(np.argmax(spectrum))])


def frame_rms_db(audio: np.ndarray, frame: float = 0.02, hop: float = 0.01) -> np.ndarray:
    size, step = int(frame * SR), int(hop * SR)
    if audio.size < size:
        return np.zeros(0, dtype=np.float32)
    count = 1 + (audio.size - size) // step
    index = np.arange(size)[None, :] + step * np.arange(count)[:, None]
    rms = np.sqrt(np.mean(audio[index].astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-8))


def transcribe(pcm: np.ndarray) -> str:
    COUNTER["n"] += 1
    active = frame_rms_db(pcm) > -40.0
    runs, start = [], None
    for index, on in enumerate(active):
        if on and start is None:
            start = index
        elif not on and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(active)))

    labels, parts = {}, []
    for begin, end in runs:
        t0, t1 = begin * 0.01, end * 0.01
        if t1 - t0 < 0.4:
            continue
        hz = round(dominant_hz(pcm[int(t0 * SR):int(t1 * SR)]) / 25) * 25
        labels.setdefault(hz, f"S{len(labels) + 1:02d}")
        parts.append(f"[{t0:.2f}][{labels[hz]}]{PHRASES[len(parts) % len(PHRASES)]}[{t1:.2f}]")
    print(f"  [fake-vllm] window #{COUNTER['n']} {pcm.size / SR:.1f}s -> {len(parts)} 段", flush=True)
    return "".join(parts)


def extract_wav(body: bytes, boundary: bytes) -> bytes:
    marker = b'Content-Disposition: form-data; name="file"'
    at = body.find(marker)
    if at < 0:
        raise ValueError("no file field in the multipart body")
    start = body.find(b"\r\n\r\n", at) + 4
    return body[start:body.find(b"--" + boundary, start)].rstrip(b"\r\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _reply(self, status: int, content_type: str, payload: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._reply(200, "application/json",
                        json.dumps({"object": "list", "data": [{"id": "fake-moss"}]}).encode())
            return
        self.send_error(404)

    def do_POST(self):
        if not self.path.endswith("/audio/transcriptions"):
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        match = re.search(rb"boundary=([^\s;]+)", (self.headers.get("Content-Type") or "").encode())
        if not match:
            self.send_error(400, "no boundary")
            return
        try:
            pcm, rate = sf.read(io.BytesIO(extract_wav(body, match.group(1))), dtype="float32")
            if rate != SR:
                raise ValueError(f"expected {SR} Hz, got {rate}")
            text = transcribe(np.asarray(pcm, dtype=np.float32).reshape(-1))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [fake-vllm] ERROR {type(exc).__name__}: {exc}", flush=True)
            self.send_error(400, str(exc))
            return
        chunks = [
            "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}),
            "data: " + json.dumps({
                "choices": [{"delta": {"content": ""}}],
                "usage": {"prompt_tokens": 400, "completion_tokens": len(text) // 2},
            }),
            "data: [DONE]",
            "",
        ]
        self._reply(200, "text/event-stream", "\n".join(chunks).encode())


def main() -> int:
    # 控制台编码跟着区域设置走（英文 Windows 是 cp1252），而这里的提示是中文。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    parser = argparse.ArgumentParser(prog="realtime_dev_server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--fake-port", type=int, default=8123)
    parser.add_argument("--runs-dir", default="runs/realtime-dev")
    args = parser.parse_args()

    endpoint = ThreadingHTTPServer(("127.0.0.1", args.fake_port), Handler)
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()

    from moss_transcribe_diarize.app.realtime_cli import build_config, build_embedder, build_probe, build_retranscriber
    from moss_transcribe_diarize.app.realtime_server import create_realtime_app
    from moss_transcribe_diarize.app.realtime_cli import build_transcriber_factory, parse_args
    import uvicorn

    argv = ["--backend", "vllm", "--vllm-base-url", f"http://127.0.0.1:{args.fake_port}",
            "--runs-dir", args.runs_dir, "--no-speaker"]
    cli_args = parse_args(argv)
    config = build_config(cli_args)
    app = create_realtime_app(
        config=config,
        transcriber_factory=build_transcriber_factory(cli_args, config),
        embedder=build_embedder(cli_args),
        runs_dir=Path(args.runs_dir),
        probe=build_probe(cli_args),
        retranscribe=build_retranscriber(cli_args, config),
    )

    print(f"假端点在 http://127.0.0.1:{args.fake_port}；打开 http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
