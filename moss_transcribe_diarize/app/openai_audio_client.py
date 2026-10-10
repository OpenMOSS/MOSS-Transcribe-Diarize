"""OpenAI 兼容的音频转写客户端。**刻意不 import torch / transformers**。

`vllm_runner` 不能服务实时路径：它 `from .model_runner import ...`，而那个模块顶层
`import torch`。所以把 multipart 构造、wav 编码、SSE 消费、响应取文本这几件事放在
这里，字幕工坊与实时服务共用同一份实现，而两边都不因此拖进 torch。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Iterable

import numpy as np
import soundfile as sf

ProgressCallback = Callable[[int], None]


def _api_root(base_url: str) -> str:
    """把用户给的 base_url 归一成 OpenAI 兼容 API 的根（含 ``/v1``）。

    **所有对外请求都必须走这里**：探测与请求各拼各的，就会出现"转写请求走
    ``/v1/audio/transcriptions`` 而连通性探测打 ``/models``"这种自相矛盾——服务明明是好的，
    页面却报"后端不可达"。vLLM / SGLang 只在 ``/v1`` 下注册 OpenAI 兼容路由，裸 ``/models``
    是 404。
    """
    base = base_url.rstrip("/")
    if base.endswith("/audio/transcriptions"):
        base = base[: -len("/audio/transcriptions")].rstrip("/")
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def transcriptions_url(base_url: str) -> str:
    """把用户给的 base_url 归一成 /v1/audio/transcriptions 的完整地址。"""
    return _api_root(base_url) + "/audio/transcriptions"


def encode_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """float32 单声道 -> 16-bit PCM WAV 字节。

    用 PCM_16 而不是 float32：兼容性最好，且 20 秒窗口只有 640 KB。
    """
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(pcm, dtype=np.float32).reshape(-1), int(sample_rate),
             format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def build_multipart_body(
    *,
    boundary: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    content_type: str,
    file_bytes: bytes,
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks)


def extract_transcription_text(response: dict[str, Any]) -> str:
    content = response.get("text")
    return content.strip() if isinstance(content, str) else ""


def check_endpoint(
    base_url: str,
    *,
    api_key: str = "EMPTY",
    timeout: float = 2.0,
) -> dict[str, Any]:
    """探一下端点是否活着，供 ``/api/runtime`` 报告连通性。

    用户忘了先起 vLLM 服务是这条路上最常见的一次失误，而它现在的表现是"每跑一窗
    失败一次"——那要到第一次推理才看得见。这里用 ``/v1/models`` 做一次**短超时**
    的 GET，让服务在开始之前就能把话说清楚。

    **绝不抛异常**：探测失败是它要报告的结果，不是它的错误。
    """
    url = _api_root(base_url) + "/models"
    request = urllib.request.Request(url, method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if 200 <= response.status < 300:
                return {"reachable": True, "detail": f"GET {url} -> {response.status}"}
            return {"reachable": False, "detail": f"GET {url} -> {response.status}"}
    except urllib.error.HTTPError as exc:
        return {"reachable": False, "detail": f"GET {url} -> HTTP {exc.code}"}
    except Exception as exc:                                  # 连接被拒、超时、DNS……
        return {"reachable": False, "detail": f"GET {url} -> {type(exc).__name__}: {exc}"}


def consume_sse_transcription(
    response: Iterable[bytes],
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    parts: list[str] = []
    usage: dict[str, int] = {}

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        chunk = json.loads(data)

        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            prompt_tokens = chunk_usage.get("prompt_tokens")
            completion_tokens = chunk_usage.get("completion_tokens")
            if isinstance(prompt_tokens, int):
                usage["prompt_tokens"] = prompt_tokens
            if isinstance(completion_tokens, int):
                usage["completion_tokens"] = completion_tokens
                if on_progress is not None:
                    on_progress(completion_tokens)

        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                parts.append(delta["content"])

    return {"text": "".join(parts), "usage": usage}


def transcribe_bytes(
    *,
    base_url: str,
    model: str,
    prompt: str,
    file_bytes: bytes,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
    api_key: str = "EMPTY",
    timeout: float = 600.0,
    max_new_tokens: int = 1024,
    decoding: str = "greedy",
    temperature: float | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """POST 一段音频字节到 OpenAI 兼容端点，返回 {"text", "usage"}。"""
    boundary = f"----mtd-{uuid.uuid4().hex}"
    fields = {
        "model": model,
        "prompt": prompt.strip(),
        "response_format": "json",
        "stream": "true",
        "stream_include_usage": "true",
        "stream_continuous_usage_stats": "true",
        "max_completion_tokens": str(int(max_new_tokens)),
        "temperature": str(float(temperature if decoding == "sample" and temperature is not None else 0.0)),
    }
    body = build_multipart_body(
        boundary=boundary, fields=fields, file_field="file",
        filename=filename, content_type=content_type, file_bytes=file_bytes,
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        transcriptions_url(base_url), data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                return consume_sse_transcription(response, on_progress=on_progress)
            raw = response.read().decode("utf-8")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"text": raw}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"transcription request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach the transcription endpoint: {exc.reason}") from exc
