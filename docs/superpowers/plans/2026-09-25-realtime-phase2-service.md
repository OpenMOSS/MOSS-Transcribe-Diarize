# 实时会议转写 — 阶段二：服务与协议 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把阶段一那个库变成一个能跑起来的 `mtd-realtime` 服务：浏览器或脚本把 16 kHz PCM 推到 WebSocket，服务端按滑动窗口推理，把定稿/临时/说话人/状态事件推回来，并支持历史会话的查看与导出。

**Architecture:** 一个 FastAPI 应用，每个 WebSocket 连接对应一个 `RealtimeSession`（阶段一交付）。音频以二进制帧进入，控制指令以 JSON 文本帧进入。**每会话一个后台任务**按 `poll_interval` 驱动 `run_pending`，事件直接转发给该连接。转写后端两个：`--backend vllm` 走 OpenAI 兼容端点（torch-free），`--backend hf` 走本地 `ModelRunner`。

**Tech Stack:** Python 3.10+、FastAPI、uvicorn（需 `websockets` 支持）、阶段一的 `moss_transcribe_diarize.realtime`、`subtitle/export.py`。

**Spec:** `docs/superpowers/specs/2026-09-24-realtime-meeting-transcription-design.md`（阶段二相关：§4.5、§4.8、§4.9、§4.10、§5 之外的部分、§7、§12）

## Global Constraints

- `requires-python = ">=3.10"`。新代码一律 `from __future__ import annotations` + `X | None` 风格。
- **实时路径不得 import torch。** `moss_transcribe_diarize/realtime/*` 与 `app/realtime_server.py`、`app/realtime_cli.py` 在 `--backend vllm` 下**一个 torch 相关模块都不能加载**；`app/openai_audio_client.py` 必须 torch-free。已有的 `tests/test_realtime_speaker_onnx.py` 里那种"`sys.modules[x] = None` 阻断后仍能导入"的测试是这条约束的验证方式。
- 新增依赖只有 `websockets`（uvicorn 的 WS 支持），进 `[project.optional-dependencies] realtime`。**不预装用不到的依赖。**
- 本阶段允许修改的既有文件是这些，**且只做说明里的事**：`app/vllm_runner.py`（把 HTTP 管道抽出去）、`pyproject.toml`（入口与依赖）、`realtime/transcriber.py`（加 `VllmWindowTranscriber`、改预算推导）、`realtime/config.py`（`effective_max_new_tokens` 的定位变化）。
- **另外四处改动是为协议服务的**，spec §4.9 定义了 `set_prompt` / `set_hotwords` / `rename_speaker` / `reassign_segment` 四条指令，而 `RealtimeSession` 现在没有对应入口。与其让服务层伸手去改 `session._prompt`、`session._gallery`、`session._committed` 这些私有属性（阶段一已经为 `_gallery.rename` 破过一次例，那属于欠债），不如补上一组「会话自己拥有它的转写与说话人表」的公开方法：
  - `RealtimeSession.set_prompt(prompt: str) -> None`
  - `RealtimeSession.prompt: str`（属性，读回当前 prompt）
  - `RealtimeSession.speakers() -> list[dict]`（说话人表）
  - `RealtimeSession.rename_speaker(speaker_id: str, name: str) -> None`（未知 id 抛 `KeyError`）
  - `RealtimeSession.reassign_speaker(segment_id: str, speaker_id: str) -> CommittedSegment | None`（找不到返回 `None`；同时改写内存里的段落与 `transcript.jsonl`）
  - `SessionStore.rewrite_committed(segments) -> None`（供上面那条把改写后的段落落盘）
  **除这几处之外，`session.py` 与 `store.py` 不动。** `ModelRunner`、`JobManager`、`server.py`、`stitch.py`、`buffer.py`、`window.py`、`energy.py`、`speaker.py` 一行都不改。
- 测试用 `unittest.TestCase` 风格，pytest 运行。现有 `tests/test_vllm_runner.py` 是抽取 `openai_audio_client` 的回归保障，必须保持全绿。
- 运行命令里的 `python` 一律指 `.venv/Scripts/python.exe`（裸 `python` 在本机是 Store 占位程序，静默不执行）。

## Review Focus

以下五类是这个阶段会碰到、但 spec 没有逐条写明的输入与条件。每条都在拥有对应代码的任务里有测试。

1. **WebSocket 二进制帧的字节序与类型。** 浏览器端发的是 float32 小端；客户端若发 int16、或发了一个奇数长度的帧，`np.frombuffer` 会给出**长度不对但类型正确**的数组，转写结果会变成噪声而不是报错。必须校验帧长度是 4 的倍数并给出一条能看懂的 `error` 事件。
2. **控制指令出现在错误时机。** 音频帧到达时若还没收到 `start`、或 `start` 收到两次、或 `stop` 之后又来音频——每种都必须是可预期行为而不是异常穿透到连接层。
3. **两个会话并发。** 两个 WebSocket 同时连上：它们必须拿到各自独立的 `RealtimeSession` 与独立的会话目录，互不干扰；且**同一个 vLLM 端点被两边同时调用**不应让任何一方崩掉。
4. **`--backend vllm` 下端点不可达。** 用户忘了先起 vLLM 服务：`/api/runtime` 要能报告不可达，`start` 之后每次窗口失败要变成可观测的降级（`error` + `status.degraded`），而不是让连接静默卡死。
5. **导出时的字段来源。** 说话人显示名要取 `session.json` 的说话人表（用户改过名的），不是 `transcript.jsonl` 里冻结的旧值——否则用户改了名字却发现导出还是旧的。

---

### Task 1: 抽出 torch-free 的 OpenAI 音频客户端

阶段一的 review 记过一条（I4 的同族）：`vllm_runner.py` 为了拿 `TranscriptionResult`/`StatusCallback`/`generation_progress` 而 import 了 `model_runner`，而 `model_runner` 顶层 `import torch`。所以现有的 vLLM 路径**本身就不是 torch-free 的**，实时服务不能复用它。本任务把那套 HTTP 管道抽到一个不 import 任何重依赖的模块里，两边共用。

**Files:**
- Create: `moss_transcribe_diarize/app/openai_audio_client.py`
- Modify: `moss_transcribe_diarize/app/vllm_runner.py`（改为调用新模块）
- Modify: `moss_transcribe_diarize/app/__init__.py`（改成惰性导入——见 Step 0）
- Modify: `tests/test_vllm_runner.py`（打桩点从 `_post_multipart` 换成新模块的请求函数——见 Step 6）
- Test: `tests/test_openai_audio_client.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `transcriptions_url(base_url: str) -> str`
  - `encode_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes`
  - `build_multipart_body(*, boundary, fields, file_field, filename, content_type, file_bytes) -> bytes`
  - `extract_transcription_text(response: dict) -> str`
  - `consume_sse_transcription(response, *, on_progress=None) -> dict`
  - `transcribe_bytes(*, base_url, model, prompt, file_bytes, filename="audio.wav", content_type="audio/wav", api_key="EMPTY", timeout=600.0, max_new_tokens=1024, decoding="greedy", temperature=None, on_progress=None) -> dict`，返回 `{"text": str, "usage": {"prompt_tokens": int, "completion_tokens": int}}`

- [ ] **Step 0: 先让 `app/__init__.py` 惰性化**

**这一步不做，本任务的核心主张就是假的。** `moss_transcribe_diarize/app/__init__.py` 现在只有一行 `from .server import create_app`，而 `server.py` → `jobs.py` → `model_runner.py` 在顶层 `import torch`。所以 `import moss_transcribe_diarize.app.openai_audio_client` 会**先执行父包的 `__init__.py`**，照样把 torch 拖进来——新模块自己写得多干净都白费。

这与阶段一 Task 1 在包根上做的是同一件事（那里改的是 `moss_transcribe_diarize/__init__.py`），技术也相同：PEP 562 的模块级 `__getattr__`。

把 `moss_transcribe_diarize/app/__init__.py` 整体改成：

```python
"""Subtitle and realtime applications.

``create_app`` is resolved lazily (PEP 562) so that importing a lightweight
submodule does not drag in ``server`` and, through it, torch.
``app.openai_audio_client`` in particular has to stay importable in a process
that has never loaded torch — that is the whole point of the realtime service's
slim deployment.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRIBUTES = {"create_app": "server"}

__all__ = sorted(_LAZY_ATTRIBUTES)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
```

先确认没有东西依赖它是急切的：

Run: `grep -rn "from moss_transcribe_diarize.app import\|from .app import\|from ..app import" moss_transcribe_diarize tests`

期望：要么没有命中，要么命中的是 `create_app` 这种仍然能解析的名字。**若有任何地方 `from moss_transcribe_diarize.app import *`，停下来报告**——那是惰性导入唯一会破的用法。

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 全绿（本步只换解析时机，不该改变任何行为）

- [ ] **Step 1: 读现有的 vllm_runner.py，把要搬的三段抄出来**

`moss_transcribe_diarize/app/vllm_runner.py` 里的 `_transcriptions_url`、`_multipart_body`、`_extract_transcription_text`、`_consume_sse_transcription` 是纯函数式的、没有 torch 依赖的。`_media_to_wav_bytes` **不要搬**——它用 `load_audio_item`，那会拖进 transformers；新的 `encode_wav_bytes` 只接受 numpy。

先跑一次基线，确认起点是绿的：

Run: `.venv/Scripts/python.exe -m pytest tests/test_vllm_runner.py -q`
Expected: 全部 PASS（这是本次抽取的回归保障）

- [ ] **Step 2: 写失败测试**

创建 `tests/test_openai_audio_client.py`：

```python
from __future__ import annotations

import io
import json
import subprocess
import sys
import unittest

import numpy as np
import soundfile as sf

from moss_transcribe_diarize.app.openai_audio_client import (
    build_multipart_body,
    consume_sse_transcription,
    encode_wav_bytes,
    extract_transcription_text,
    transcriptions_url,
)


class TorchFreeTest(unittest.TestCase):
    def test_module_imports_without_torch(self):
        """这个模块是实时路径的地基，它一旦拖 torch，整个服务的轻量部署就没了。"""
        code = (
            "import sys\n"
            "for name in ('torch', 'transformers', 'moss_transcribe_diarize.app.model_runner'):\n"
            "    sys.modules[name] = None\n"
            "import moss_transcribe_diarize.app.openai_audio_client as c\n"
            "assert hasattr(c, 'transcribe_bytes')\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)


class TranscriptionsUrlTest(unittest.TestCase):
    def test_appends_the_path_to_a_bare_host(self):
        self.assertEqual(
            transcriptions_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/v1/audio/transcriptions",
        )

    def test_respects_a_trailing_slash(self):
        self.assertEqual(
            transcriptions_url("http://127.0.0.1:8000/"),
            "http://127.0.0.1:8000/v1/audio/transcriptions",
        )

    def test_keeps_an_explicit_v1(self):
        self.assertEqual(
            transcriptions_url("http://host/v1"),
            "http://host/v1/audio/transcriptions",
        )

    def test_keeps_an_explicit_full_path(self):
        self.assertEqual(
            transcriptions_url("http://host/v1/audio/transcriptions"),
            "http://host/v1/audio/transcriptions",
        )


class EncodeWavBytesTest(unittest.TestCase):
    def test_round_trips_through_soundfile(self):
        pcm = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

        data = encode_wav_bytes(pcm, 16000)

        self.assertTrue(data.startswith(b"RIFF"))
        back, rate = sf.read(io.BytesIO(data), dtype="float32")
        self.assertEqual(rate, 16000)
        # 16-bit PCM，所以只要求近似（量化误差）
        np.testing.assert_allclose(back, pcm, atol=1e-4)

    def test_accepts_an_empty_window_without_crashing(self):
        data = encode_wav_bytes(np.zeros(0, dtype=np.float32), 16000)
        self.assertTrue(data.startswith(b"RIFF"))


class BuildMultipartBodyTest(unittest.TestCase):
    def test_puts_every_field_in_the_body(self):
        body = build_multipart_body(
            boundary="BOUND",
            fields={"model": "m", "prompt": "p"},
            file_field="file",
            filename="audio.wav",
            content_type="audio/wav",
            file_bytes=b"\x01\x02",
        )

        text = body.decode("latin-1")
        self.assertIn('name="model"', text)
        self.assertIn("m", text)
        self.assertIn('name="prompt"', text)
        self.assertIn('filename="audio.wav"', text)
        self.assertIn("Content-Type: audio/wav", text)
        self.assertTrue(body.endswith(b"--BOUND--\r\n"))

    def test_file_bytes_are_raw_not_escaped(self):
        payload = bytes(range(256))

        body = build_multipart_body(
            boundary="B", fields={}, file_field="file", filename="a.wav",
            content_type="audio/wav", file_bytes=payload,
        )

        self.assertIn(payload, body)


class ExtractTranscriptionTextTest(unittest.TestCase):
    def test_reads_the_text_field(self):
        self.assertEqual(extract_transcription_text({"text": "  你好  "}), "你好")

    def test_missing_text_is_empty_not_none(self):
        self.assertEqual(extract_transcription_text({}), "")
        self.assertEqual(extract_transcription_text({"text": None}), "")


class _FakeResponse:
    """把 SSE 行喂给 consume_sse_transcription，不碰网络。"""

    def __init__(self, lines: list[str]):
        self._lines = [line.encode("utf-8") for line in lines]

    def __iter__(self):
        return iter(self._lines)


class ConsumeSseTest(unittest.TestCase):
    def test_joins_delta_content(self):
        response = _FakeResponse([
            'data: {"choices":[{"delta":{"content":"[0.5][S01]"}}]}',
            'data: {"choices":[{"delta":{"content":"你好[1.5]"}}]}',
            "data: [DONE]",
        ])

        result = consume_sse_transcription(response)

        self.assertEqual(result["text"], "[0.5][S01]你好[1.5]")

    def test_collects_usage_when_present(self):
        response = _FakeResponse([
            'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"prompt_tokens":7,"completion_tokens":9}}',
            "data: [DONE]",
        ])

        result = consume_sse_transcription(response)

        self.assertEqual(result["usage"], {"prompt_tokens": 7, "completion_tokens": 9})

    def test_on_progress_gets_the_completion_token_count(self):
        seen: list[int] = []
        response = _FakeResponse([
            'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":3}}',
            'data: {"choices":[{"delta":{"content":"y"}}],"usage":{"completion_tokens":4}}',
            "data: [DONE]",
        ])

        consume_sse_transcription(response, on_progress=seen.append)

        self.assertEqual(seen, [3, 4])

    def test_ignores_lines_that_are_not_data(self):
        response = _FakeResponse(["", ": keep-alive", "data: [DONE]"])

        self.assertEqual(consume_sse_transcription(response)["text"], "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_openai_audio_client.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.app.openai_audio_client'`

- [ ] **Step 4: 实现 `openai_audio_client.py`**

```python
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


def transcriptions_url(base_url: str) -> str:
    """把用户给的 base_url 归一成 /v1/audio/transcriptions 的完整地址。"""
    base = base_url.rstrip("/")
    if base.endswith("/audio/transcriptions"):
        return base
    if base.endswith("/v1"):
        return base + "/audio/transcriptions"
    return base + "/v1/audio/transcriptions"


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
```

- [ ] **Step 5: 运行新测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_openai_audio_client.py -q`
Expected: 全部 PASS

- [ ] **Step 6: 让 `vllm_runner.py` 改用新模块**

**先注意**：`tests/test_vllm_runner.py` 现在**直接依赖**几处私有接口——它打桩 `runner._post_multipart`（:26），并直接调 `runner._transcriptions_url()`（:58、:62）。所以这一步不只是改实现，**还要改那个测试的打桩点**，否则"回归保障"会变成"被打挂的东西"（Step 8 的验收会失败）。这是本任务必须一起做的一步，不是可选清理。

打开 `moss_transcribe_diarize/app/vllm_runner.py`，做这些替换（**只做这些**）：

1. 删掉模块级的 `_transcriptions_url`、`_multipart_body`、`_extract_transcription_text`、`_consume_sse_transcription` 四个自由函数，以及 `VllmRunner` 上的 `_build_fields`、`_transcriptions_url`、`_post_multipart` 三个方法（它们的内容现在住在 `transcribe_bytes` 里）。
2. 加 import：

```python
from .openai_audio_client import encode_wav_bytes, extract_transcription_text, transcribe_bytes
```

3. `transcribe()` 里把 `self._post_multipart(...)` 那一段换成：

```python
        response = transcribe_bytes(
            base_url=self.base_url,
            model=self.model_path,
            prompt=prompt.strip() or DEFAULT_PROMPT,
            file_bytes=wav_bytes,
            filename="audio.wav",
            api_key=self.api_key,
            timeout=self.timeout,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
            on_progress=(
                None
                if status_callback is None
                else lambda tokens: status_callback(
                    "transcribing", generation_progress(tokens, max_new_tokens), tokens
                )
            ),
        )
        text = extract_transcription_text(response)
        usage = response.get("usage") or {}
```

4. 清掉因此变成死引用的 import（`io`、`urllib.*`、`uuid`、`soundfile`——`io`/`soundfile` 若 `_media_to_wav_bytes` 还用得到就留着，见 Step 7）。

5. **改 `tests/test_vllm_runner.py` 的两处**：

   - 把打桩从 `runner._post_multipart = fake_post_multipart` 换成**打桩新模块的请求函数**。最省事的形式是在 `vllm_runner` 的命名空间里替换掉它引用的那个名字：

```python
            import moss_transcribe_diarize.app.vllm_runner as vllm_runner_module

            def fake_transcribe_bytes(**kwargs):
                sent.append(kwargs)
                return {"text": "[0.5][S01]你好[1.5]", "usage": {"completion_tokens": 9}}

            vllm_runner_module.transcribe_bytes = fake_transcribe_bytes
            self.addCleanup(setattr, vllm_runner_module, "transcribe_bytes", original)
```

   （`original` 在替换前先存下来。用 `addCleanup` 还原，别靠测试自己收尾。）

   - 把 `runner._transcriptions_url()` 的断言改成对新模块自由函数的断言：

```python
        from moss_transcribe_diarize.app.openai_audio_client import transcriptions_url

        self.assertEqual(
            transcriptions_url("http://host:8000/v1"), "http://host:8000/v1/audio/transcriptions"
        )
```

   **别把这两条断言删掉**——它们钉的是 URL 归一，那正是被搬走的逻辑；搬到新模块上继续钉，才算回归保障。

- [ ] **Step 7: 收敛 `_media_to_wav_bytes`**

它现在是 `io.BytesIO()` + `sf.write(buffer, audio, 16000, format="WAV")`。换成新模块的 `encode_wav_bytes`，**但保留 `load_audio_item`**（那是解码视频容器用的，仍然需要 transformers）：

```python
def _media_to_wav_bytes(path: str | Path) -> bytes:
    path = Path(path).expanduser()
    audio = load_audio_item(str(path), sampling_rate=16000)
    return encode_wav_bytes(audio, 16000)
```

于是 `vllm_runner.py` 顶部的 `import io` 与 `import soundfile as sf` 若再无他用，删掉。

- [ ] **Step 8: 跑回归 + 全量**

Run: `.venv/Scripts/python.exe -m pytest tests/test_vllm_runner.py -q`
Expected: 全部 PASS —— **这是本次抽取的验收依据**。`tests/test_vllm_runner.py` 覆盖了 URL 归一、multipart、SSE 消费与文本提取，正是被搬走的那几件事。

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 全绿

- [ ] **Step 9: 提交**

```bash
git add moss_transcribe_diarize/app/openai_audio_client.py moss_transcribe_diarize/app/vllm_runner.py tests/test_openai_audio_client.py
git commit -m "refactor(app): extract a torch-free OpenAI audio client

vllm_runner imports model_runner for TranscriptionResult and generation_progress,
and model_runner imports torch at module level, so the existing vLLM path could
not serve the realtime service. Move the multipart construction, wav encoding,
SSE consumption and text extraction into a module that pulls in nothing heavier
than numpy and soundfile, and have vllm_runner call it. encode_wav_bytes takes a
waveform rather than a path, which is what the realtime window adapter needs."
```

---

### Task 2: `VllmWindowTranscriber` 与按实际窗口长度推导的 token 预算

阶段一的 review 查出：`RealtimeConfig.effective_max_new_tokens()` 没有任何消费者，而 `HfWindowTranscriber` 把同一个数硬编码了第二份（`1020`）；更糟的是，4.7 的覆盖性保证会让**部分窗口长于 `window`**（被迫放行的窗口、驱动停摆后的窗口，实测可达 100 秒），而按 `window` 定死的预算会把它们的尾部截掉。本任务改成按**拿到的音频**推算。

**Files:**
- Modify: `moss_transcribe_diarize/realtime/transcriber.py`
- Test: `tests/test_realtime_transcriber.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `encode_wav_bytes`、`transcribe_bytes`、`extract_transcription_text`
- Produces:
  - `TOKENS_PER_AUDIO_SECOND = 51`、`MIN_NEW_TOKENS = 256`（从 `config.py` 迁来，见 Step 6）
  - `token_budget(audio: np.ndarray, sample_rate: int, *, floor: int = MIN_NEW_TOKENS, cap: int | None = None) -> int`
  - `VllmWindowTranscriber(*, base_url, model, api_key=None, timeout=600.0, decoding="greedy", sample_rate=16000, token_budget_floor=1024, max_new_tokens=None)`
  - `HfWindowTranscriber(runner, scratch_dir, *, sample_rate=16000, decoding="greedy", token_budget_floor=1024, max_new_tokens=None)` —— **签名变了**：`max_new_tokens` 从"硬编码的默认值"变成"可选的硬上限"，并新增 `token_budget_floor`

- [ ] **Step 1: 写失败测试**

在 `tests/test_realtime_transcriber.py` 追加：

```python
class TokenBudgetTest(unittest.TestCase):
    def test_scales_with_the_actual_audio(self):
        # 20 秒 -> 20*51 = 1020
        self.assertEqual(token_budget(np.zeros(20 * 16000, dtype=np.float32), 16000), 1020)

    def test_a_long_forced_window_gets_a_bigger_budget(self):
        # 覆盖性保证会让被迫放行的窗口长于 config.window（实测到过 100 秒），
        # 按 window 定死的预算会把它们的尾部截掉。
        self.assertEqual(token_budget(np.zeros(60 * 16000, dtype=np.float32), 16000), 3060)

    def test_floor_applies_to_short_windows(self):
        self.assertEqual(token_budget(np.zeros(16000, dtype=np.float32), 16000, floor=1024), 1024)

    def test_cap_wins_when_given(self):
        self.assertEqual(
            token_budget(np.zeros(60 * 16000, dtype=np.float32), 16000, cap=999), 999
        )

    def test_never_returns_less_than_the_minimum(self):
        self.assertEqual(token_budget(np.zeros(16, dtype=np.float32), 16000), 256)


class HfWindowTranscriberBudgetTest(unittest.TestCase):
    """沿用本文件已有的 FakeRunner。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name) / "scratch"

    def test_budget_follows_the_window_it_is_given(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch, token_budget_floor=256)

        transcriber.transcribe_window(np.zeros(20 * 16000, dtype=np.float32), prompt="p")

        self.assertEqual(runner.calls[-1]["max_new_tokens"], 1020)

    def test_explicit_max_new_tokens_still_caps(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch, max_new_tokens=777)

        transcriber.transcribe_window(np.zeros(60 * 16000, dtype=np.float32), prompt="p")

        self.assertEqual(runner.calls[-1]["max_new_tokens"], 777)


class VllmWindowTranscriberTest(unittest.TestCase):
    def test_is_a_window_transcriber(self):
        transcriber = VllmWindowTranscriber(base_url="http://x", model="m")
        self.assertIsInstance(transcriber, WindowTranscriber)

    def test_window_is_encoded_and_posted_without_touching_torch(self):
        sent: list[dict] = []

        def fake_post(**kwargs):
            sent.append(kwargs)
            return {"text": "[0.5][S01]你好[1.5]", "usage": {"completion_tokens": 9}}

        transcriber = VllmWindowTranscriber(
            base_url="http://x", model="m", token_budget_floor=256, post=fake_post
        )

        text = transcriber.transcribe_window(np.zeros(20 * 16000, dtype=np.float32), prompt="p")

        self.assertEqual(text, "[0.5][S01]你好[1.5]")
        self.assertEqual(sent[0]["model"], "m")
        self.assertEqual(sent[0]["prompt"], "p")
        self.assertEqual(sent[0]["max_new_tokens"], 1020)
        self.assertTrue(sent[0]["file_bytes"].startswith(b"RIFF"))

    def test_module_imports_without_torch(self):
        code = (
            "import sys\n"
            "for name in ('torch', 'transformers', 'moss_transcribe_diarize.app.model_runner'):\n"
            "    sys.modules[name] = None\n"
            "import moss_transcribe_diarize.realtime.transcriber as t\n"
            "assert hasattr(t, 'VllmWindowTranscriber')\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_transcriber.py -q`
Expected: FAIL —— `ImportError: cannot import name 'token_budget'` / `VllmWindowTranscriber`

- [ ] **Step 3: 实现预算与 `VllmWindowTranscriber`**

在 `moss_transcribe_diarize/realtime/transcriber.py` 顶部把 import 补成：

```python
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from moss_transcribe_diarize.app.openai_audio_client import (
    encode_wav_bytes,
    extract_transcription_text,
    transcribe_bytes,
)
```

（`soundfile` 的 import 留着给 `HfWindowTranscriber` 用。）

然后追加：

```python
TOKENS_PER_AUDIO_SECOND = 51
MIN_NEW_TOKENS = 256

PostFunction = Callable[..., dict]


def token_budget(
    audio: np.ndarray,
    sample_rate: int,
    *,
    floor: int = MIN_NEW_TOKENS,
    cap: int | None = None,
) -> int:
    """按**实际**音频长度推算输出预算。

    不能按 ``config.window`` 推算：4.7 的覆盖性保证会让被迫放行的窗口长于 window
    （实测到过 100 秒），定死的预算会把它们的尾部静默截掉。
    """
    seconds = float(np.asarray(audio).size) / float(sample_rate)
    budget = max(MIN_NEW_TOKENS, int(round(seconds * TOKENS_PER_AUDIO_SECOND)))
    budget = max(budget, int(floor))
    if cap is not None:
        budget = min(budget, int(cap))
    return budget


class VllmWindowTranscriber:
    """经 OpenAI 兼容端点转写，**不 import torch**。

    端点的请求函数可注入（``post=``），所以测试不需要一个真的 vLLM 服务在跑。
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        decoding: str = "greedy",
        sample_rate: int = 16000,
        token_budget_floor: int = 1024,
        max_new_tokens: int | None = None,
        post: PostFunction | None = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or "EMPTY"
        self.timeout = float(timeout)
        self.decoding = decoding
        self._sample_rate = int(sample_rate)
        self._floor = int(token_budget_floor)
        self._cap = None if max_new_tokens is None else int(max_new_tokens)
        self._post = post or transcribe_bytes

    def transcribe_window(self, audio: np.ndarray, *, prompt: str) -> str:
        response = self._post(
            base_url=self.base_url,
            model=self.model,
            prompt=prompt,
            file_bytes=encode_wav_bytes(audio, self._sample_rate),
            filename="window.wav",
            api_key=self.api_key,
            timeout=self.timeout,
            max_new_tokens=token_budget(
                audio, self._sample_rate, floor=self._floor, cap=self._cap
            ),
            decoding=self.decoding,
        )
        return extract_transcription_text(response)
```

- [ ] **Step 4: 改 `HfWindowTranscriber` 用同一套预算**

把它的构造函数签名与 `transcribe_window` 改成：

```python
    def __init__(
        self,
        runner,
        scratch_dir: str | Path,
        *,
        sample_rate: int = 16000,
        decoding: str = "greedy",
        token_budget_floor: int = 1024,
        max_new_tokens: int | None = None,
    ):
        self._runner = runner
        self._scratch_dir = Path(scratch_dir).expanduser()
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._scratch_dir / "window.wav"
        self._sample_rate = int(sample_rate)
        self._decoding = decoding
        self._floor = int(token_budget_floor)
        self._cap = None if max_new_tokens is None else int(max_new_tokens)

    def transcribe_window(self, audio: np.ndarray, *, prompt: str) -> str:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        sf.write(str(self._path), arr, self._sample_rate, format="WAV", subtype="PCM_16")
        result = self._runner.transcribe(
            self._path,
            prompt=prompt,
            max_new_tokens=token_budget(arr, self._sample_rate, floor=self._floor, cap=self._cap),
            decoding=self._decoding,
        )
        return result.text
```

**注意**：老签名里的 `max_new_tokens: int = 1020` 是个**默认值**，新签名里它是**可选的上限**。本文件已有的 `test_forwards_prompt_and_budget` 显式传 777，语义不变 ✓；`test_returns_runner_text` 等不传预算的用例会走到"按音频推算"，而它们的音频是 1600 样本 → 预算 256（floor 默认 1024 → 1024）。**若某条既有用例断言了 1020，那是它钉住了旧的硬编码，应当改成断言按音频推算的值。**

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_transcriber.py -q`
Expected: 全部 PASS

- [ ] **Step 6: 把魔数从 `config.py` 收敛掉**

`moss_transcribe_diarize/realtime/config.py` 里现在有 `TOKENS_PER_AUDIO_SECOND = 51` 与 `MIN_MAX_NEW_TOKENS = 256`，以及 `effective_max_new_tokens()`。把它改成从 `transcriber` 引入常量，避免同一个数写两份：

```python
from .transcriber import MIN_NEW_TOKENS, TOKENS_PER_AUDIO_SECOND
```

并把 `effective_max_new_tokens()` 的文档改成它现在的角色：

```python
    def effective_max_new_tokens(self) -> int:
        """按 ``window`` 推算的预算——**只用作下限**。

        真正的预算是按每次拿到的音频长度算的（见 ``transcriber.token_budget``）：
        4.7 的覆盖性保证会让被迫放行的窗口长于 ``window``，按 ``window`` 定死的预算
        会把它们的尾部截掉。这里保留这个方法，是给调用方一个"一个 window 长的窗口
        至少需要多少 token"的下限。
        """
        if self.max_new_tokens is not None:
            return self.max_new_tokens
        return max(MIN_NEW_TOKENS, int(round(self.window * TOKENS_PER_AUDIO_SECOND)))
```

Hmm — **注意循环 import**：`config.py` 引 `transcriber.py`，而 `transcriber.py` **不**引 `config.py` ✓ 没有环。但 `transcriber.py` 现在引 `app.openai_audio_client`，后者引 numpy/soundfile ✓ 仍然 torch-free ✓。

- [ ] **Step 7: 全量 + 提交**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 全绿

```bash
git add moss_transcribe_diarize/realtime/transcriber.py moss_transcribe_diarize/realtime/config.py tests/test_realtime_transcriber.py
git commit -m "feat(realtime): add the vLLM window transcriber and per-window token budgets

The budget was derived from config.window, but window coverage guarantees that
some windows run longer than window — a forced or stalled window measured up to
100 seconds — so a budget fixed at window silently truncated their tails. Derive
it from the audio the transcriber was actually handed instead, with the config's
value as a floor and an optional explicit cap.

Adds VllmWindowTranscriber on top of the torch-free client, so the realtime
service can run with --backend vllm without importing torch."
```

---

### Task 3: `realtime/export.py` —— 桥接与五路导出

**Files:**
- Create: `moss_transcribe_diarize/realtime/export.py`
- Test: `tests/test_realtime_export.py`

**Interfaces:**
- Consumes: `moss_transcribe_diarize.subtitle` 的 `SubtitleSegment`、`export_srt`、`export_json`、`export_ass`；`realtime/stitch.py` 的 `Segment`
- Produces:
  - `to_subtitle_segments(rows: Iterable[dict], speaker_names: dict[str, str] | None = None) -> list[SubtitleSegment]`
  - `export_markdown(rows, *, speaker_names=None) -> str`
  - `export_text(rows, *, speaker_names=None) -> str`
  - `export_session(rows, fmt: str, *, speaker_names=None) -> str`，`fmt ∈ {"srt","json","ass","md","txt"}`
  - `EXPORT_FORMATS: tuple[str, ...]`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_export.py`：

```python
from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.export import (
    EXPORT_FORMATS,
    export_markdown,
    export_session,
    export_text,
    to_subtitle_segments,
)

ROWS = [
    {"id": "seg-1", "start": 0.12, "end": 2.86, "speaker": "S01",
     "speaker_name": "S01", "text": "大家早上好", "speaker_confident": True},
    {"id": "seg-2", "start": 7.08, "end": 10.61, "speaker": "S02",
     "speaker_name": "S02", "text": "我先说一下进度", "speaker_confident": True},
]
NAMES = {"S01": "张总", "S02": "李工"}


class ToSubtitleSegmentsTest(unittest.TestCase):
    def test_maps_every_field(self):
        segments = to_subtitle_segments(ROWS)

        self.assertEqual([s.id for s in segments], ["seg-1", "seg-2"])
        self.assertEqual(segments[0].start, 0.12)
        self.assertEqual(segments[0].end, 2.86)
        self.assertEqual(segments[0].text, "大家早上好")

    def test_uses_the_speaker_name_when_the_roster_has_one(self):
        """用户把 S01 改成了"张总"，导出就该是"张总"，而不是冻结在 jsonl 里的旧值。"""
        segments = to_subtitle_segments(ROWS, NAMES)

        self.assertEqual(segments[0].speaker, "张总")

    def test_falls_back_to_the_stored_name(self):
        segments = to_subtitle_segments(ROWS, {"S02": "李工"})

        self.assertEqual(segments[0].speaker, "S01")
        self.assertEqual(segments[1].speaker, "李工")

    def test_skips_rows_without_a_speaker(self):
        rows = ROWS + [{"id": "seg-3", "start": 1.0, "end": 2.0, "text": "无"}]  # 但缺 speaker
        segments = to_subtitle_segments(rows)
        self.assertEqual(len(segments), 3)


class MarkdownExportTest(unittest.TestCase):
    def test_has_a_heading_and_one_line_per_segment(self):
        text = export_markdown(ROWS, speaker_names=NAMES)

        self.assertTrue(text.startswith("# "), text[:20])
        self.assertIn("张总", text)
        self.assertIn("大家早上好", text)
        self.assertIn("00:00", text)

    def test_empty_transcript_still_produces_a_document(self):
        text = export_markdown([])
        self.assertTrue(text.startswith("# "))


class TextExportTest(unittest.TestCase):
    def test_has_no_markdown_syntax(self):
        text = export_text(ROWS, speaker_names=NAMES)

        self.assertNotIn("#", text)
        self.assertIn("张总", text)
        self.assertIn("大家早上好", text)

    def test_blank_line_between_segments_for_readability(self):
        self.assertEqual(export_text(ROWS).count("\n\n"), 2)


class ExportSessionTest(unittest.TestCase):
    def test_every_advertised_format_produces_something(self):
        for fmt in EXPORT_FORMATS:
            with self.subTest(fmt=fmt):
                self.assertTrue(export_session(ROWS, fmt).strip())

    def test_srt_looks_like_srt(self):
        self.assertIn("-->", export_session(ROWS, "srt"))

    def test_json_is_a_list_of_the_bridged_segments(self):
        import json

        payload = json.loads(export_session(ROWS, "json"))
        self.assertEqual(payload[0]["id"], "seg-1")

    def test_unknown_format_raises_and_names_the_known_ones(self):
        with self.assertRaises(ValueError) as ctx:
            export_session(ROWS, "docx")
        self.assertIn("srt", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_export.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.export'`

- [ ] **Step 3: 实现 `export.py`**

```python
"""把一个会话的定稿段落导成人类与机器都读得懂的几种格式。

``srt`` / ``json`` / ``ass`` 直接复用 ``subtitle/export.py``——那里的实现已经被
字幕工坊用测试钉住，重复一份只会让两边漂移。``md`` / ``txt`` 那个包没有，在这里写。

说话人显示名一律**先查说话人表**：用户把 S01 改成"张总"之后，``transcript.jsonl``
里冻结的还是旧值，导出必须用新的。
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from moss_transcribe_diarize.subtitle import (
    SubtitleSegment,
    export_ass,
    export_json,
    export_srt,
)

EXPORT_FORMATS: tuple[str, ...] = ("srt", "json", "ass", "md", "txt")


def to_subtitle_segments(
    rows: Iterable[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
) -> list[SubtitleSegment]:
    """把 ``transcript.jsonl`` 的行桥到 ``SubtitleSegment``。

    两者的字段几乎一一对应，差别只有两处：这里没有 ``speaker_confident``，
    而 ``speaker`` 要换成显示名。
    """
    names = dict(speaker_names or {})
    segments: list[SubtitleSegment] = []
    for index, row in enumerate(rows):
        speaker_id = str(row.get("speaker") or "")
        fallback = str(row.get("speaker_name") or speaker_id)
        segments.append(
            SubtitleSegment(
                id=str(row.get("id") or f"seg-{index + 1}"),
                start=float(row.get("start") or 0.0),
                end=float(row.get("end") or 0.0),
                speaker=names.get(speaker_id) or fallback,
                text=str(row.get("text") or ""),
            )
        )
    return segments


def _as_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    return "{:02d}:{:02d}:{:02d}".format(total // 3600, (total % 3600) // 60, total % 60)


def export_markdown(rows: Iterable[dict[str, Any]], *, speaker_names: dict[str, str] | None = None) -> str:
    segments = to_subtitle_segments(rows, speaker_names)
    lines = ["# 实时会议转写", ""]
    if not segments:
        lines.append("（本次会话没有定稿段落）")
    for seg in segments:
        lines.append("- `{}` **{}**：{}".format(_as_clock(seg.start), seg.speaker, seg.text))
    return "\n".join(lines) + "\n"


def export_text(rows: Iterable[dict[str, Any]], *, speaker_names: dict[str, str] | None = None) -> str:
    segments = to_subtitle_segments(rows, speaker_names)
    if not segments:
        return "（本次会话没有定稿段落）\n"
    return "\n\n".join(
        "[{}] {}: {}".format(_as_clock(seg.start), seg.speaker, seg.text) for seg in segments
    ) + "\n"


def export_session(
    rows: Iterable[dict[str, Any]],
    fmt: str,
    *,
    speaker_names: dict[str, str] | None = None,
) -> str:
    key = str(fmt or "").lower()
    if key not in EXPORT_FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; known: {', '.join(EXPORT_FORMATS)}")
    if key in ("md", "txt"):
        rows = list(rows)
        return (
            export_markdown(rows, speaker_names=speaker_names)
            if key == "md"
            else export_text(rows, speaker_names=speaker_names)
        )
    segments = to_subtitle_segments(rows, speaker_names)
    if key == "srt":
        return export_srt(segments)
    if key == "json":
        return export_json(segments)
    return export_ass(segments)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_export.py -q`
Expected: 全部 PASS

**若 `export_srt`/`export_json`/`export_ass` 的实际签名与推断不符**（比如它们要求额外的关键字参数），**停下来报告**而不是改这个模块的接口去迁就——先确认那个包的真实签名，再据此调整。

- [ ] **Step 5: 全量 + 提交**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

```bash
git add moss_transcribe_diarize/realtime/export.py tests/test_realtime_export.py
git commit -m "feat(realtime): export sessions as srt, json, ass, markdown or text

srt/json/ass reuse subtitle/export.py so the two callers cannot drift; md and
txt do not exist there and live here. Speaker display names are looked up in the
session roster first, because transcript.jsonl froze the name at commit time and
the user may have renamed the speaker since."
```

---

### Task 4: `app/realtime_server.py` —— HTTP 路由

**Files:**
- Create: `moss_transcribe_diarize/app/realtime_server.py`
- Test: `tests/test_realtime_api.py`

**Interfaces:**
- Consumes: `SessionStore`、`realtime/export.py`、阶段一的全部组件
- Produces:
  - `create_realtime_app(*, config: RealtimeConfig, transcriber_factory, embedder, runs_dir, static_dir=None) -> FastAPI`
  - `SessionRegistry`：管理活跃会话（`start`, `get`, `stop`, `list_active`）
  - 路由见 spec §4.8（**本任务只做 HTTP，WebSocket 在 Task 5**）

**说明**：`transcriber_factory` 是个零参可调用对象，每次新会话调用一次——因为 `HfWindowTranscriber` **只能串行使用**（它固定写同一个 `window.wav`），每会话必须有自己的实例。这条约束来自 Task 8 的复审，已写进 spec 的计划文本。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_api.py`：

```python
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.session import RealtimeSession
from moss_transcribe_diarize.realtime.store import SessionStore


class ScriptedTranscriber:
    def __init__(self, replies=None):
        self.replies = list(replies or ["[1.0][S01]你好[2.0]"] * 8)
        self.windows = 0
        self.prompts: list[str] = []          # 让测试能观测 prompt 真的换了

    def transcribe_window(self, audio, *, prompt):
        self.windows += 1
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


def _app(tmp: Path, **config_kwargs):
    from moss_transcribe_diarize.app.realtime_server import create_realtime_app

    created: list[ScriptedTranscriber] = []

    def factory() -> ScriptedTranscriber:
        instance = ScriptedTranscriber()
        created.append(instance)
        return instance

    config = RealtimeConfig(silence_gate=False, **config_kwargs)
    app = create_realtime_app(
        config=config,
        transcriber_factory=factory,
        embedder=None,
        runs_dir=tmp / "runs",
    )
    app.state.created_transcribers = created
    return app


class RuntimeRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client = TestClient(_app(Path(self._tmp.name)))

    def test_reports_the_effective_config(self):
        payload = self.client.get("/api/runtime").json()

        self.assertEqual(payload["config"]["window"], 20.0)
        self.assertEqual(payload["config"]["hop"], 5.0)
        self.assertIn("backend", payload)

    def test_reports_speaker_state(self):
        payload = self.client.get("/api/runtime").json()
        self.assertIn("speaker", payload)
        self.assertIn("enabled", payload["speaker"])


class SessionsRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        # 先手工造一个已完成的会话，供读取路由使用
        store = SessionStore(self.runs, "s1", name="周会")
        store.append_committed([])
        store.write_meta(speakers=[{"id": "S01", "name": "张总", "samples": 2}])
        SessionStore(self.runs, "s1").append_committed(_Rows())
        store.finalize([], [{"id": "S01", "name": "张总", "samples": 2}])
        self.client = TestClient(_app(Path(self._tmp.name)))

    def test_lists_sessions(self):
        payload = self.client.get("/api/sessions").json()
        self.assertEqual([s["session_id"] for s in payload["sessions"]], ["s1"])

    def test_returns_a_session_with_its_segments(self):
        payload = self.client.get("/api/sessions/s1").json()
        self.assertEqual(payload["session"]["name"], "周会")
        self.assertEqual(payload["segments"][0]["id"], "seg-1")

    def test_unknown_session_is_404_not_500(self):
        self.assertEqual(self.client.get("/api/sessions/nope").status_code, 404)

    def test_a_path_traversal_id_is_rejected_not_served(self):
        response = self.client.get("/api/sessions/..%2F..%2Fetc")
        self.assertIn(response.status_code, (400, 404))

    def test_exports_every_advertised_format(self):
        for fmt in ("srt", "json", "ass", "md", "txt"):
            with self.subTest(fmt=fmt):
                response = self.client.get(f"/api/sessions/s1/export?format={fmt}")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.text.strip())

    def test_export_uses_the_renamed_speaker(self):
        response = self.client.get("/api/sessions/s1/export?format=txt")
        self.assertIn("张总", response.text)

    def test_unknown_export_format_is_400(self):
        self.assertEqual(
            self.client.get("/api/sessions/s1/export?format=docx").status_code, 400
        )


class _Rows(list):
    def __iter__(self):
        return iter([
            {"id": "seg-1", "start": 1.0, "end": 2.0, "speaker": "S01",
             "speaker_name": "S01", "text": "你好", "speaker_confident": True},
        ])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_api.py -q`
Expected: FAIL —— `ModuleNotFoundError`

- [ ] **Step 3: 实现 `realtime_server.py` 的 HTTP 部分**

```python
"""实时转写的本地服务：WebSocket 收音频，HTTP 看历史与导出。

**本模块不得 import torch。** `--backend vllm` 下它必须能在一个没装 torch 的进程里
跑起来——这正是阶段一第一件事做的那层隔离的用途。转写器由 `transcriber_factory`
注入，本模块不知道它是 vLLM 还是本地模型。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.export import EXPORT_FORMATS, export_session
from moss_transcribe_diarize.realtime.store import SessionStore

STATIC_DIR = Path(__file__).with_name("static")


def _speaker_names(meta: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["id"]): str(item.get("name") or item["id"])
        for item in meta.get("speakers") or []
        if isinstance(item, dict) and item.get("id")
    }


def _with_hotwords(prompt: str, hotwords: Any) -> str:
    """把热词拼进 prompt——模型只认 prompt，没有单独的热词参数。

    `_start` 与 `set_hotwords` 都走这里，免得两处各写一份拼接逻辑。
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

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = assets / "realtime.html"
        if not page.exists():
            return HTMLResponse("<h1>MOSS Realtime</h1><p>realtime.html 尚未提供。</p>")
        return HTMLResponse(page.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

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
            "backend": _transcriber_info(transcriber_factory),
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
        meta = SessionStore.load_meta(runs, session_id)     # 见 Step 4
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
        rows = load_rows(session_id)
        meta = SessionStore.load_meta(runs, session_id)
        try:
            text = export_session(rows, format, speaker_names=_speaker_names(meta))
        except ValueError as exc:
            return error("invalid_format", str(exc), 400)
        return JSONResponse({"format": format, "text": text})

    return app


def _transcriber_info(transcriber_factory: Callable[[], Any]) -> dict[str, Any]:
    """尽量在不构造转写器的前提下描述后端。"""
    return {"factory": getattr(transcriber_factory, "__name__", repr(transcriber_factory))}
```

- [ ] **Step 4: 给 `SessionStore` 加 `load_meta`**

`get_session` 需要读一个会话的 meta。`SessionStore` 有 `read_meta()`（实例方法，针对已构造的 store），但这里是静态读任意会话——加一个静态方法比每处 new 一个 store 干净：

在 `moss_transcribe_diarize/realtime/store.py` 的 `load_committed` 旁边加：

```python
    @staticmethod
    def load_meta(runs_dir: str | Path, session_id: str) -> dict:
        """读取任意已有会话的 session.json；不存在或损坏时返回空 dict。

        会先校验 session_id：这层的调用方是 HTTP 路由，URL 段直接传进来。
        """
        path = SessionStore.session_dir(runs_dir, session_id) / "session.json"
        return _read_json_object(path)
```

Hmm — **`SessionStore.session_dir` 现在会抛 `ValueError` 吗？** 会（前面那个任务的校验）✓。所以非法 id 会抛，路由要接住 ✓（`load_rows` 已经接了；`load_meta` 的调用点也要接）。**Step 3 的 `get_session` 与 `export` 里对 `load_meta` 的调用要用 try/except ValueError 包住**——实现时统一放进一个 `_load_meta(session_id)` 辅助函数里，与 `load_rows` 同样处理。

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_api.py -q`
Expected: 全部 PASS

- [ ] **Step 6: 全量 + 提交**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

```bash
git add moss_transcribe_diarize/app/realtime_server.py moss_transcribe_diarize/realtime/store.py tests/test_realtime_api.py
git commit -m "feat(app): add the realtime service's HTTP surface

Runtime info, session listing, session detail, recording download and export.
Export resolves speaker display names from the session roster rather than from
the transcript rows, which froze the name at commit time. Session ids reaching
a path join are validated by the store primitive, so a traversal attempt is a
400 rather than a 500 or a served file."
```

---

### Task 5: WebSocket 端点与每会话驱动

**Files:**
- Modify: `moss_transcribe_diarize/app/realtime_server.py`
- Test: `tests/test_realtime_api.py`（追加）

**Interfaces:**
- Consumes: Task 4 的 app、`RealtimeSession`、`SessionStore`
- Produces: `WS /ws/realtime`，协议见 spec §4.9

**协议实现要点：**

- 二进制帧 → `np.frombuffer(frame, dtype="<f4")`，**先校验 `len(frame) % 4 == 0`**，否则回一条 `error` 并丢弃该帧（不关连接）。
- 文本帧 → JSON 控制指令。收到 `start` 之前到达的音频**先缓冲在内存里**（不丢），`start` 到达后再一并推进会话——浏览器常常先开麦后发指令。
- `start` 之前到达的控制指令只有 `start` 有意义；别的回 `error`。
- 每会话一个 `asyncio.Task`：`while True: events = await session.run_pending(); 转发; await asyncio.sleep(poll_interval)`。
- `stop` 或连接断开：**先取消那个任务并 await 它结束，再 `await session.close()`**——这是 spec §4.7 的串行契约（两个并发会让两个窗口落在两个线程里并改共享状态，而且不会报错）。
- 事件原样转发 ✓（`session` 已经把它们造成 dict）。

- [ ] **Step 1: 写失败测试**

在 `tests/test_realtime_api.py` 追加：

```python
class WebSocketTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client = TestClient(_app(Path(self._tmp.name), min_first_window=0.5))

    def _pcm(self, seconds: float) -> bytes:
        t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("<f4").tobytes()

    def test_first_event_is_the_session_descriptor(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start", "session_name": "测试会议"})
            first = ws.receive_json()
            self.assertEqual(first["type"], "session")
            self.assertTrue(first["session_id"])

    def test_audio_produces_committed_segments(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            ws.send_bytes(self._pcm(1.0))
            seen = self._drain(ws, want="committed")
            self.assertTrue(seen["segments"])

    def test_audio_before_start_is_buffered_not_dropped(self):
        """浏览器常常先开麦后发 start，那些音频不该丢。"""
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_bytes(self._pcm(1.0))
            ws.send_json({"type": "start"})
            ws.receive_json()
            seen = self._drain(ws, want="committed")
            self.assertTrue(seen["segments"])

    def test_a_misaligned_frame_is_reported_and_the_connection_survives(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            ws.send_bytes(b"\x01\x02\x03")           # 不是 4 的倍数
            error = self._drain(ws, want="error")
            self.assertEqual(error["code"], "invalid_audio_frame")
            ws.send_bytes(self._pcm(1.0))            # 连接仍然可用
            self.assertTrue(self._drain(ws, want="committed")["segments"])

    def test_stop_closes_the_session_and_persists_it(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            session_id = ws.receive_json()["session_id"]
            ws.send_bytes(self._pcm(1.0))
            self._drain(ws, want="committed")
            ws.send_json({"type": "stop"})
            self._drain(ws, want="speaker")

        meta = SessionStore.load_meta(Path(self._tmp.name) / "runs", session_id)
        self.assertEqual(meta["status"], "done")

    def test_two_sessions_get_separate_directories(self):
        ids = []
        for _ in range(2):
            with self.client.websocket_connect("/ws/realtime") as ws:
                ws.send_json({"type": "start"})
                ids.append(ws.receive_json()["session_id"])
                ws.send_json({"type": "stop"})
        self.assertNotEqual(ids[0], ids[1])
        for session_id in ids:
            self.assertTrue((Path(self._tmp.name) / "runs" / session_id).is_dir())

    def test_rename_speaker_updates_the_roster(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            ws.send_bytes(self._pcm(1.0))
            self._drain(ws, want="committed")
            ws.send_json({"type": "rename_speaker", "speaker_id": "S01", "name": "张总"})
            roster = self._drain(ws, want="speaker")
            self.assertIn("张总", [item["name"] for item in roster["speakers"]])

    def test_renaming_an_unknown_speaker_is_an_error(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            ws.send_json({"type": "rename_speaker", "speaker_id": "S99", "name": "谁"})
            self.assertEqual(self._drain(ws, want="error")["code"], "unknown_speaker")

    def test_set_hotwords_reaches_the_transcriber(self):
        """模型只认 prompt，没有单独的热词参数——所以热词要拼进 prompt 交给下一次推理。"""
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            ws.send_json({"type": "set_hotwords", "hotwords": ["阿里云", "降噪"]})
            ws.send_bytes(self._pcm(1.0))
            self._drain(ws, want="committed")

        prompts = self.client.app.state.created_transcribers[-1].prompts
        self.assertTrue(prompts)
        self.assertIn("阿里云", prompts[-1])
        self.assertIn("降噪", prompts[-1])

    def test_reassigning_a_segment_emits_it_again_with_the_new_speaker(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            ws.send_bytes(self._pcm(1.0))
            segment_id = self._drain(ws, want="committed")["segments"][0]["id"]

            ws.send_json({"type": "reassign_segment", "segment_id": segment_id, "speaker_id": "S01"})

            again = self._drain(ws, want="committed")
            self.assertEqual(again["segments"][0]["id"], segment_id)
            self.assertEqual(again["segments"][0]["speaker"], "S01")

    def test_reassignment_survives_into_the_stored_transcript(self):
        """改了说话人却不落盘，下次打开就没了——而导出正是读那份文件。"""
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            session_id = ws.receive_json()["session_id"]
            ws.send_bytes(self._pcm(1.0))
            segment_id = self._drain(ws, want="committed")["segments"][0]["id"]
            ws.send_json({"type": "reassign_segment", "segment_id": segment_id, "speaker_id": "U99"})
            self._drain(ws, want="committed")
            ws.send_json({"type": "stop"})
            self._drain(ws, want="speaker")

        rows = SessionStore.load_committed(Path(self._tmp.name) / "runs", session_id)
        self.assertEqual(rows[0]["speaker"], "U99")

    def test_reassigning_an_unknown_segment_is_an_error(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            ws.send_json({"type": "reassign_segment", "segment_id": "seg-999", "speaker_id": "S01"})
            self.assertEqual(self._drain(ws, want="error")["code"], "unknown_segment")

    def test_unknown_control_message_is_an_error_not_a_crash(self):
        with self.client.websocket_connect("/ws/realtime") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            ws.send_json({"type": "wat"})
            self.assertEqual(self._drain(ws, want="error")["code"], "unknown_command")

    def _drain(self, ws, *, want: str, limit: int = 60) -> dict:
        """读到第一条指定类型的事件为止（别的类型直接跳过）。"""
        for _ in range(limit):
            event = ws.receive_json()
            if event["type"] == want:
                return event
        self.fail(f"没有收到 {want} 事件")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_api.py -q`
Expected: FAIL —— WebSocket 路由不存在（404 或连接失败）

- [ ] **Step 3: 先补 `RealtimeSession` 与 `SessionStore` 的公开入口**

先做这个，端点才有东西可调。**这也是把阶段一那笔债还掉的地方**：服务层原本要伸手改 `session._prompt`、`session._gallery`、`session._committed`。

在 `moss_transcribe_diarize/realtime/session.py` 里，把顶部的 `from dataclasses import dataclass` 改成 `from dataclasses import dataclass, replace`，然后在 `close()` 后面加：

```python
    @property
    def prompt(self) -> str:
        return self._prompt

    def set_prompt(self, prompt: str) -> None:
        """换掉后续窗口用的 prompt。

        只影响**之后**的窗口；已经定稿的内容不会被改写——这与"定稿永不回改"一致。
        """
        self._prompt = str(prompt or self._prompt)

    def speakers(self) -> list[dict]:
        return self._gallery.speakers()

    def rename_speaker(self, speaker_id: str, name: str) -> None:
        """给全局说话人一个显示名。未知 id 抛 `KeyError`。"""
        self._gallery.rename(speaker_id, name)

    def reassign_speaker(self, segment_id: str, speaker_id: str) -> CommittedSegment | None:
        """把一条已定稿段落改到另一个说话人，并**立刻落盘**。

        不落盘的话重开会话就丢了——而导出读的正是那份文件。找不到该 id 返回 ``None``。
        """
        for index, item in enumerate(self._committed):
            if item.id != segment_id:
                continue
            updated = replace(
                item,
                speaker_id=speaker_id,
                speaker_name=self._gallery.display_name(speaker_id),
            )
            self._committed[index] = updated
            self._store.rewrite_committed(self._committed)
            return updated
        return None
```

在 `moss_transcribe_diarize/realtime/store.py` 的 `append_committed` 之后加：

```python
    def rewrite_committed(self, segments: Iterable[Any]) -> None:
        """整体重写 ``transcript.jsonl``。

        段落被改动过（例如改了说话人）时用这个，而不是再 append 一遍——append 会留下
        同 id 的第二行，``load_committed`` 读出来就是重复。

        与 ``append_committed`` 同样的纪律：先把整批序列化好，再写一次，所以中途失败不会
        留下半截文件。
        """
        lines = [json.dumps(item.to_dict(), ensure_ascii=False) for item in segments]
        self.transcript_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
```

在 `tests/test_realtime_session.py` 追加（`_config` / `_speech` / `_run` / `ScriptedTranscriber` 沿用该文件已有的辅助）：

```python
class PublicControlSurfaceTest(unittest.TestCase):
    """服务层不再伸进私有属性——这些就是它要用的那组公开入口。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, **kwargs) -> RealtimeSession:
        return RealtimeSession(
            _config(),
            transcriber=ScriptedTranscriber(["[1][S01]你好[2]"] * 4),
            store=SessionStore(self.runs, "s1"),
            **kwargs,
        )

    def test_prompt_can_be_read_and_replaced(self):
        session = self._session()
        self.assertTrue(session.prompt)

        session.set_prompt("新 prompt")

        self.assertEqual(session.prompt, "新 prompt")

    def test_setting_an_empty_prompt_keeps_the_current_one(self):
        session = self._session()
        before = session.prompt

        session.set_prompt("")

        self.assertEqual(session.prompt, before)

    def test_speakers_is_empty_without_an_embedder(self):
        self.assertEqual(self._session().speakers(), [])

    def test_renaming_an_unknown_speaker_raises(self):
        with self.assertRaises(KeyError):
            self._session().rename_speaker("S99", "谁")

    def test_reassigning_an_unknown_segment_returns_none(self):
        self.assertIsNone(self._session().reassign_speaker("seg-999", "S01"))

    def test_reassigning_rewrites_the_stored_row(self):
        session = self._session()
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())
        segment_id = session.committed[0].id

        updated = session.reassign_speaker(segment_id, "U99")

        self.assertEqual(updated.speaker_id, "U99")
        rows = SessionStore.load_committed(self.runs, "s1")
        self.assertEqual([row["speaker"] for row in rows], ["U99"])
```

- [ ] **Step 4: 实现 WebSocket 端点**

在 `create_realtime_app` 里加（`app = FastAPI(...)` 之后、其余路由旁边）：

```python
    @app.websocket("/ws/realtime")
    async def realtime_socket(websocket):
        from fastapi import WebSocketDisconnect

        await websocket.accept()
        session = None
        pump: asyncio.Task | None = None
        pending: list[np.ndarray] = []

        async def pump_windows():
            while True:
                try:
                    events = await session.run_pending()
                except Exception as exc:                       # 兜底：绝不让驱动任务静默死掉
                    await _send({"type": "error", "code": "driver_failed", "detail": str(exc)})
                    return
                for event in events:
                    await _send(event)
                await asyncio.sleep(config.poll_interval)

        async def _send(payload: dict) -> None:
            try:
                await websocket.send_json(payload)
            except Exception:                                   # 客户端已断开
                pass

        def _start(source: dict) -> None:
            nonlocal session, pump
            if session is not None:
                return
            prompt = _with_hotwords(str(source.get("prompt") or ""), source.get("hotwords")) or None
            store = SessionStore(runs, name=str(source.get("session_name") or ""))
            session = RealtimeSession(
                config,
                transcriber=transcriber_factory(),
                store=store,
                embedder=embedder,
                **({"prompt": prompt} if prompt else {}),
            )
            registry.add(store.session_id, session)
            pump = asyncio.create_task(pump_windows())
            for frame in pending:
                session.push_audio(frame)
            pending.clear()
            asyncio.create_task(_send({
                "type": "session",
                "session_id": store.session_id,
                "started_at": time.time(),
                "config": {
                    "window": config.window, "hop": config.hop, "tail": config.tail,
                    "silence_gate": config.silence_gate,
                },
            }))

        async def _shutdown() -> None:
            nonlocal session, pump
            if session is None:
                return
            if pump is not None:                                # 先让驱动停下来，再收尾
                pump.cancel()
                try:
                    await pump
                except (asyncio.CancelledError, Exception):
                    pass
                pump = None
            current = session
            session = None
            for event in await current.close():
                await _send(event)
            registry.drop(current._store.session_id)

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
                    if session is None:
                        pending.append(frame)                   # start 之前先攒着，不丢
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
                kind = str((command or {}).get("type") or "")
                if kind == "start":
                    _start(command or {})
                elif kind == "stop":
                    break
                elif kind == "rename_speaker":
                    if session is None:
                        await _send({"type": "error", "code": "no_session", "detail": kind})
                        continue
                    try:
                        session.rename_speaker(
                            str(command.get("speaker_id")), str(command.get("name") or "")
                        )
                    except KeyError:
                        await _send({"type": "error", "code": "unknown_speaker",
                                     "detail": str(command.get("speaker_id"))})
                        continue
                    await _send({"type": "speaker", "speakers": session.speakers()})
                elif kind in ("set_prompt", "set_hotwords"):
                    if session is None:
                        await _send({"type": "error", "code": "no_session", "detail": kind})
                        continue
                    base = str(command.get("prompt") or session.prompt)
                    session.set_prompt(_with_hotwords(base, command.get("hotwords")))
                elif kind == "reassign_segment":
                    if session is None:
                        await _send({"type": "error", "code": "no_session", "detail": kind})
                        continue
                    updated = session.reassign_speaker(
                        str(command.get("segment_id")), str(command.get("speaker_id"))
                    )
                    if updated is None:
                        await _send({"type": "error", "code": "unknown_segment",
                                     "detail": str(command.get("segment_id"))})
                        continue
                    # 客户端对 committed 是"增量追加"语义；同一个 id 再来一次即原地替换
                    await _send({"type": "committed", "segments": [updated.to_dict()]})
                    await _send({"type": "speaker", "speakers": session.speakers()})
                else:
                    await _send({"type": "error", "code": "unknown_command", "detail": kind})
        except WebSocketDisconnect:
            pass
        finally:
            await _shutdown()
```

同时把文件顶部的 import 补上：`import asyncio`、`import numpy as np`、`from moss_transcribe_diarize.realtime.session import RealtimeSession`。

**注意**：`rename_speaker` 用了 `session._gallery`（私有）。**这是有意的取舍**：`RealtimeSession` 没有暴露改名入口，而这个阶段不打算动 `session.py`（Global Constraints 里写死了）。实现时在那一行上面加一句注释说明原因，并在报告里把这条提出来——**它值得在后续把 `RealtimeSession` 加一个 `rename_speaker()` 公开方法时消掉**。

- [ ] **Step 5: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_api.py tests/test_realtime_session.py -q`
Expected: 全部 PASS

**若 `TestClient` 的 WebSocket 上下文在 `stop` 之后立刻关闭**导致 `_drain` 收不到 `speaker` 事件，**不要改测试去放宽**——先确认 `_shutdown` 里的 `close()` 是否真的跑了（在它后面加一条 `print` 临时看），因为"收尾事件有没有发出去"正是这条测试要钉的东西。

- [ ] **Step 6: 全量 + 提交**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

```bash
git add moss_transcribe_diarize/app/realtime_server.py moss_transcribe_diarize/realtime/session.py moss_transcribe_diarize/realtime/store.py tests/test_realtime_api.py tests/test_realtime_session.py
git commit -m "feat(app): add the realtime websocket endpoint

Binary frames are float32 PCM; a frame whose length is not a multiple of four is
reported and dropped rather than decoded into nonsense. Audio arriving before
the start command is buffered rather than discarded, because a browser can open
the microphone before it sends the command. Each session gets its own driver
task, and stop cancels that task before closing the session — the serial
contract the session documents, whose violation corrupts shared state silently.

The protocol's set_prompt, set_hotwords, rename_speaker and reassign_segment
commands need the session to own its transcript and roster, so RealtimeSession
gains a small public surface for them and SessionStore gains rewrite_committed.
The server no longer reaches into private attributes, which retires a debt phase
one left behind when it renamed speakers through _gallery."
```

---

### Task 6: `mtd-realtime` CLI 与入口

**Files:**
- Create: `moss_transcribe_diarize/app/realtime_cli.py`
- Modify: `pyproject.toml`（`websockets` 依赖 + `mtd-realtime` 脚本入口）
- Test: `tests/test_realtime_cli.py`

**Interfaces:**
- Consumes: Task 4/5 的 `create_realtime_app`；`ModelRunner`（仅 `--backend hf` 时）
- Produces: `main(argv=None) -> int`、`parse_args(argv) -> Namespace`、`build_transcriber_factory(args) -> Callable`、`build_embedder(args) -> object | None`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_cli.py`：

```python
from __future__ import annotations

import unittest

from moss_transcribe_diarize.app.realtime_cli import build_embedder, parse_args


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self):
        args = parse_args([])

        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 7870)
        self.assertEqual(args.window, 20.0)
        self.assertEqual(args.hop, 5.0)
        self.assertEqual(args.tail, 6.0)
        self.assertEqual(args.backend, "hf")

    def test_vllm_backend_takes_a_base_url(self):
        args = parse_args(["--backend", "vllm", "--vllm-base-url", "http://host:8000"])
        self.assertEqual(args.vllm_base_url, "http://host:8000")

    def test_no_speaker_disables_the_embedder_flag(self):
        self.assertFalse(parse_args(["--no-speaker"]).speaker)
        self.assertTrue(parse_args([]).speaker)

    def test_no_silence_gate_and_no_record(self):
        args = parse_args(["--no-silence-gate", "--no-record"])
        self.assertFalse(args.silence_gate)
        self.assertFalse(args.record)

    def test_help_mentions_the_loopback_default_so_nobody_exposes_it_by_accident(self):
        import contextlib, io

        buffer = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buffer):
            parse_args(["--help"])
        self.assertIn("127.0.0.1", buffer.getvalue())


class BuildEmbedderTest(unittest.TestCase):
    def test_no_speaker_gives_none(self):
        self.assertIsNone(build_embedder(parse_args(["--no-speaker"])))

    def test_speaker_model_path_is_honoured_without_downloading(self):
        args = parse_args(["--speaker-model", "some/path.onnx"])
        # 只验证它把路径传下去了；真正构造需要 onnxruntime 与那个文件
        self.assertEqual(args.speaker_model, "some/path.onnx")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_cli.py -q`
Expected: FAIL —— `ModuleNotFoundError`

- [ ] **Step 3: 实现 `realtime_cli.py`**

```python
"""`mtd-realtime` 的入口。

**本模块不得 import torch。** `--backend hf` 的实现里才会去 import
`ModelRunner`（那是本地跑模型必须的），`--backend vllm` 下永远不加载——所以那句
import 写在工厂函数里面，不写在文件顶部。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

from moss_transcribe_diarize.realtime.config import RealtimeConfig

LOOPBACK_WARNING = (
    "服务默认只绑 127.0.0.1，且**没有鉴权**。绑到 0.0.0.0 等于把会议录音的转写接口"
    "开放给同网段，请只在可信网络里这么做。"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mtd-realtime",
        description="实时会议语音转文字。默认绑 127.0.0.1:7870；服务无鉴权，别直接暴露到公网。",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize",
                        help="--backend hf 时的模型路径或 HF id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--vllm-api-key", default=None)
    parser.add_argument("--window", type=float, default=20.0)
    parser.add_argument("--hop", type=float, default=5.0)
    parser.add_argument("--tail", type=float, default=6.0)
    parser.add_argument("--silence-gate", dest="silence_gate", action="store_true", default=True)
    parser.add_argument("--no-silence-gate", dest="silence_gate", action="store_false")
    parser.add_argument("--record", dest="record", action="store_true", default=True)
    parser.add_argument("--no-record", dest="record", action="store_false",
                        help="不落盘完整录音，只保留转写文本")
    parser.add_argument("--speaker", dest="speaker", action="store_true", default=True)
    parser.add_argument("--no-speaker", dest="speaker", action="store_false",
                        help="关闭全局说话人一致，退化为每窗口局部标签")
    parser.add_argument("--speaker-model", default=None, help="本地声纹模型路径；默认自动下载")
    parser.add_argument("--speaker-threshold", type=float, default=0.55)
    parser.add_argument("--lang", choices=("zh", "en"), default="zh", help="声纹模型语言")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="输出 token 上限；默认按每次窗口的实际长度推算")
    parser.add_argument("--runs-dir", default="runs/realtime")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> RealtimeConfig:
    return RealtimeConfig(
        window=args.window,
        hop=args.hop,
        tail=args.tail,
        silence_gate=args.silence_gate,
        speaker_threshold=args.speaker_threshold,
        max_new_tokens=args.max_new_tokens,
        poll_interval=args.poll_interval,
    )


def build_embedder(args: argparse.Namespace) -> Any:
    if not args.speaker:
        return None
    from moss_transcribe_diarize.realtime.speaker import OnnxCampplusEmbedder

    return OnnxCampplusEmbedder(model_path=args.speaker_model, lang=args.lang)


def build_transcriber_factory(args: argparse.Namespace, config: RealtimeConfig) -> Callable[[], Any]:
    """每次新会话调一次——`HfWindowTranscriber` 只能串行使用，每会话必须独立实例。"""
    if args.backend == "vllm":
        if not args.vllm_base_url:
            raise SystemExit("--backend vllm 需要 --vllm-base-url")
        from moss_transcribe_diarize.realtime.transcriber import VllmWindowTranscriber

        def make_vllm() -> Any:
            return VllmWindowTranscriber(
                base_url=args.vllm_base_url,
                model=args.vllm_model or args.model,
                api_key=args.vllm_api_key,
                token_budget_floor=config.effective_max_new_tokens(),
                max_new_tokens=args.max_new_tokens,
            )

        return make_vllm

    def make_hf() -> Any:
        import tempfile

        from moss_transcribe_diarize.app.model_runner import ModelRunner      # 只有这条路需要 torch
        from moss_transcribe_diarize.realtime.transcriber import HfWindowTranscriber

        runner = ModelRunner(args.model, device=args.device, dtype=args.dtype)
        scratch = Path(tempfile.mkdtemp(prefix="mtd-realtime-"))
        return HfWindowTranscriber(
            runner, scratch,
            token_budget_floor=config.effective_max_new_tokens(),
            max_new_tokens=args.max_new_tokens,
        )

    return make_hf


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    embedder = build_embedder(args)
    factory = build_transcriber_factory(args, config)

    from moss_transcribe_diarize.app.realtime_server import create_realtime_app

    app = create_realtime_app(
        config=config, transcriber_factory=factory, embedder=embedder, runs_dir=args.runs_dir
    )

    print(LOOPBACK_WARNING)
    print(f"后端 {args.backend}；打开 http://{args.host}:{args.port}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_cli.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 改 `pyproject.toml`**

在 `[project.scripts]` 加一行：

```toml
mtd-realtime = "moss_transcribe_diarize.app.realtime_cli:main"
```

在 `[project.optional-dependencies] realtime` 里补 `websockets`（uvicorn 的 WS 支持需要它）：

```toml
realtime = [
  "onnxruntime",
  "kaldi-native-fbank",
  "websockets",
]
```

并把那段注释改成现在仍然成立的说法：`websockets` 是服务端要的，装 `[realtime]` 就一起有了；只做离线声纹的人可以只装前两个。

- [ ] **Step 6: 重装并确认入口可用**

Run: `.venv/Scripts/python.exe -m pip install -e ".[dev,torch-runtime,realtime]" -q && .venv/Scripts/python.exe -m mtd_realtime --help 2>/dev/null || .venv/Scripts/mtd-realtime.exe --help`
Expected: 打出帮助，里面有 `127.0.0.1`（即 `parse_args` 那条测试要钉的东西在真入口上也成立）

- [ ] **Step 7: 全量 + 提交**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

```bash
git add moss_transcribe_diarize/app/realtime_cli.py pyproject.toml tests/test_realtime_cli.py
git commit -m "feat(app): add the mtd-realtime entry point

Two backends behind one flag. The hf backend's ModelRunner import lives inside
its factory so that --backend vllm never loads torch. Each session gets its own
transcriber instance because HfWindowTranscriber writes every window to one
scratch file and is therefore serial-only. The default bind is loopback and the
startup banner says why."
```

---

### Task 7: `scripts/realtime_client.py` 与端到端冒烟

无浏览器的冒烟客户端：把一个 wav 按**实时速度**推到 WebSocket，把收到的事件打到终端。它同时是这台服务的手工验证入口。

**Files:**
- Create: `scripts/realtime_client.py`
- Test: `tests/test_realtime_client.py`

**Interfaces:**
- Produces:
  - `chunk_pcm(pcm: np.ndarray, chunk_samples: int) -> Iterator[np.ndarray]`
  - `run(url: str, wav_path: Path, *, realtime: bool = True) -> dict`，返回末尾统计 `{"committed": int, "errors": int, "events": int}`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_client.py`：

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from realtime_client import chunk_pcm, wav_to_pcm16k  # noqa: E402


class ChunkPcmTest(unittest.TestCase):
    def test_splits_into_fixed_sized_chunks(self):
        chunks = list(chunk_pcm(np.arange(10, dtype=np.float32), 4))
        self.assertEqual([c.size for c in chunks], [4, 4, 2])

    def test_empty_input_gives_no_chunks(self):
        self.assertEqual(list(chunk_pcm(np.zeros(0, dtype=np.float32), 4)), [])

    def test_chunk_is_float32_little_endian_ready(self):
        chunk = next(chunk_pcm(np.array([1.5], dtype=np.float32), 1))
        self.assertEqual(chunk.dtype, np.float32)
        self.assertEqual(len(chunk.tobytes()), 4)


class WavLoadingTest(unittest.TestCase):
    def test_resamples_to_16k_mono(self):
        import soundfile as sf
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            sf.write(str(path), np.zeros(48000, dtype=np.float32), 48000)

            pcm = wav_to_pcm16k(path)

            self.assertEqual(pcm.shape, (16000,))
            self.assertEqual(pcm.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_client.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'realtime_client'`

- [ ] **Step 3: 实现 `scripts/realtime_client.py`**

```python
"""把一段 wav 按实时速度推给 mtd-realtime，并把事件打到终端。

    python scripts/realtime_client.py runs/demo-audio/meeting.wav
    python scripts/realtime_client.py file.wav --url ws://127.0.0.1:7870/ws/realtime --fast

没有浏览器也能验证整条链路，也是这台服务的手工验证入口。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf


def wav_to_pcm16k(path: str | Path, target_rate: int = 16000) -> np.ndarray:
    """读成 16 kHz 单声道 float32。重采样交给 soundfile/librosa 已有的实现。"""
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if rate == target_rate:
        return mono
    import soxr

    return np.asarray(soxr.resample(mono, rate, target_rate), dtype=np.float32)


def chunk_pcm(pcm: np.ndarray, chunk_samples: int) -> Iterator[np.ndarray]:
    for start in range(0, len(pcm), chunk_samples):
        yield pcm[start:start + chunk_samples]


def _drain(pcm, *, chunk_samples: int, realtime: bool, rate: int) -> None:
    """把分块按时间节奏吐出来——不在这个函数里做网络，好让节奏可单独测。"""
    interval = chunk_samples / rate
    for chunk in chunk_pcm(pcm, chunk_samples):
        yield chunk
        if realtime:
            time.sleep(interval)


async def run(url: str, wav_path: Path, *, realtime: bool = True, rate: int = 16000,
              chunk_ms: int = 100, connect=None) -> dict:
    pcm = wav_to_pcm16k(wav_path, rate)
    chunk_samples = int(rate * chunk_ms / 1000)
    stats = {"committed": 0, "errors": 0, "events": 0}

    if connect is None:                                  # 真的连；测试注入假的
        import websockets

        connect = websockets.connect

    async with connect(url) as socket:
        await socket.send(json.dumps({"type": "start", "session_name": wav_path.stem}))

        async def pump() -> None:
            for chunk in _drain(pcm, chunk_samples=chunk_samples, realtime=realtime, rate=rate):
                await socket.send(chunk.tobytes())

        async def listen() -> None:
            async for raw in socket:
                if isinstance(raw, bytes):
                    continue
                event = json.loads(raw)
                stats["events"] += 1
                kind = event.get("type")
                if kind == "committed":
                    stats["committed"] += len(event["segments"])
                    for seg in event["segments"]:
                        print("  + {:6.2f}-{:6.2f}  {}  {}".format(
                            seg["start"], seg["end"], seg["speaker_name"], seg["text"]))
                elif kind == "provisional":
                    body = "（空）" if not event["segments"] else " | ".join(
                        "{} {:.1f}-{:.1f}".format(s["speaker"], s["start"], s["end"])
                        for s in event["segments"])
                    print("  ~ 临时  " + body)
                elif kind == "status":
                    print("  · 状态  rtf={:.2f} 落后={:.1f}s 已跳过={}窗".format(
                        event["rtf"], event["lag_sec"], event.get("gated_windows", 0)))
                elif kind == "speaker":
                    print("  · 说话人  " + ", ".join(
                        "{}={}".format(s["id"], s["name"]) for s in event["speakers"]))
                elif kind == "error":
                    stats["errors"] += 1
                    print("  ! 错误  {} {}".format(event["code"], event["detail"]))

        listener = asyncio.create_task(listen())
        await pump()
        await asyncio.sleep(1.5)                         # 让最后几窗跑完
        await socket.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(1.0)
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="realtime_client")
    parser.add_argument("wav", type=Path, help="要推的音频文件")
    parser.add_argument("--url", default="ws://127.0.0.1:7870/ws/realtime")
    parser.add_argument("--fast", action="store_true", help="不按实时速度推，尽快推完")
    parser.add_argument("--chunk-ms", type=int, default=100)
    args = parser.parse_args(argv)

    stats = asyncio.run(run(args.url, args.wav, realtime=not args.fast, chunk_ms=args.chunk_ms))
    print("\n定稿 {} 段，事件 {} 条，错误 {} 次".format(
        stats["committed"], stats["events"], stats["errors"]))
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_client.py -q`
Expected: 全部 PASS

- [ ] **Step 5: 对真服务做一次端到端手工验证（本任务的核心交付）**

开两个终端。第一个：

```bash
.venv/Scripts/python.exe -m moss_transcribe_diarize.app.realtime_cli --backend hf --model runs/model --device cuda:0 --runs-dir runs/realtime
```

hmm — `python -m` 需要模块里有 `if __name__ == "__main__"` ✓ 有。

第二个（`--fast` 先跑一遍看通路，再按实时速度跑一遍看门控）：

```bash
.venv/Scripts/python.exe scripts/realtime_client.py runs/demo-audio/meeting.wav --fast
```

**预期**：终端打出 `+ 定稿 …` 若干行，最后一行是 `定稿 N 段，事件 M 条，错误 0 次`；`runs/realtime/<session-id>/` 下有 `session.json` / `transcript.jsonl` / `audio.wav` / `provisional.json`。

再把 `--fast` 去掉跑一遍（按实时速度，70 秒音频约 70 秒），确认状态事件里的 `已跳过=` 在中间那段静音里增长。

**把两次运行的终端输出贴进报告。**

- [ ] **Step 6: 全量 + 提交**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`

```bash
git add scripts/realtime_client.py tests/test_realtime_client.py
git commit -m "feat: add a no-browser smoke client for the realtime service

Pushes a wav over the websocket, either as fast as it can or paced to real time,
and prints the events as they arrive. This is how the service gets verified
without a microphone, and it is the seed of the phase 3 frontend's transport."
```

---

## 阶段二完成标志

```bash
.venv/Scripts/python.exe -m pytest tests/ -q                 # 全绿
.venv/Scripts/mtd-realtime --help                            # 入口存在
```

再加上 Task 7 Step 5 那两次手工运行都符合预期。

## 后续计划（阶段三）

浏览器前端：`audio-worklet.js`（AudioContext({sampleRate:16000}) + 双路混音 + `getUserMedia`/`getDisplayMedia` 采集）、`realtime.html`/`realtime.js`（定稿区 + 临时区两级渲染、说话人改名、rtf/积压显示）、i18n 词条、README 与真机调参。

阶段二的 `scripts/realtime_client.py` 已经定义了前端要说的那套协议，前端就是它的浏览器版。
