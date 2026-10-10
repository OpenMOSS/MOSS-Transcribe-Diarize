# 实时会议转写 — 阶段一：核心实时管线 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个纯 Python 的实时转写管线：喂进 16 kHz 单声道 PCM，吐出带绝对时间戳和全局说话人编号的定稿段落，以及一份随时可被替换的临时快照。

**Architecture:** 模型不是流式模型，所以"实时"由滑动窗口重复推理实现。本次任务实现管线本体：环形音频缓冲、窗口调度策略、把模型输出切分成定稿区与临时区的缝合器、跨窗口的说话人声纹匹配、会话编排与落盘。服务端和浏览器前端是后续两个独立计划。

**Tech Stack:** Python 3.10+、numpy、soundfile、pytest。本阶段不引入任何新依赖。

**Spec:** `docs/superpowers/specs/2026-09-24-realtime-meeting-transcription-design.md`

## 前置条件：Python 环境

这台机器目前**没有装好依赖的解释器**（只有 `D:\Tools\miniforge3` 的 base 环境，没有 torch、transformers、soundfile）。下面所有 `python -m pytest` 命令都要用建好的虚拟环境，否则第一步就会失败。

本阶段还需要一个**装上 torch 和 transformers 的环境**——不是因为实时管线要用它们，而是因为现有测试套件（`tests/test_app_api.py`、`tests/test_model_runner.py`、`tests/test_vllm_runner.py`）本来就需要。Task 1 改完之后，`moss_transcribe_diarize.realtime.*` 才真正与它们解耦。

```bash
D:/Tools/miniforge3/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install -U pip
.venv/Scripts/python.exe -m pip install -e ".[dev,torch-runtime]"
.venv/Scripts/python.exe -m pytest tests/ -q
```

最后一条命令应当在动手改任何代码之前就全绿。这是基线——如果它不绿，先把基线修好，否则后面分不清是新引入的问题还是本来就有的。

后续所有测试命令里的 `python` 都指 `.venv/Scripts/python.exe`：

```bash
.venv/Scripts/python.exe -m pytest tests/test_realtime_buffer.py -v
```

`.gitignore` 已经把这两件事都覆盖了：第 1 行 `.venv/`，第 13 行 `runs/`。所以虚拟环境和会话录音（`runs/realtime/<session-id>/audio.wav`）都不会被误提交，不需要额外处理。第 20 行的 `*.onnx` 也已忽略——阶段二的声纹模型默认缓存在 `~/.cache/mtd-speaker/`，不在仓库里。

## Global Constraints

- `requires-python = ">=3.10"`（`pyproject.toml` 现状）。新代码一律 `from __future__ import annotations` 开头，类型标注用 `X | None` 写法，与现有代码风格一致。
- **本阶段不引入任何新依赖。** 只用 `numpy` 和 `soundfile`（两者都已在 `pyproject.toml` 的 `dependencies` 里）。声纹嵌入通过 `SpeakerEmbedder` 协议注入，本阶段不实现 ONNX 版本，也不 import `onnxruntime` / `kaldi-native-fbank`。
- `ModelRunner`、`VllmRunner`、`app/server.py`、`app/jobs.py` 一行都不改。本阶段唯一允许修改的现有文件是 `moss_transcribe_diarize/__init__.py` 和 `moss_transcribe_diarize/inference_utils.py`，且只做 Task 1 描述的事。
- 音频约定：16 kHz、单声道、`np.float32`、取值范围 `[-1, 1]`。
- 时间约定：一律是单位为秒的 `float`。"绝对时间"指相对会话开始（第一帧音频）的秒数。
- `moss_transcribe_diarize/realtime/__init__.py` 保持只有模块 docstring，不做任何再导出。消费者直接 import 子模块（例如 `from moss_transcribe_diarize.realtime.session import RealtimeSession`）。这样每个模块的依赖关系在 import 语句里一眼可见，也避免 `__init__` 成为一个必须随每次改动更新的清单。
- 测试用 `unittest.TestCase` 风格，与现有 `tests/test_transcript_parser.py` 一致，用 `pytest` 运行。

## Review Focus

以下五类输入是 spec 没有明说、但实现必然会遇到的情况。每条都在拥有对应代码的那个 Task 里配了测试。

1. **窗口内没有任何有效音频**：会议刚开始时缓冲里只有零点几秒，或某个窗口全是被门控跳过的静音。空音频送进处理器会直接抛 `Audio must contain at least one sample`。管线必须在送进去之前就拦住。
2. **模型输出偏离约定格式**：漏掉说话人标签、时间戳顺序颠倒、夹杂解释性文字。解析器已经能丢弃不可解析的部分，缝合器必须保证**剩余段照常定稿**，而不是整窗丢弃。
3. **模型给出越界时间戳**：负数、或超过窗口时长的值。这类值一旦被写进 `committed_until` 水位线，水位线会被推到未来，之后**所有**段落都会被判定为"已定稿过"而永久丢失。水位线只能由已经裁剪过的、落在窗口内的段落推进。
4. **相邻窗口给出不同的分段边界**：同一句话在两次推理里被切成不同区间。结果里不能出现重复文字——重复比少量丢失更刺眼。
5. **说话人分组退化**：一个窗口里所有段都太短拿不到声纹嵌入，或所有段其实是同一个人。前者不能伪造归属，后者不能把说话人库撑成一堆单样本条目，也不能让库无上限增长。

---

### Task 1: torch 隔离（两处上游改动）

这一步必须最先做。不做的后果是：`import moss_transcribe_diarize.realtime.config` 会先执行父包 `__init__.py`，把 torch 和 transformers 一起拉起来——实时进程一行模型代码都不跑，却要为此装 2.5 GB 的 torch。同时 `DEFAULT_PROMPT` 现在住在 `inference_utils.py` 里，而那个模块顶层 `import torch`，实时管线需要这个字符串但不能为它拖进 torch。

**Files:**
- Create: `moss_transcribe_diarize/prompts.py`
- Modify: `moss_transcribe_diarize/__init__.py`（整体改写）
- Modify: `moss_transcribe_diarize/inference_utils.py:1-21`（`DEFAULT_PROMPT` 改为再导出）
- Test: `tests/test_package_lazy_import.py`

**Interfaces:**
- Consumes: 无（这是第一个 Task）
- Produces:
  - `moss_transcribe_diarize.prompts.DEFAULT_PROMPT: str` —— torch-free 的默认转写 prompt
  - `moss_transcribe_diarize.DEFAULT_PROMPT` / `.TranscriptSegment` / `.MossTranscribeDiarizeModel` 等全部现有名字继续可用，但改为惰性解析
  - 保证：`import moss_transcribe_diarize.realtime.config` 在 torch 被阻断的解释器里也能成功

- [ ] **Step 1: 写失败测试**

> **实现时的修正（以此为准）**：下面这段测试代码在实现过程中被修正过，实际交付内容见 `tests/test_package_lazy_import.py`。
>
> 原本规定的 `test_realtime_modules_import_without_torch` **无法失败**——它导入一个不存在的探针模块，然后断言 stderr 含 `ModuleNotFoundError` 且不含 `torch`。改动前父包因 `transformers` 被阻断而抛 `ModuleNotFoundError`（"transformers" 这个词里不含子串 "torch"），改动后因探针模块不存在而抛同样的错，所以它在 RED 和 GREEN 都通过，对这个任务的核心主张什么都没验证。
>
> 交付版把它换成 `test_realtime_import_path_is_torch_free`：在 torch/transformers 被阻断的子进程里导入**真实存在的**父包 `moss_transcribe_diarize`（这正是 `realtime.*` 导入最先执行的一步）并断言 `IMPORT_OK`；另加正向对照 `test_torch_backed_module_import_fails_when_blocked`，证明阻断确实生效、避免 `IMPORT_OK` 断言空转。`test_lazy_attribute_still_resolves` 也改为在全新解释器中运行并断言名字访问前不在 `vars(mtd)`、访问后被缓存进去，否则它是恒真断言。

创建 `tests/test_package_lazy_import.py`：

```python
from __future__ import annotations

import subprocess
import sys
import unittest


def _run_with_blocked(module: str, blocked: tuple[str, ...]) -> subprocess.CompletedProcess:
    """在子进程里导入 module，并让 blocked 里的顶层包不可用。"""
    code = (
        "import sys\n"
        f"for name in {blocked!r}:\n"
        "    sys.modules[name] = None\n"
        f"import {module}\n"
        "print('IMPORT_OK')\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


class LazyPackageImportTest(unittest.TestCase):
    def test_realtime_modules_import_without_torch(self):
        result = _run_with_blocked(
            "moss_transcribe_diarize.realtime.prompts_probe",
            ("torch", "transformers"),
        )
        # 该模块不存在，期望 ImportError 而不是 torch 相关的错误，
        # 说明父包 __init__ 没有把 torch 拉进来。
        self.assertNotIn("IMPORT_OK", result.stdout)
        self.assertIn("ModuleNotFoundError", result.stderr)
        self.assertNotIn("torch", result.stderr.lower())

    def test_prompts_module_imports_without_torch(self):
        result = _run_with_blocked(
            "moss_transcribe_diarize.prompts",
            ("torch", "transformers"),
        )
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)
        self.assertNotIn("torch", result.stderr.lower())

    def test_lazy_attribute_still_resolves(self):
        import moss_transcribe_diarize as mtd
        from moss_transcribe_diarize.transcript_parser import TranscriptSegment

        self.assertIs(mtd.TranscriptSegment, TranscriptSegment)
        self.assertIsInstance(mtd.DEFAULT_PROMPT, str)

    def test_unknown_attribute_raises(self):
        import moss_transcribe_diarize as mtd

        with self.assertRaises(AttributeError):
            mtd.definitely_not_a_real_name

    def test_all_names_are_importable(self):
        import moss_transcribe_diarize as mtd

        for name in mtd.__all__:
            self.assertTrue(hasattr(mtd, name), name)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_package_lazy_import.py -v`
Expected: `test_realtime_modules_import_without_torch` 和 `test_prompts_module_imports_without_torch` 失败——错误信息里出现 `torch`（因为 `__init__.py` 现在无条件 import `modeling_*`）。`test_lazy_attribute_still_resolves` 现在是通过的。

- [ ] **Step 3: 创建 `moss_transcribe_diarize/prompts.py`**

```python
"""Torch-free prompt constants shared by the inference and realtime paths."""

from __future__ import annotations

DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
```

字符串必须与 `inference_utils.py` 里原来的逐字一致——它已经过模型验证，改一个标点都可能影响输出格式。

- [ ] **Step 4: 让 `inference_utils.py` 改为再导出**

在 `moss_transcribe_diarize/inference_utils.py` 里，把原来 `DEFAULT_PROMPT = (...)` 那一段（第 13-17 行）替换为一行再导出，并加上 import：

```python
from moss_transcribe_diarize.prompts import DEFAULT_PROMPT
```

保留 `DEFAULT_PROMPT` 这个名字，`from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT` 这个现有用法不能破。

- [ ] **Step 5: 把 `__init__.py` 改成惰性导入**

整体改写 `moss_transcribe_diarize/__init__.py`：

```python
"""MOSS-Transcribe-Diarize: inference code and remote-code model implementation.

Public names are resolved lazily (PEP 562). Importing a lightweight submodule
such as ``moss_transcribe_diarize.prompts`` must not pull in torch and
transformers through this module, so the heavy names are imported only when
they are actually accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRIBUTES: dict[str, str] = {
    # torch-free
    "DEFAULT_PROMPT": "prompts",
    "TranscriptParseError": "transcript_parser",
    "TranscriptSegment": "transcript_parser",
    "TranscriptStreamParser": "transcript_parser",
    "iter_transcript_segments": "transcript_parser",
    "parse_transcript": "transcript_parser",
    "SubtitleSegment": "subtitle",
    "SubtitleStyle": "subtitle",
    "coerce_subtitle_segments": "subtitle",
    "export_ass": "subtitle",
    "export_json": "subtitle",
    "export_srt": "subtitle",
    "normalize_segments": "subtitle",
    "subtitle_segments_from_transcript": "subtitle",
    # requires torch
    "MossTranscribeDiarizeConfig": "configuration_moss_transcribe_diarize",
    "MossTranscribeDiarizeForConditionalGeneration": "modeling_moss_transcribe_diarize",
    "MossTranscribeDiarizeModel": "modeling_moss_transcribe_diarize",
    "MossTranscribeDiarizePreTrainedModel": "modeling_moss_transcribe_diarize",
    "MossTranscribeDiarizeProcessor": "processing_moss_transcribe_diarize",
    "VQAdaptor": "modeling_moss_transcribe_diarize",
}

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

`DEFAULT_PROMPT` 这个公开名字原来不在 `__init__.py` 的 `__all__` 里，现在加进去是顺手补的一个小缺口；它本来就是公开常量。

- [ ] **Step 6: 运行新测试与全量回归**

Run: `python -m pytest tests/test_package_lazy_import.py -v`
Expected: 全部 PASS

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS。这一步是这次改动的主要风险控制——`tests/test_transcript_parser.py`、`tests/test_subtitle_export.py`、`tests/test_app_api.py`、`tests/test_vllm_runner.py` 覆盖了所有受影响的 import 路径。

- [ ] **Step 7: 提交**

```bash
git add moss_transcribe_diarize/__init__.py moss_transcribe_diarize/prompts.py moss_transcribe_diarize/inference_utils.py tests/test_package_lazy_import.py
git commit -m "refactor: resolve package exports lazily so lightweight imports skip torch

moss_transcribe_diarize/__init__.py imported modeling_* and processing_*
unconditionally, both of which import torch at module level. Any import of a
lightweight submodule therefore dragged in torch and transformers. Switch to
PEP 562 lazy attribute resolution and move DEFAULT_PROMPT to a torch-free
prompts module that inference_utils re-exports."
```

---

### Task 2: `RealtimeConfig`

**Files:**
- Create: `moss_transcribe_diarize/realtime/__init__.py`
- Create: `moss_transcribe_diarize/realtime/config.py`
- Test: `tests/test_realtime_config.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `RealtimeConfig` —— 冻结 dataclass，字段与默认值见下。后续所有 Task 都从它取值。
  - `RealtimeConfig.effective_max_new_tokens() -> int`
  - `moss_transcribe_diarize.realtime` 包（`__init__.py` 只有 docstring）

- [ ] **Step 1: 创建包骨架**

`moss_transcribe_diarize/realtime/__init__.py`：

```python
"""Realtime meeting transcription pipeline.

Submodules are imported directly (``from ...realtime.session import
RealtimeSession``) rather than re-exported here, so that each module's
dependencies stay visible at its import site.
"""
```

- [ ] **Step 2: 写失败测试**

创建 `tests/test_realtime_config.py`：

```python
from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.config import RealtimeConfig


class RealtimeConfigTest(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = RealtimeConfig()

        self.assertEqual(config.window, 20.0)
        self.assertEqual(config.hop, 5.0)
        self.assertEqual(config.tail, 6.0)
        self.assertEqual(config.sample_rate, 16000)
        self.assertTrue(config.silence_gate)

    def test_effective_max_new_tokens_scales_with_window(self):
        self.assertEqual(RealtimeConfig(window=20.0).effective_max_new_tokens(), 1020)
        self.assertEqual(RealtimeConfig(window=40.0).effective_max_new_tokens(), 2040)

    def test_effective_max_new_tokens_has_a_floor(self):
        # window=1.0 必须同时给出 hop 与 tail：默认 hop=5.0 会让 __post_init__
        # 先以 hop > window 拒绝这个配置，断言根本执行不到。
        config = RealtimeConfig(window=1.0, hop=1.0, tail=0.0)

        self.assertEqual(config.effective_max_new_tokens(), 256)

    def test_explicit_max_new_tokens_wins(self):
        config = RealtimeConfig(window=20.0, max_new_tokens=77)

        self.assertEqual(config.effective_max_new_tokens(), 77)

    def test_rejects_hop_longer_than_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=10.0, hop=20.0)

    def test_rejects_tail_at_or_past_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=10.0, hop=5.0, tail=10.0)

    def test_rejects_buffer_too_small_for_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=20.0, hop=5.0, buffer_capacity=25.0)

    def test_rejects_out_of_range_similarity_threshold(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(speaker_threshold=1.5)

    def test_rejects_out_of_range_silence_ratio(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(silence_frame_ratio=0.0)

    def test_config_is_frozen(self):
        config = RealtimeConfig()

        with self.assertRaises(Exception):
            config.window = 30.0


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_config.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.config'`

- [ ] **Step 4: 实现 `config.py`**

```python
"""Tunable parameters for the realtime transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass

# 输出 token 速率估算：默认 prompt 下约每音频秒 51 个 token。
# 这是给 max_new_tokens 定默认值的起点，真机实测后应据实调整。
TOKENS_PER_AUDIO_SECOND = 51
MIN_MAX_NEW_TOKENS = 256


@dataclass(frozen=True, slots=True)
class RealtimeConfig:
    """实时转写管线的全部可调参数。

    算力开销约等于 ``window / hop`` 倍实时，静音门控会再省下一部分。
    """

    window: float = 20.0
    """每次推理覆盖的音频长度（秒）。越长上下文越足、分离越准，算力线性增长。"""

    hop: float = 5.0
    """两次推理之间的最小间隔（秒）。越小越跟手，算力线性增长。"""

    tail: float = 6.0
    """窗口末尾多少秒判为不稳定。越大定稿越稳、延迟越高。"""

    min_first_window: float = 8.0
    """首次推理前至少要攒够多少秒音频。"""

    sample_rate: int = 16000

    buffer_capacity: float = 180.0
    """环形缓冲容量（秒）。必须不小于 window + 2 * hop。"""

    silence_gate: bool = True
    silence_rms_db: float = -45.0
    silence_frame_ratio: float = 0.05

    speaker_threshold: float = 0.55
    min_segment_sec: float = 0.4
    """短于此长度的段落不做声纹，继承同组判定。"""

    max_new_tokens: int | None = None
    """None 表示按 window 推算，见 effective_max_new_tokens。"""

    poll_interval: float = 0.5
    """编排层检查是否该跑窗口的间隔（秒）。"""

    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.hop <= 0:
            raise ValueError("hop must be positive")
        if self.hop > self.window:
            raise ValueError("hop must not exceed window")
        if not 0.0 <= self.tail < self.window:
            raise ValueError("tail must be in [0, window)")
        if self.min_first_window <= 0:
            raise ValueError("min_first_window must be positive")
        if self.buffer_capacity < self.window + 2 * self.hop:
            raise ValueError("buffer_capacity must be at least window + 2 * hop")
        if not 0.0 < self.silence_frame_ratio <= 1.0:
            raise ValueError("silence_frame_ratio must be in (0, 1]")
        if not -120.0 <= self.silence_rms_db <= 0.0:
            raise ValueError("silence_rms_db must be in [-120, 0]")
        if not 0.0 <= self.speaker_threshold <= 1.0:
            raise ValueError("speaker_threshold must be in [0, 1]")
        if self.min_segment_sec < 0:
            raise ValueError("min_segment_sec must not be negative")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive when set")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self.max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")

    def effective_max_new_tokens(self) -> int:
        """窗口越长需要的输出 token 越多；未显式设置时按窗口秒数推算。"""
        if self.max_new_tokens is not None:
            return self.max_new_tokens
        return max(MIN_MAX_NEW_TOKENS, int(round(self.window * TOKENS_PER_AUDIO_SECOND)))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_config.py -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add moss_transcribe_diarize/realtime/__init__.py moss_transcribe_diarize/realtime/config.py tests/test_realtime_config.py
git commit -m "feat(realtime): add RealtimeConfig with validated tuning knobs"
```

---

### Task 3: `AudioRingBuffer`

**Files:**
- Create: `moss_transcribe_diarize/realtime/buffer.py`
- Test: `tests/test_realtime_buffer.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `AudioRingBuffer(capacity_seconds: float = 180.0, sample_rate: int = 16000)`
  - `.append(pcm: np.ndarray) -> None` —— 接受任意长度 float32 一维数组，超出容量的最旧数据被丢弃
  - `.slice(start_sec: float, end_sec: float) -> np.ndarray | None` —— 请求区间有任意部分已被丢弃时返回 `None`
  - `.total_seconds: float`（属性）、`.total_samples: int`（属性）、`.available_seconds: float`（属性）、`.sample_rate: int`（属性）
  - 线程安全：接收音频在事件循环线程，推理在工作线程，内部用 `threading.Lock`

关键不变量：逻辑索引 `i` 恒定存放在物理位置 `i % capacity`。因为环形缓冲始终按逻辑顺序连续写入，回绕不会破坏这个映射。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_buffer.py`：

```python
from __future__ import annotations

import threading
import unittest

import numpy as np

from moss_transcribe_diarize.realtime.buffer import AudioRingBuffer


def _ramp(start: float, count: int) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.float32)


class AudioRingBufferTest(unittest.TestCase):
    def test_slice_returns_appended_audio(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 5))

        np.testing.assert_allclose(buffer.slice(0.0, 0.5), _ramp(0, 5))

    def test_slice_uses_seconds_not_samples(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 10))

        np.testing.assert_allclose(buffer.slice(0.2, 0.5), _ramp(2, 3))

    def test_total_and_available_seconds(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 6))

        self.assertAlmostEqual(buffer.total_seconds, 0.6)
        self.assertAlmostEqual(buffer.available_seconds, 0.6)

    def test_wrap_preserves_logical_order(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 7))
        buffer.append(_ramp(7, 6))  # 总量 13，容量 10，最旧 3 个样本被丢弃

        self.assertAlmostEqual(buffer.available_seconds, 1.0)
        np.testing.assert_allclose(buffer.slice(0.3, 1.3), _ramp(3, 10))

    def test_slice_before_retained_head_returns_none(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 13))

        self.assertIsNone(buffer.slice(0.0, 0.5))
        self.assertIsNotNone(buffer.slice(0.3, 1.0))

    def test_slice_clamps_at_total(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 5))

        np.testing.assert_allclose(buffer.slice(0.0, 99.0), _ramp(0, 5))

    def test_slice_with_inverted_range_returns_none(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 5))

        self.assertIsNone(buffer.slice(0.4, 0.2))

    def test_slice_past_total_after_head_returns_none(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 5))

        self.assertIsNone(buffer.slice(0.6, 0.9))

    def test_oversized_append_keeps_only_the_tail(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(_ramp(0, 25))

        self.assertAlmostEqual(buffer.total_seconds, 2.5)
        self.assertAlmostEqual(buffer.available_seconds, 1.0)
        np.testing.assert_allclose(buffer.slice(1.5, 2.5), _ramp(15, 10))

    def test_empty_append_is_a_noop(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append(np.zeros(0, dtype=np.float32))

        self.assertEqual(buffer.total_samples, 0)

    def test_append_accepts_non_float32_input(self):
        buffer = AudioRingBuffer(capacity_seconds=1.0, sample_rate=10)
        buffer.append([0.0, 0.5, 1.0])

        np.testing.assert_allclose(buffer.slice(0.0, 0.3), np.array([0.0, 0.5, 1.0], dtype=np.float32))

    def test_concurrent_append_and_slice(self):
        buffer = AudioRingBuffer(capacity_seconds=2.0, sample_rate=100)
        errors: list[BaseException] = []

        def writer():
            try:
                for _ in range(200):
                    buffer.append(np.ones(50, dtype=np.float32))
            except BaseException as exc:  # pragma: no cover - 只在失败时触发
                errors.append(exc)

        def reader():
            try:
                for _ in range(200):
                    window = buffer.slice(0.5, 1.5)
                    if window is not None:
                        self.assertEqual(window.dtype, np.float32)
            except BaseException as exc:  # pragma: no cover - 只在失败时触发
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_buffer.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.buffer'`

- [ ] **Step 3: 实现 `buffer.py`**

```python
"""Fixed-capacity ring buffer for incoming 16 kHz mono audio."""

from __future__ import annotations

import threading

import numpy as np


class AudioRingBuffer:
    """保留最近 ``capacity_seconds`` 秒音频的环形缓冲。

    逻辑索引 ``i``（相对会话开始）恒定存放在物理位置 ``i % capacity``：缓冲区
    始终按逻辑顺序连续写入，回绕不破坏这个映射。
    """

    def __init__(self, capacity_seconds: float = 180.0, sample_rate: int = 16000):
        if capacity_seconds <= 0:
            raise ValueError("capacity_seconds must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._sample_rate = int(sample_rate)
        self._capacity = max(1, int(round(capacity_seconds * self._sample_rate)))
        self._data = np.zeros(self._capacity, dtype=np.float32)
        self._write = 0
        self._total = 0
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._total

    @property
    def total_seconds(self) -> float:
        """自会话开始收到的全部音频时长，包括已被淘汰的部分。"""
        return self.total_samples / self._sample_rate

    @property
    def available_seconds(self) -> float:
        """当前仍可读取的音频时长。"""
        with self._lock:
            return min(self._total, self._capacity) / self._sample_rate

    def append(self, pcm: np.ndarray) -> None:
        """追加任意长度的音频；超出容量的最旧样本被丢弃。

        ``self._write`` 恒等于 ``self._total % self._capacity``，因此从它开始按
        逻辑顺序写入就维持了「逻辑索引 i 存放在物理位置 i % capacity」这一不变量
        ——即便单次追加超过容量、需要多圈回绕也一样。

        这里有两个数容易搞混，写错任何一个都会让 slice 静默返回错位样本（下游每个
        时间戳跟着错，且没有任何东西会发现）：
          - ``_total`` 按**收到的**样本数推进。它是会话时钟，不是保留量。若按截断后
            的长度推进，``_total`` 永远长不过 capacity，"已淘汰的头部"恒为 0，
            ``slice`` 就永远不会返回 ``None``。
          - 写入位置按**实际落盘的**长度走，且每一圈都要取模，不能拿
            ``_write + 收到的长度`` 直接当结束位置——超出 capacity 时那样会越界。
        把 ``_write`` 从 ``_total`` 派生（而不是自己累加）让不变量只有一个来源。
        """
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        count = arr.size
        if count == 0:
            return
        with self._lock:
            position = self._write
            written = 0
            while written < count:
                chunk = min(count - written, self._capacity - position)
                self._data[position : position + chunk] = arr[written : written + chunk]
                written += chunk
                position = (position + chunk) % self._capacity
            self._total += count
            self._write = self._total % self._capacity

    def slice(self, start_sec: float, end_sec: float) -> np.ndarray | None:
        """返回 ``[start_sec, end_sec)`` 的音频，区间已被淘汰时返回 ``None``。"""
        if end_sec <= start_sec:
            return None
        start = int(round(start_sec * self._sample_rate))
        end = int(round(end_sec * self._sample_rate))
        with self._lock:
            head = self._total - min(self._total, self._capacity)
            if start < head:
                return None
            start = max(start, 0)
            end = min(end, self._total)
            if end <= start:
                return None
            count = end - start
            offset = start % self._capacity
            out = np.empty(count, dtype=np.float32)
            if offset + count <= self._capacity:
                out[:] = self._data[offset : offset + count]
            else:
                head_len = self._capacity - offset
                out[:head_len] = self._data[offset:]
                out[head_len:] = self._data[: count - head_len]
            return out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_buffer.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add moss_transcribe_diarize/realtime/buffer.py tests/test_realtime_buffer.py
git commit -m "feat(realtime): add AudioRingBuffer for rolling 16 kHz audio"
```

---

### Task 4: `WindowPolicy`

**Files:**
- Create: `moss_transcribe_diarize/realtime/window.py`
- Test: `tests/test_realtime_window.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `WindowDecision` —— 冻结 dataclass，字段 `should_run: bool`、`start_sec: float`、`end_sec: float`
  - `WindowPolicy(*, window: float, hop: float, min_first_window: float)`
  - `.decide(*, total_seconds: float, last_run_sec: float | None) -> WindowDecision`

首次运行不等满 `window` 秒，而是等满 `min_first_window`，否则开场会有一段时间一片空白。首个窗口左边界是 0，本来也没有更早的音频可作上下文。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_window.py`：

```python
from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.window import WindowPolicy


def _policy(**kwargs) -> WindowPolicy:
    params = {"window": 20.0, "hop": 5.0, "min_first_window": 8.0}
    params.update(kwargs)
    return WindowPolicy(**params)


class WindowPolicyTest(unittest.TestCase):
    def test_waits_for_min_first_window(self):
        policy = _policy()

        self.assertFalse(policy.decide(total_seconds=7.9, last_run_sec=None).should_run)
        decision = policy.decide(total_seconds=8.0, last_run_sec=None)
        self.assertTrue(decision.should_run)
        self.assertEqual(decision.start_sec, 0.0)
        self.assertEqual(decision.end_sec, 8.0)

    def test_first_window_start_is_clamped_to_zero(self):
        policy = _policy(min_first_window=30.0)

        decision = policy.decide(total_seconds=30.0, last_run_sec=None)

        self.assertEqual(decision.start_sec, 0.0)

    def test_later_runs_are_throttled_by_hop(self):
        policy = _policy()

        self.assertFalse(policy.decide(total_seconds=32.4, last_run_sec=28.0).should_run)
        self.assertTrue(policy.decide(total_seconds=33.0, last_run_sec=28.0).should_run)

    def test_later_window_is_window_seconds_wide(self):
        policy = _policy()

        decision = policy.decide(total_seconds=60.0, last_run_sec=55.0)

        self.assertEqual(decision.start_sec, 40.0)
        self.assertEqual(decision.end_sec, 60.0)

    def test_later_window_start_is_clamped_at_zero(self):
        policy = _policy(window=40.0, hop=5.0)

        decision = policy.decide(total_seconds=25.0, last_run_sec=20.0)

        self.assertEqual(decision.start_sec, 0.0)
        self.assertEqual(decision.end_sec, 25.0)

    def test_rejects_hop_longer_than_window(self):
        with self.assertRaises(ValueError):
            _policy(window=10.0, hop=11.0)

    def test_rejects_non_positive_min_first_window(self):
        with self.assertRaises(ValueError):
            _policy(min_first_window=0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_window.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.window'`

- [ ] **Step 3: 实现 `window.py`**

```python
"""Decide when to run inference and over which span of audio."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowDecision:
    should_run: bool
    start_sec: float = 0.0
    end_sec: float = 0.0


class WindowPolicy:
    """纯逻辑的窗口调度策略。不做 IO，不持有状态。"""

    def __init__(self, *, window: float, hop: float, min_first_window: float):
        if window <= 0:
            raise ValueError("window must be positive")
        if hop <= 0:
            raise ValueError("hop must be positive")
        if hop > window:
            raise ValueError("hop must not exceed window")
        if min_first_window <= 0:
            raise ValueError("min_first_window must be positive")
        self.window = float(window)
        self.hop = float(hop)
        self.min_first_window = float(min_first_window)

    def decide(self, *, total_seconds: float, last_run_sec: float | None) -> WindowDecision:
        if last_run_sec is None:
            if total_seconds < self.min_first_window:
                return WindowDecision(False)
            # 首次运行的左边界恒为 0，**不能**套用下面那条统一公式：当
            # min_first_window > window 时，统一公式算出的左边界会落在 0 之后，
            # 而后续窗口都是 [t - window, t]，永远够不到开头——于是
            # [0, total_seconds - window) 这段音频不会被任何一次推理覆盖，被静默
            # 丢弃。转写看起来一切正常，只是开头少了内容，下游没有任何东西能发现。
            return WindowDecision(True, start_sec=0.0, end_sec=float(total_seconds))
        if total_seconds - last_run_sec < self.hop:
            return WindowDecision(False)
        return WindowDecision(
            True,
            start_sec=max(0.0, total_seconds - self.window),
            end_sec=float(total_seconds),
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_window.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add moss_transcribe_diarize/realtime/window.py tests/test_realtime_window.py
git commit -m "feat(realtime): add WindowPolicy for sliding-window scheduling"
```

---

### Task 5: `stitch` —— 解析与时间换算

本 Task 只做输入侧的规范化，不涉及定稿逻辑。产出三个模块级函数和一个 `Segment` 类型。

**Files:**
- Create: `moss_transcribe_diarize/realtime/stitch.py`
- Test: `tests/test_realtime_stitch.py`

**Interfaces:**
- Consumes: `moss_transcribe_diarize.transcript_parser.TranscriptStreamParser`（已有）
- Produces:
  - `Segment` —— 冻结 dataclass，字段 `start: float`、`end: float`、`speaker: str`、`text: str`、`window_id: int`。全部时间是绝对秒。
  - `MIN_SEGMENT_DURATION = 0.05`
  - `parse_window_segments(raw_text: str, window_start: float, window_id: int) -> list[Segment]`
  - `clamp_to_window(seg: Segment, window_start: float, window_end: float) -> Segment | None`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_stitch.py`：

```python
from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.stitch import (
    MIN_SEGMENT_DURATION,
    Segment,
    clamp_to_window,
    parse_window_segments,
)


class ParseWindowSegmentsTest(unittest.TestCase):
    def test_shifts_local_times_to_absolute(self):
        segments = parse_window_segments("[0.5][S01]你好[1.5]", window_start=40.0, window_id=3)

        self.assertEqual(
            segments,
            [Segment(start=40.5, end=41.5, speaker="S01", text="你好", window_id=3)],
        )

    def test_parses_multiple_segments(self):
        raw = "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.speaker for s in segments], ["S01", "S02"])
        self.assertEqual([s.text for s in segments], ["Welcome", "Ready"])

    def test_whitespace_separated_noise_does_not_block_later_segments(self):
        # 杂散文本由空白分隔时走 _after_end 的 _pending_after_end 分支：下一个 [ 到来
        # 即干净地 emit 本段，后续片段照常产出。这是"偏离格式不作废整次推理"的良性
        # 一侧。
        #
        # 注意这个良性结论**不能推广**——见下面那条无分隔的用例。区分二者是必要的：
        # 只有这一条的话，会从一个刻意选了良性输入的例子推出关于所有格式偏离的错误结论。
        raw = "垃圾开头 [0.5][S01]有效[1.5] [2.0][S02]也好[3.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["有效", "也好"])

    def test_brackets_inside_speech_are_preserved(self):
        # transcript_parser 有明确的契约（tests/test_transcript_parser.py 的
        # test_numeric_brackets_inside_text_are_preserved）：正文里的方括号是被刻意
        # 保留的，不是污染。任何"正文含方括号就丢弃"的判据都会误伤这条。
        #
        # 这条用例是防止重犯同一错误的回归守卫——它比看起来更重要：本阶段曾有人
        # （控制器）为了防污染加过一道方括号守卫，恰好会丢掉这类合法内容，而当时
        # 的测试全都用不含括号的中文正文，覆盖不到误伤路径。
        raw = "[0][S01]第[2024]年，编号[001]继续[4]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["第[2024]年，编号[001]继续"])

    def test_unseparated_noise_pollutes_a_segment_and_we_pass_it_through(self):
        # 没有空白分隔时，解析器把 [1.5] 折回正文，再把 [2.0] 当成该段的结束时间
        # （2.0 >= 0.5 故 _read_end 接受），于是本段变成
        # (0.5, 2.0, "S01", "有效[1.5]垃圾")——文字与结束时间双双被污染；而 [S02] 在
        # _READ_START 状态解析失败被 reset，"也好" 随之静默消失。
        #
        # 本模块原样透传，不做修补：解析器无法区分这种污染与它刻意保留的正文方括号
        # （见下面那条用例），任何"含括号就丢弃"的判据都会误伤合法内容。这段音频会由
        # 下一个窗口重新覆盖，所以是暂时性丢失。
        #
        # 钉住这个丑陋的行为是刻意的：后来者若想"顺手修一下"，会先看到自己拿什么去换。
        raw = "[0.5][S01]有效[1.5]垃圾[2.0][S02]也好[3.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["有效[1.5]垃圾"])
        self.assertEqual(segments[0].end, 2.0)

    def test_trailing_prose_after_the_last_segment_drops_that_segment(self):
        # 解析器在 [end] 之后遇到非空白字符会把 [end] 折回正文并退回"读取正文"
        # 状态，于是该片段永不闭合、close() 也不吐出它。窗口的最后一段因此会丢；
        # 下一个窗口会重新覆盖这段音频，所以是暂时性丢失而非永久丢失。
        raw = "这是模型的解释性文字 [0.5][S01]有效内容[1.5] 更多废话"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual(segments, [])

    def test_trailing_segment_without_end_is_dropped(self):
        raw = "[0.5][S01]完整[1.5][2.0][S02]缺尾巴"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["完整"])

    def test_inverted_timestamps_drop_the_segment(self):
        # 解析器只接受 end >= start；顺序颠倒时它把 [end] 折回正文、该片段不闭合，
        # 于是整段被丢弃。所以 parse_window_segments 里不需要交换分支——那会是一段
        # 永远执行不到的死代码（_parse_timestamp 也不产生负值）。
        raw = "[3.0][S01]顺序反了[1.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual(segments, [])

    def test_empty_text_yields_nothing(self):
        self.assertEqual(parse_window_segments("", window_start=0.0, window_id=0), [])


class ClampToWindowTest(unittest.TestCase):
    def test_segment_inside_window_is_unchanged(self):
        seg = Segment(start=5.0, end=7.0, speaker="S01", text="a", window_id=0)

        self.assertEqual(clamp_to_window(seg, 0.0, 20.0), seg)

    def test_segment_crossing_left_edge_is_truncated(self):
        seg = Segment(start=-1.0, end=3.0, speaker="S01", text="a", window_id=0)

        clamped = clamp_to_window(seg, 0.0, 20.0)

        self.assertEqual(clamped.start, 0.0)
        self.assertEqual(clamped.end, 3.0)
        self.assertEqual(clamped.text, "a")

    def test_segment_crossing_right_edge_is_dropped(self):
        seg = Segment(start=19.0, end=21.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_entirely_before_window_is_dropped(self):
        seg = Segment(start=-5.0, end=-1.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_entirely_after_window_is_dropped(self):
        seg = Segment(start=25.0, end=27.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_shorter_than_minimum_after_clamping_is_dropped(self):
        seg = Segment(
            start=-1.0,
            end=MIN_SEGMENT_DURATION / 2.0,
            speaker="S01",
            text="a",
            window_id=0,
        )

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_stitch.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.stitch'`

- [ ] **Step 3: 实现 `stitch.py` 的解析部分**

```python
"""Turn one window's raw model output into window-normalised segments."""

from __future__ import annotations

from dataclasses import dataclass, replace

from moss_transcribe_diarize.transcript_parser import TranscriptStreamParser

MIN_SEGMENT_DURATION = 0.05


@dataclass(frozen=True, slots=True)
class Segment:
    """一个转写段落。所有时间都是相对会话开始的绝对秒。"""

    start: float
    end: float
    speaker: str
    text: str
    window_id: int


def parse_window_segments(raw_text: str, window_start: float, window_id: int) -> list[Segment]:
    """解析模型输出，把窗口内的局部时间换算成绝对时间。

    解析器（``TranscriptStreamParser``）会丢弃它无法解析的部分，所以多数格式偏离
    只损失对应片段、不会让整次推理作废。

    两个必须知道的例外，都源于解析器的 ``_after_end``：``[end]`` 之后若跟着非空白的
    杂散文本，它会把 ``[end]`` 折回正文并退回"读取正文"状态。

    1. 流尾的杂散文本：该片段永不闭合，``close()`` 也不吐出它——窗口的最后一段丢失。
    2. 中间位置的杂散文本：紧随其后的 ``[时间戳]`` 会被当成**本片段**的结束时间，
       于是本片段文字被污染（正文里混入 ``[1.5]垃圾`` 之类）、结束时间被顶到下一段的
       起点，而那个 ``[Sxx]`` 在 ``_READ_START`` 状态下解析失败被丢弃，下一段随之
       静默消失。这比单纯丢失更糟。

    注意"后续片段照常存活"**只在杂散文本由空白分隔时成立**（空白走
    ``_pending_after_end`` 分支，下一个 ``[`` 到来即干净 emit）。无分隔时不成立。

    本模块**不**去修补这两种情况，因为修补不了：解析器把"正文里本来就有的方括号"
    和"被折回的时间戳"处理成同一种结果（都走 ``_after_end``，后面跟的都是任意文本），
    所以任何"正文含方括号就丢弃"之类的判据都会误伤合法内容——正文里的方括号是
    ``transcript_parser`` 有测试固定下来的刻意保留行为
    （``tests/test_transcript_parser.py`` 的 ``test_numeric_brackets_inside_text_are_preserved``）。
    误伤的代价是静默丢内容，比带可见 ``[1.5]`` 痕迹的污染更隐蔽。

    两种例外都由**下一个窗口重新覆盖该段音频**来恢复，属于暂时性丢失而非永久丢失。

    另：时间戳顺序颠倒时解析器同样不闭合该片段（``_read_end`` 只接受
    ``end >= start``），且 ``_parse_timestamp`` 不产生负值，所以这里不需要交换分支
    ——那会是一段永远执行不到的死代码。
    """
    parser = TranscriptStreamParser()
    local = list(parser.feed(raw_text))
    local.extend(parser.close())
    out: list[Segment] = []
    for item in local:
        out.append(
            Segment(
                start=window_start + float(item.start),
                end=window_start + float(item.end),
                speaker=item.speaker or "",
                text=item.text,
                window_id=window_id,
            )
        )
    return out


def clamp_to_window(seg: Segment, window_start: float, window_end: float) -> Segment | None:
    """把段落裁剪进窗口，或判定它应当被丢弃。

    跨出**左**边界的段落裁剪起点——模型偶尔会给出早于窗口起点的时间戳，裁剪比
    丢弃少丢内容。跨出**右**边界的段落直接丢弃——右边界就是"现在"，下一轮推理
    会重新看到它，而裁剪会让文字长度和时间跨度对不上。
    """
    if seg.end <= window_start or seg.start >= window_end:
        return None
    if seg.end > window_end:
        return None
    start = max(seg.start, window_start)
    if seg.end - start < MIN_SEGMENT_DURATION:
        return None
    if start == seg.start:
        return seg
    return replace(seg, start=start)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_stitch.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add moss_transcribe_diarize/realtime/stitch.py tests/test_realtime_stitch.py
git commit -m "feat(realtime): parse model output into absolute-time window segments"
```

---

### Task 6: `Stitcher` —— 定稿水位线与去重

**Files:**
- Modify: `moss_transcribe_diarize/realtime/stitch.py`（追加 `StitchResult` 和 `Stitcher`）
- Test: `tests/test_realtime_stitch.py`（追加 `StitcherTest`）

**Interfaces:**
- Consumes: Task 5 的 `Segment`、`parse_window_segments`、`clamp_to_window`
- Produces:
  - `StitchResult` —— dataclass，字段 `committed: list[Segment]`、`provisional: list[Segment]`
  - `Stitcher(*, window: float, tail: float)`
  - `.committed_until: float`（属性）、`.provisional: list[Segment]`（属性，返回副本）
  - `.ingest(*, window_start: float, window_end: float, raw_text: str, window_id: int) -> StitchResult`
  - `.flush() -> list[Segment]`

**水位线的两条铁律**（对应 Review Focus 第 3 条）：`committed_until` 只能由已经裁剪过、且落在窗口内的段落推进；跨水位线的段落一律丢弃，不推进水位线。

- [ ] **Step 1: 写失败测试**

在 `tests/test_realtime_stitch.py` 追加：

```python
from moss_transcribe_diarize.realtime.stitch import Stitcher


def _rawl(*triples) -> str:
    """把 (start, end, speaker, text) 四元组拼成模型输出格式。"""
    return "".join(f"[{s}][{sp}]{t}[{e}]" for s, e, sp, t in triples)


class StitcherTest(unittest.TestCase):
    def test_commits_segments_older_than_tail(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0,
            window_end=20.0,
            raw_text=_rawl((1.0, 3.0, "S01", "早")),
            window_id=0,
        )

        self.assertEqual([s.text for s in result.committed], ["早"])
        self.assertEqual(result.provisional, [])
        self.assertEqual(stitcher.committed_until, 3.0)

    def test_holds_back_segments_inside_tail(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0,
            window_end=20.0,
            raw_text=_rawl((16.0, 19.0, "S01", "新")),
            window_id=0,
        )

        self.assertEqual(result.committed, [])
        self.assertEqual([s.text for s in result.provisional], ["新"])
        self.assertEqual(stitcher.committed_until, 0.0)

    def test_provisional_is_replaced_not_appended(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((16.0, 19.0, "S01", "第一版")), window_id=0,
        )

        result = stitcher.ingest(
            window_start=5.0, window_end=25.0,
            raw_text=_rawl((20.5, 24.0, "S01", "第二版")), window_id=1,
        )

        self.assertEqual([s.text for s in result.provisional], ["第二版"])

    def test_already_committed_segment_is_not_recommitted(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((1.0, 3.0, "S01", "早")), window_id=0,
        )

        result = stitcher.ingest(
            window_start=5.0, window_end=25.0,
            raw_text=_rawl((6.0, 8.0, "S01", "早")), window_id=1,
        )

        self.assertEqual(result.committed, [])

    def test_segment_straddling_the_watermark_is_dropped(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((5.0, 10.0, "S01", "前半")), window_id=0,
        )

        result = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((8.0, 12.0, "S01", "重叠")), window_id=1,
        )

        self.assertEqual(result.committed, [])
        self.assertEqual(stitcher.committed_until, 10.0)

    def test_out_of_range_timestamp_never_advances_the_watermark(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((1.0, 3.0, "S01", "正常"), (5000.0, 9000.0, "S02", "未来")),
            window_id=0,
        )

        self.assertEqual([s.text for s in result.committed], ["正常"])
        self.assertEqual(stitcher.committed_until, 3.0)

    def test_later_window_can_still_commit_after_an_out_of_range_spike(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((5000.0, 9000.0, "S02", "未来")), window_id=0,
        )

        result = stitcher.ingest(
            window_start=5.0, window_end=25.0,
            raw_text=_rawl((6.0, 9.0, "S01", "后续正常")), window_id=1,
        )

        self.assertEqual([s.text for s in result.committed], ["后续正常"])

    def test_committed_segments_are_returned_in_time_order(self):
        stitcher = Stitcher(window=20.0, tail=0.0)

        result = stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((8.0, 9.0, "S02", "后"), (1.0, 2.0, "S01", "前")),
            window_id=0,
        )

        self.assertEqual([s.text for s in result.committed], ["前", "后"])

    def test_flush_promotes_remaining_provisional(self):
        stitcher = Stitcher(window=20.0, tail=6.0)
        stitcher.ingest(
            window_start=0.0, window_end=20.0,
            raw_text=_rawl((16.0, 19.0, "S01", "末尾")), window_id=0,
        )

        flushed = stitcher.flush()

        self.assertEqual([s.text for s in flushed], ["末尾"])
        self.assertEqual(stitcher.flush(), [])
        self.assertEqual(stitcher.committed_until, 19.0)

    def test_rejects_tail_at_or_past_window(self):
        with self.assertRaises(ValueError):
            Stitcher(window=10.0, tail=10.0)


class StitcherUnparseableOutputTest(unittest.TestCase):
    def test_garbage_text_does_not_block_valid_segments(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0,
            window_end=20.0,
            raw_text="抱歉，我无法处理这段音频。" + _rawl((2.0, 4.0, "S01", "有效")),
            window_id=0,
        )

        self.assertEqual([s.text for s in result.committed], ["有效"])

    def test_fully_unparseable_output_yields_nothing(self):
        stitcher = Stitcher(window=20.0, tail=6.0)

        result = stitcher.ingest(
            window_start=0.0, window_end=20.0, raw_text="没有任何时间戳", window_id=0,
        )

        self.assertEqual(result.committed, [])
        self.assertEqual(result.provisional, [])
        self.assertEqual(stitcher.committed_until, 0.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_stitch.py -v`
Expected: FAIL —— `ImportError: cannot import name 'Stitcher'`

- [ ] **Step 3: 实现 `Stitcher`**

在 `moss_transcribe_diarize/realtime/stitch.py` 末尾追加：

```python
@dataclass(slots=True)
class StitchResult:
    committed: list[Segment]
    provisional: list[Segment]


class Stitcher:
    """把逐次窗口推理的结果，切成定稿区与临时区。

    ``committed_until`` 是一条只进不退的水位线：只有严格位于它之后的段落才会
    被定稿。靠它天然去重——相邻窗口的定稿区在时间上重叠，但同一段内容只会被
    定稿一次。
    """

    def __init__(self, *, window: float, tail: float):
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0.0 <= tail < window:
            raise ValueError("tail must be in [0, window)")
        self.window = float(window)
        self.tail = float(tail)
        self._committed_until = 0.0
        self._provisional: list[Segment] = []

    @property
    def committed_until(self) -> float:
        return self._committed_until

    @property
    def provisional(self) -> list[Segment]:
        return list(self._provisional)

    def ingest(
        self,
        *,
        window_start: float,
        window_end: float,
        raw_text: str,
        window_id: int,
    ) -> StitchResult:
        candidates: list[Segment] = []
        for seg in parse_window_segments(raw_text, window_start, window_id):
            clamped = clamp_to_window(seg, window_start, window_end)
            if clamped is None:
                continue
            if clamped.end <= self._committed_until:
                continue
            if clamped.start < self._committed_until:
                # 跨水位线，几乎总是上一次推理已定稿内容的重叠部分。重复比少量
                # 丢失更刺眼，所以整段丢弃，且不推进水位线。
                continue
            candidates.append(clamped)
        candidates.sort(key=lambda item: (item.start, item.end))

        cutoff = window_end - self.tail
        committed = [seg for seg in candidates if seg.end <= cutoff]
        self._provisional = [seg for seg in candidates if seg.end > cutoff]
        if committed:
            self._committed_until = max(self._committed_until, committed[-1].end)
        return StitchResult(committed=committed, provisional=list(self._provisional))

    def flush(self) -> list[Segment]:
        """把临时区里剩下的段落全部定稿。会话结束时调用。"""
        remaining = self._provisional
        self._provisional = []
        if remaining:
            self._committed_until = max(self._committed_until, remaining[-1].end)
        return remaining
```

注意 `test_later_window_can_still_commit_after_an_out_of_range_spike` 为什么通过：越界的时间戳在 `clamp_to_window` 里就被丢弃（`seg.start >= window_end`），根本进不了候选列表，所以水位线不会被污染。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_stitch.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add moss_transcribe_diarize/realtime/stitch.py tests/test_realtime_stitch.py
git commit -m "feat(realtime): add Stitcher with commit watermark and dedup"
```

---

### Task 7: `SpeakerGallery` 与 `SpeakerEmbedder` 协议

**Files:**
- Create: `moss_transcribe_diarize/realtime/speaker.py`
- Test: `tests/test_realtime_speaker.py`

**Interfaces:**
- Consumes: Task 5 的 `Segment`
- Produces:
  - `SpeakerEmbedder` —— `runtime_checkable` Protocol，属性 `embedding_dim: int`，方法 `embed(audio: np.ndarray, sample_rate: int) -> np.ndarray | None`
  - `UNKNOWN_SPEAKER_ID = "U00"`
  - `Assignment` —— 冻结 dataclass，字段 `segment: Segment`、`speaker_id: str`、`confident: bool`
  - `SpeakerGallery(embedder, *, threshold=0.55, min_segment_sec=0.4, sample_rate=16000)`
  - `.assign(segments, audio_of) -> list[Assignment]`，`audio_of: Callable[[Segment], np.ndarray | None]`
  - `.rename(speaker_id: str, name: str) -> None`（未知 ID 抛 `KeyError`）
  - `.display_name(speaker_id: str) -> str`
  - `.speakers() -> list[dict]`，每项 `{"id", "name", "samples"}`
  - `.enabled: bool`（属性）

设计要点：模型自己输出的局部标签是强先验——同一窗口内标着同一个 `[S02]` 的段落几乎肯定是同一个人。所以按 `(window_id, 局部标签)` 分组处理，每组挑最长的那段算一次声纹，用这次的结果决定整组的全局归属。这既省算力，又避免同一个人在一个窗口内被拆成两个全局说话人。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_speaker.py`：

```python
from __future__ import annotations

import unittest

import numpy as np

from moss_transcribe_diarize.realtime.speaker import (
    UNKNOWN_SPEAKER_ID,
    SpeakerGallery,
)
from moss_transcribe_diarize.realtime.stitch import Segment


class FakeEmbedder:
    """按预设的向量表返回嵌入，未登记的输入返回 None。"""

    embedding_dim = 3

    def __init__(self, mapping: dict[bytes, list[float]], *, default=None):
        self.mapping = {key: np.array(value, dtype=np.float32) for key, value in mapping.items()}
        self.default = None if default is None else np.array(default, dtype=np.float32)

    def embed(self, audio, sample_rate):
        key = np.asarray(audio, dtype=np.float32).tobytes()
        return self.mapping.get(key, self.default)


def _voice(value: float, *, samples: int = 8000) -> np.ndarray:
    """一段足够长的假音频，用填充值区分身份。

    长度必须超过 SpeakerGallery 的 min_segment_sec 门槛（默认 0.4 秒 = 6400 样本），
    否则每个分组都拿不到嵌入，测试会静默退化成"未知说话人"而看不出为什么。
    想测试"太短所以拿不到嵌入"的场景，就显式传一个小的 samples。
    """
    return np.full(samples, value, dtype=np.float32)


def _seg(start: float, end: float, speaker: str, window_id: int, text: str = "x") -> Segment:
    return Segment(start=start, end=end, speaker=speaker, text=text, window_id=window_id)


class SpeakerGalleryTest(unittest.TestCase):
    def test_disabled_gallery_falls_back_to_local_labels(self):
        gallery = SpeakerGallery(None)

        self.assertFalse(gallery.enabled)
        assignments = gallery.assign([_seg(0.0, 1.0, "S02", 0)], lambda seg: None)

        self.assertEqual(assignments[0].speaker_id, "S02")
        self.assertFalse(assignments[0].confident)

    def test_first_speaker_gets_first_global_id(self):
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))

        assignments = gallery.assign([_seg(0.0, 1.0, "S02", 0)], lambda seg: voice)

        self.assertEqual(assignments[0].speaker_id, "S01")
        self.assertTrue(assignments[0].confident)

    def test_same_voice_across_windows_keeps_one_global_id(self):
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))
        audio_of = lambda seg: voice  # noqa: E731

        gallery.assign([_seg(0.0, 1.0, "S01", 0)], audio_of)
        assignments = gallery.assign([_seg(20.0, 21.0, "S01", 1)], audio_of)

        self.assertEqual(assignments[0].speaker_id, "S01")
        self.assertEqual(len(gallery.speakers()), 1)

    def test_different_voice_gets_a_new_global_id(self):
        first = _voice(1.0)
        second = _voice(2.0)
        embedder = FakeEmbedder(
            {first.tobytes(): [1.0, 0.0, 0.0], second.tobytes(): [0.0, 1.0, 0.0]}
        )
        gallery = SpeakerGallery(embedder)

        gallery.assign([_seg(0.0, 1.0, "S01", 0)], lambda seg: first)
        assignments = gallery.assign([_seg(20.0, 21.0, "S01", 1)], lambda seg: second)

        self.assertEqual(assignments[0].speaker_id, "S02")
        self.assertEqual(len(gallery.speakers()), 2)

    def test_local_labels_are_only_a_within_window_prior(self):
        """同一窗口里两个局部标签若声纹相同，应归到同一个全局说话人。"""
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))
        audio_of = lambda seg: voice  # noqa: E731

        assignments = gallery.assign(
            [_seg(0.0, 1.0, "S01", 0), _seg(2.0, 3.0, "S02", 0)], audio_of
        )

        self.assertEqual([a.speaker_id for a in assignments], ["S01", "S01"])

    def test_group_members_inherit_the_group_decision(self):
        """同窗口同局部标签的段落共享一次判定，包括太短而拿不到声纹的那段。"""
        # 两段都超过 min_segment_sec=0.1 的门槛（1600 样本），但 long_voice 更长，
        # 所以组代表是它；short_voice 的字节不在向量表里，一旦被选为组代表就会
        # 拿到 None，测试随即暴露选择逻辑写反了。
        long_voice = _voice(1.0, samples=2000)
        short_voice = _voice(2.0, samples=1700)
        gallery = SpeakerGallery(
            FakeEmbedder({long_voice.tobytes(): [1.0, 0.0, 0.0]}, default=None),
            min_segment_sec=0.1,
        )

        def audio_of(seg):
            return long_voice if seg.text == "long" else short_voice

        assignments = gallery.assign(
            [
                _seg(0.0, 5.0, "S03", 0, text="long"),
                _seg(6.0, 6.02, "S03", 0, text="short"),
            ],
            audio_of,
        )

        self.assertEqual([a.speaker_id for a in assignments], ["S01", "S01"])

    def test_unembeddable_group_is_marked_unconfident(self):
        gallery = SpeakerGallery(FakeEmbedder({}), min_segment_sec=0.1)

        assignments = gallery.assign([_seg(0.0, 5.0, "S01", 0)], lambda seg: _voice(9.0))

        self.assertEqual(assignments[0].speaker_id, UNKNOWN_SPEAKER_ID)
        self.assertFalse(assignments[0].confident)

    def test_too_short_segment_is_not_embedded(self):
        # 100 样本远低于 min_segment_sec=1.0 对应的 16000 样本门槛
        voice = _voice(1.0, samples=100)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}), min_segment_sec=1.0)

        assignments = gallery.assign([_seg(0.0, 0.2, "S01", 0)], lambda seg: voice)

        self.assertFalse(assignments[0].confident)

    def test_unknown_speaker_appears_in_the_roster(self):
        gallery = SpeakerGallery(FakeEmbedder({}), min_segment_sec=0.1)

        gallery.assign([_seg(0.0, 5.0, "S01", 0)], lambda seg: _voice(9.0))

        roster = {item["id"]: item for item in gallery.speakers()}
        self.assertIn(UNKNOWN_SPEAKER_ID, roster)

    def test_rename_and_display_name(self):
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))
        gallery.assign([_seg(0.0, 1.0, "S01", 0)], lambda seg: voice)

        self.assertEqual(gallery.display_name("S01"), "S01")
        gallery.rename("S01", "张总")
        self.assertEqual(gallery.display_name("S01"), "张总")

    def test_rename_unknown_speaker_raises(self):
        gallery = SpeakerGallery(None)

        with self.assertRaises(KeyError):
            gallery.rename("S99", "谁")

    def test_centroid_update_keeps_a_consistent_voice_matched(self):
        """质心按样本数滑动平均更新，同一嗓音重复出现不应分裂出新的说话人。"""
        voice = _voice(1.0)
        gallery = SpeakerGallery(FakeEmbedder({voice.tobytes(): [1.0, 0.0, 0.0]}))
        audio_of = lambda seg: voice  # noqa: E731

        for index in range(5):
            assignments = gallery.assign([_seg(float(index) * 10, float(index) * 10 + 1, "S01", index)], audio_of)
            self.assertEqual(assignments[0].speaker_id, "S01")

        roster = gallery.speakers()
        self.assertEqual(len(roster), 1)
        self.assertEqual(roster[0]["samples"], 5)

    def test_empty_input_returns_empty(self):
        gallery = SpeakerGallery(None)

        self.assertEqual(gallery.assign([], lambda seg: None), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_speaker.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.speaker'`

- [ ] **Step 3: 实现 `speaker.py`**

```python
"""Cross-window speaker identity via speaker embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .stitch import Segment

UNKNOWN_SPEAKER_ID = "U00"
UNKNOWN_SPEAKER_NAME = "未知"


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """声纹嵌入。实现方负责把任意长度音频压成一个定长向量。"""

    embedding_dim: int

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """返回嵌入，或在这段音频无法可靠嵌入时返回 ``None``。"""


@dataclass(frozen=True, slots=True)
class Assignment:
    segment: Segment
    speaker_id: str
    confident: bool


@dataclass(slots=True)
class _GalleryEntry:
    id: str
    centroid: np.ndarray
    samples: int = 1
    name: str = ""


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not np.isfinite(norm):
        return arr
    return arr / norm


class SpeakerGallery:
    """把窗口内的局部说话人标签映射到全局一致的说话人编号。

    模型输出的 ``[S01]``/``[S02]`` 只在单个窗口内自洽。跨窗口的一致性是靠声纹
    嵌入的余弦相似度匹配做出来的，局部标签只是"同一窗口内这些段落属于同一个人"
    这个先验。
    """

    def __init__(
        self,
        embedder: SpeakerEmbedder | None,
        *,
        threshold: float = 0.55,
        min_segment_sec: float = 0.4,
        sample_rate: int = 16000,
    ):
        self._embedder = embedder
        self._threshold = float(threshold)
        self._sample_rate = int(sample_rate)
        self._min_samples = max(1, int(round(min_segment_sec * self._sample_rate)))
        self._entries: dict[str, _GalleryEntry] = {}
        self._next_index = 1
        self._unknown_assignments = 0

    @property
    def enabled(self) -> bool:
        return self._embedder is not None

    def assign(
        self,
        segments: list[Segment],
        audio_of: Callable[[Segment], np.ndarray | None],
    ) -> list[Assignment]:
        if not segments:
            return []
        if self._embedder is None:
            return [Assignment(seg, seg.speaker, False) for seg in segments]

        groups: dict[tuple[int, str], list[Segment]] = {}
        order: list[tuple[int, str]] = []
        for seg in segments:
            key = (seg.window_id, seg.speaker)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(seg)

        decided: dict[tuple[int, str], tuple[str, bool]] = {}
        for key in order:
            decided[key] = self._decide_group(groups[key], audio_of)

        return [
            Assignment(seg, *decided[(seg.window_id, seg.speaker)]) for seg in segments
        ]

    def _decide_group(
        self,
        group: list[Segment],
        audio_of: Callable[[Segment], np.ndarray | None],
    ) -> tuple[str, bool]:
        best_audio: np.ndarray | None = None
        best_size = 0
        for seg in group:
            audio = audio_of(seg)
            if audio is None:
                continue
            size = int(np.asarray(audio).size)
            if size < self._min_samples:
                continue
            if size > best_size:
                best_audio, best_size = audio, size

        if best_audio is None:
            self._unknown_assignments += 1
            return UNKNOWN_SPEAKER_ID, False

        embedding = self._embedder.embed(best_audio, self._sample_rate)
        if embedding is None:
            self._unknown_assignments += 1
            return UNKNOWN_SPEAKER_ID, False

        return self._match(embedding), True

    def _match(self, embedding: np.ndarray) -> str:
        vec = _normalize(embedding)
        best_id: str | None = None
        best_score = -1.0
        for entry in self._entries.values():
            score = float(np.dot(entry.centroid, vec))
            if score > best_score:
                best_id, best_score = entry.id, score

        if best_id is not None and best_score >= self._threshold:
            entry = self._entries[best_id]
            entry.centroid = _normalize(entry.centroid * entry.samples + vec)
            entry.samples += 1
            return best_id

        new_id = f"S{self._next_index:02d}"
        self._next_index += 1
        self._entries[new_id] = _GalleryEntry(id=new_id, centroid=vec, samples=1)
        return new_id

    def rename(self, speaker_id: str, name: str) -> None:
        entry = self._entries.get(speaker_id)
        if entry is None:
            raise KeyError(speaker_id)
        entry.name = str(name).strip()

    def display_name(self, speaker_id: str) -> str:
        entry = self._entries.get(speaker_id)
        if entry is None:
            return UNKNOWN_SPEAKER_NAME if speaker_id == UNKNOWN_SPEAKER_ID else speaker_id
        return entry.name or entry.id

    def speakers(self) -> list[dict]:
        roster = [
            {"id": entry.id, "name": entry.name or entry.id, "samples": entry.samples}
            for entry in self._entries.values()
        ]
        if self._unknown_assignments:
            roster.append(
                {"id": UNKNOWN_SPEAKER_ID, "name": UNKNOWN_SPEAKER_NAME, "samples": self._unknown_assignments}
            )
        return roster
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_speaker.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add moss_transcribe_diarize/realtime/speaker.py tests/test_realtime_speaker.py
git commit -m "feat(realtime): add SpeakerGallery for cross-window speaker identity"
```

---

### Task 8: `WindowTranscriber` 协议与 `HfWindowTranscriber`

**Files:**
- Create: `moss_transcribe_diarize/realtime/transcriber.py`
- Test: `tests/test_realtime_transcriber.py`

**Interfaces:**
- Consumes: 无（`HfWindowTranscriber` 通过构造函数接收一个 runner 对象，不 import `model_runner`）
- Produces:
  - `WindowTranscriber` —— `runtime_checkable` Protocol，方法 `transcribe_window(audio: np.ndarray, *, prompt: str) -> str`
  - `HfWindowTranscriber(runner, scratch_dir, *, sample_rate=16000, max_new_tokens=1020, decoding="greedy")`

`runner` 只要求有一个 `transcribe(path, *, prompt, max_new_tokens, decoding)` 方法、返回带 `.text` 属性的对象——这正是现有 `ModelRunner` 的形状。测试用假 runner，所以本 Task 完全不碰 torch。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_transcriber.py`：

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from moss_transcribe_diarize.realtime.transcriber import (
    HfWindowTranscriber,
    WindowTranscriber,
)


class _Result:
    def __init__(self, text: str):
        self.text = text


class FakeRunner:
    def __init__(self, text: str = "[0.5][S01]你好[1.5]"):
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, audio_path, *, prompt, max_new_tokens, decoding):
        self.calls.append(
            {
                "path": Path(audio_path),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "decoding": decoding,
            }
        )
        return _Result(self.text)


class HfWindowTranscriberTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name) / "scratch"

    def test_is_a_window_transcriber(self):
        transcriber = HfWindowTranscriber(FakeRunner(), self.scratch)

        self.assertIsInstance(transcriber, WindowTranscriber)

    def test_returns_runner_text(self):
        transcriber = HfWindowTranscriber(FakeRunner("结果"), self.scratch)

        self.assertEqual(
            transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="p"),
            "结果",
        )

    def test_creates_the_scratch_directory(self):
        HfWindowTranscriber(FakeRunner(), self.scratch)

        self.assertTrue(self.scratch.is_dir())

    def test_writes_a_readable_16k_wav(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch)
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

        transcriber.transcribe_window(audio, prompt="p")

        written, rate = sf.read(str(runner.calls[0]["path"]), dtype="float32")
        self.assertEqual(rate, 16000)
        self.assertEqual(written.shape, (16000,))
        self.assertAlmostEqual(float(np.max(np.abs(written - audio))), 0.0, places=4)

    def test_overwrites_the_same_file_each_call(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch)
        transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="p")
        first_size = runner.calls[0]["path"].stat().st_size

        transcriber.transcribe_window(np.zeros(800, dtype=np.float32), prompt="p")

        self.assertEqual(runner.calls[0]["path"], runner.calls[1]["path"])
        self.assertLess(runner.calls[1]["path"].stat().st_size, first_size)

    def test_forwards_prompt_and_budget(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch, max_new_tokens=777, decoding="sample")

        transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="自定义")

        self.assertEqual(runner.calls[0]["prompt"], "自定义")
        self.assertEqual(runner.calls[0]["max_new_tokens"], 777)
        self.assertEqual(runner.calls[0]["decoding"], "sample")

    def test_accepts_an_empty_window_without_crashing(self):
        """空窗口在管线里应被上游拦住，但这里也不能因为写文件而崩。"""
        runner = FakeRunner("")
        transcriber = HfWindowTranscriber(runner, self.scratch)

        self.assertEqual(
            transcriber.transcribe_window(np.zeros(0, dtype=np.float32), prompt="p"), ""
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_transcriber.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.transcriber'`

- [ ] **Step 3: 实现 `transcriber.py`**

```python
"""Window-level transcription backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import soundfile as sf


@runtime_checkable
class WindowTranscriber(Protocol):
    """把一个音频窗口转成模型原始输出文本。"""

    def transcribe_window(self, audio: np.ndarray, *, prompt: str) -> str: ...


class HfWindowTranscriber:
    """本地 HF 后端：窗口写临时 wav，交给 ModelRunner。

    ``runner`` 只需要有 ``transcribe(path, *, prompt, max_new_tokens, decoding)``
    方法并返回带 ``.text`` 的对象，所以测试可以注入假实现，不必加载 torch。
    """

    def __init__(
        self,
        runner,
        scratch_dir: str | Path,
        *,
        sample_rate: int = 16000,
        max_new_tokens: int = 1020,
        decoding: str = "greedy",
    ):
        self._runner = runner
        self._scratch_dir = Path(scratch_dir).expanduser()
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._scratch_dir / "window.wav"
        self._sample_rate = int(sample_rate)
        self._max_new_tokens = int(max_new_tokens)
        self._decoding = decoding

    @property
    def window_path(self) -> Path:
        return self._path

    def transcribe_window(self, audio: np.ndarray, *, prompt: str) -> str:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        sf.write(str(self._path), arr, self._sample_rate, format="WAV", subtype="PCM_16")
        result = self._runner.transcribe(
            self._path,
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            decoding=self._decoding,
        )
        return result.text
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_transcriber.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add moss_transcribe_diarize/realtime/transcriber.py tests/test_realtime_transcriber.py
git commit -m "feat(realtime): add WindowTranscriber protocol and HF window backend"
```

---

### Task 9: `SessionStore`

**Files:**
- Create: `moss_transcribe_diarize/realtime/store.py`
- Test: `tests/test_realtime_store.py`

**Interfaces:**
- Consumes: `CommittedSegment` 的 `to_dict()` 和 `Segment` 的字段（Task 10 定义 `CommittedSegment`；本 Task 的测试用一个最小的鸭子类型替身，两者字段一致）
- Produces:
  - `SessionStore(runs_dir, session_id=None, *, sample_rate=16000, record_audio=True, name="")`
  - 属性：`.session_id`、`.dir`、`.audio_path`、`.transcript_path`、`.provisional_path`、`.meta_path`
  - `.append_audio(pcm) -> None`
  - `.append_committed(segments) -> None`
  - `.write_provisional(segments) -> None`
  - `.write_meta(**fields) -> None`
  - `.finalize(speakers, *, status="done") -> None`
  - `SessionStore.list_sessions(runs_dir) -> list[dict]`
  - `SessionStore.session_dir(runs_dir, session_id) -> Path`
  - `SessionStore.load_committed(runs_dir, session_id) -> list[dict]`

**已知局限（记录在案，不在本 Task 解决）**：`audio.wav` 以流式写入。进程崩溃时 WAV 头里的长度字段会是过期值，音频内容本身还在文件里但需要修复头部才能播。`transcript.jsonl` 才是崩溃恢复的权威来源；音频恢复是尽力而为。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_store.py`：

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from moss_transcribe_diarize.realtime.stitch import Segment
from moss_transcribe_diarize.realtime.store import SessionStore


class FakeCommitted:
    """与 CommittedSegment 字段一致的鸭子类型替身。"""

    def __init__(self, id, start, end, speaker_id, speaker_name, text, confident):
        self.id = id
        self.start = start
        self.end = end
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.text = text
        self.confident = confident

    def to_dict(self):
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "speaker_confident": self.confident,
        }


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _store(self, **kwargs) -> SessionStore:
        return SessionStore(self.runs, "sess-1", **kwargs)

    def test_creates_the_session_directory_and_meta(self):
        store = self._store()

        self.assertTrue(store.dir.is_dir())
        self.assertEqual(store.dir, self.runs / "sess-1")
        meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["session_id"], "sess-1")
        self.assertEqual(meta["status"], "recording")

    def test_audio_is_readable_after_finalize(self):
        store = self._store()
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        store.append_audio(audio)

        store.finalize([])

        written, rate = sf.read(str(store.audio_path), dtype="float32")
        self.assertEqual(rate, 16000)
        self.assertEqual(written.shape, (16000,))

    def test_audio_appends_across_calls(self):
        store = self._store()
        store.append_audio(np.zeros(800, dtype=np.float32))
        store.append_audio(np.zeros(400, dtype=np.float32))

        store.finalize([])

        written, _ = sf.read(str(store.audio_path), dtype="float32")
        self.assertEqual(written.shape, (1200,))

    def test_no_audio_file_when_recording_is_disabled(self):
        store = self._store(record_audio=False)
        store.append_audio(np.zeros(800, dtype=np.float32))

        store.finalize([])

        self.assertFalse(store.audio_path.exists())

    def test_append_after_finalize_is_ignored(self):
        store = self._store()
        store.finalize([])

        store.append_audio(np.zeros(800, dtype=np.float32))

        self.assertFalse(store.audio_path.exists())

    def test_committed_segments_are_appended_as_jsonl(self):
        store = self._store()
        store.append_committed([FakeCommitted("seg-1", 0.0, 1.0, "S01", "张总", "你好", True)])
        store.append_committed([FakeCommitted("seg-2", 1.0, 2.0, "S02", "S02", "再见", False)])

        rows = SessionStore.load_committed(self.runs, "sess-1")

        self.assertEqual([row["id"] for row in rows], ["seg-1", "seg-2"])
        self.assertEqual(rows[0]["speaker_name"], "张总")
        self.assertEqual(rows[1]["speaker"], "S02")

    def test_load_committed_skips_corrupt_lines(self):
        store = self._store()
        store.append_committed([FakeCommitted("seg-1", 0.0, 1.0, "S01", "S01", "好", True)])
        with store.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write("{ 这不是 JSON\n")

        rows = SessionStore.load_committed(self.runs, "sess-1")

        self.assertEqual([row["id"] for row in rows], ["seg-1"])

    def test_provisional_snapshot_round_trips(self):
        store = self._store()
        store.write_provisional([Segment(1.0, 2.0, "S03", "临时的", 4)])

        payload = json.loads(store.provisional_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, [{"start": 1.0, "end": 2.0, "speaker": "S03", "text": "临时的"}])

    def test_finalize_records_speakers_and_end_time(self):
        store = self._store()
        store.finalize([{"id": "S01", "name": "张总", "samples": 3}], status="done")

        meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "done")
        self.assertEqual(meta["speakers"], [{"id": "S01", "name": "张总", "samples": 3}])
        self.assertIn("ended_at", meta)

    def test_finalize_is_idempotent(self):
        store = self._store()
        store.finalize([], status="done")

        store.finalize([], status="failed")

        meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "done")

    def test_write_meta_preserves_existing_fields(self):
        store = self._store()
        store.write_meta(name="周会")

        store.write_meta(status="running")

        meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["name"], "周会")
        self.assertEqual(meta["status"], "running")

    def test_generated_session_id_is_unique(self):
        first = SessionStore(self.runs)
        second = SessionStore(self.runs)

        self.assertNotEqual(first.session_id, second.session_id)

    def test_list_sessions_returns_newest_first(self):
        SessionStore(self.runs, "a").write_meta(started_at=1.0)
        SessionStore(self.runs, "b").write_meta(started_at=2.0)

        listed = SessionStore.list_sessions(self.runs)

        self.assertEqual([item["session_id"] for item in listed], ["b", "a"])

    def test_list_sessions_on_missing_directory_is_empty(self):
        self.assertEqual(SessionStore.list_sessions(self.runs / "nope"), [])

    def test_corrupt_meta_is_skipped_by_list_sessions(self):
        SessionStore(self.runs, "a").write_meta()
        (self.runs / "b").mkdir(parents=True)
        (self.runs / "b" / "session.json").write_text("{ 坏了", encoding="utf-8")

        listed = SessionStore.list_sessions(self.runs)

        self.assertEqual([item["session_id"] for item in listed], ["a"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_store.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.store'`

- [ ] **Step 3: 实现 `store.py`**

```python
"""On-disk persistence for realtime sessions."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from .stitch import Segment


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
        self.session_id = session_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
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
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            for item in segments:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

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
        # 刻意**不**接收 committed 段落：它们由 append_committed 在每个窗口追加（这样
        # 会话中途崩溃也能保住已产出的内容），本方法只负责收尾。若这里再写一遍调用方
        # 传进来的段落，每次会话结束时整个转写都会重复一遍。
        if self._closed:
            return
        self._closed = True
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        self.write_meta(status=status, ended_at=time.time(), speakers=list(speakers))

    @staticmethod
    def session_dir(runs_dir: str | Path, session_id: str) -> Path:
        return Path(runs_dir).expanduser() / session_id

    @staticmethod
    def load_committed(runs_dir: str | Path, session_id: str) -> list[dict]:
        path = SessionStore.session_dir(runs_dir, session_id) / "transcript.jsonl"
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
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
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_store.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add moss_transcribe_diarize/realtime/store.py tests/test_realtime_store.py
git commit -m "feat(realtime): add SessionStore for session persistence"
```

---

### Task 10: `RealtimeSession` 编排

这是把前面所有部件接起来的地方。

**Files:**
- Create: `moss_transcribe_diarize/realtime/session.py`
- Test: `tests/test_realtime_session.py`

> **实现时的修正（以此为准）**：计划本节的测试清单不是最终形态，实际交付见 `tests/test_realtime_session.py`（25 条）。差异有三处：
>
> 1. `test_status_event_reports_transcript_lag` 的期望值原为 `17.0`，**两种读法下都不成立**（读法 B 得 28.0；而计划注释自己写的 `[10,30]` 前提得 18.0，因段落 `[1][S01]a[2]` 是区间 (1,2)、在 `[10,30]` 窗口下落在绝对 (11,12) 而非 (11,13)）。已改为 `28.0` 并重写注释说明首窗是 `[0,30]`。
> 2. 补了 **6 条测试**，每条对应一个计划未覆盖的变异（`_last_run_sec` 推进时机、把环形缓冲而非窗口副本交给声纹库、窗口失败被当作致命、失败计数不复位、`provisional` 改为追加、`close()` 不重跑推理就提升临时段）。每条施加对应编辑后**恰好只有它自己失败**，已逐条实测。
> 3. `push_audio` 的空输入守卫**刻意不测**——它确实没有可观测后果（`buffer.py` 与 `store.py` 各自已守住空输入），删掉它不会改变任何行为。一条删掉无影响的守卫，如实报告比用一条无法失败的测试糊上更好。

**Interfaces:**
- Consumes: `RealtimeConfig`、`AudioRingBuffer`、`WindowPolicy`、`Stitcher`/`Segment`、`SpeakerGallery`/`SpeakerEmbedder`/`Assignment`、`WindowTranscriber`、`SessionStore`

  **注意**：`HfWindowTranscriber` 把窗口固定写到 `<scratch_dir>/window.wav`，所以**一个实例只能串行使用**——两个并发的 `transcribe_window` 会互相覆盖同一个文件，结果错乱而不会报错。本任务的用法是每个会话构造一个实例、按顺序调用，符合这个约束。将来若要做并行推理，必须改成每窗口一个独立路径（例如带上 `window_id`），不能共享实例。
- Produces:
  - `CommittedSegment` —— 冻结 dataclass，字段 `id`、`start`、`end`、`speaker_id`、`speaker_name`、`text`、`confident`；方法 `to_dict()` 返回 `{"id","start","end","speaker","speaker_name","text","speaker_confident"}`（与 Task 9 的测试替身字段一致）
  - `RealtimeSession(config, *, transcriber, store, embedder=None, prompt=DEFAULT_PROMPT)`
  - `.push_audio(pcm) -> None`（同步）
  - `.run_pending() -> list[dict]`（async，返回待发送的事件）
  - `.close() -> list[dict]`（async）
  - `.committed: list[CommittedSegment]`（属性）、`.closed: bool`（属性）

事件形状（spec 第 4.9 节）：

| type | 载荷 | 语义 |
|---|---|---|
| `committed` | `segments: list[dict]` | 增量追加 |
| `provisional` | `segments: list[dict]` | 整体替换 |
| `speaker` | `speakers: list[dict]` | 说话人清单 |
| `status` | `state`、`buffered_sec`、`rtf`、`last_window_ms`、`lag_sec`、`degraded` | 心跳 |
| `error` | `code`、`detail`、`failures` | 单次窗口失败，不终止会话 |

阻塞工作（HTTP 推理、声纹嵌入、文件写入）全部在 `run_in_executor` 的工作线程里做，不占事件循环。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_session.py`：

```python
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.session import RealtimeSession
from moss_transcribe_diarize.realtime.store import SessionStore

SILENCE = np.zeros(0, dtype=np.float32)  # 占位，实际音频由 _speech/_silence 生成


def _speech(seconds: float, *, amplitude: float = 0.3, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(seconds: float, *, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


class ScriptedTranscriber:
    """按调用次序返回预设文本，并记录每次收到的窗口时长。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.window_seconds: list[float] = []

    def transcribe_window(self, audio, *, prompt):
        self.window_seconds.append(len(audio) / 16000.0)
        if not self.replies:
            return ""
        return self.replies.pop(0)


class FailingTranscriber:
    def __init__(self, message: str = "boom"):
        self.message = message
        self.calls = 0

    def transcribe_window(self, audio, *, prompt):
        self.calls += 1
        raise RuntimeError(self.message)


def _config(**kwargs) -> RealtimeConfig:
    params = {
        "window": 20.0,
        "hop": 5.0,
        "tail": 6.0,
        "min_first_window": 8.0,
        "silence_gate": False,
    }
    params.update(kwargs)
    return RealtimeConfig(**params)


class RealtimeSessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, transcriber, config=None, **store_kwargs) -> RealtimeSession:
        return RealtimeSession(
            config or _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1", **store_kwargs),
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def test_does_not_run_before_min_first_window(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(5.0))

        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_empty_buffer_returns_no_events(self):
        """缓冲里一帧音频都没有时，绝不能把空音频送进处理器。"""
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)

        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_empty_push_is_ignored(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(np.zeros(0, dtype=np.float32))

        self.assertEqual(self._run(session.run_pending()), [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_first_window_runs_at_min_first_window(self):
        transcriber = ScriptedTranscriber(["[1][S01]开场[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        self.assertAlmostEqual(transcriber.window_seconds[0], 8.0, places=2)
        self.assertIn("committed", [event["type"] for event in events])

    def test_committed_segments_carry_ids_and_absolute_times(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual(committed["segments"][0]["id"], "seg-1")
        self.assertAlmostEqual(committed["segments"][0]["start"], 1.0, places=2)
        self.assertAlmostEqual(committed["segments"][0]["end"], 2.0, places=2)
        self.assertEqual(committed["segments"][0]["text"], "你好")

    def test_provisional_event_is_emitted_even_when_empty(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        provisional = next(event for event in events if event["type"] == "provisional")
        self.assertEqual(provisional["segments"], [])

    def test_late_segment_stays_provisional_then_commits_later(self):
        # 第一次窗口 [0,20]，段落落在 18-19 秒，在 tail=6 之内，所以只能是临时段。
        # 第二次窗口是 [10,30]，同一段内容此时落在窗口内的 8-9 秒处，换算回绝对
        # 时间仍是 18-19 秒，而 cutoff 是 30-6=24，于是被定稿。
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[8][S01]结尾[9]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        events = self._run(session.run_pending())
        self.assertEqual([e for e in events if e["type"] == "committed"], [])

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual([seg["text"] for seg in committed["segments"]], ["结尾"])
        self.assertAlmostEqual(committed["segments"][0]["start"], 18.0, places=2)
        self.assertAlmostEqual(committed["segments"][0]["end"], 19.0, places=2)

    def test_hop_throttles_repeated_runs(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())

        session.push_audio(_speech(2.0))
        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(len(transcriber.window_seconds), 1)

    def test_window_covers_at_most_window_seconds(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(60.0))

        self._run(session.run_pending())
        session.push_audio(_speech(10.0))
        self._run(session.run_pending())

        self.assertAlmostEqual(transcriber.window_seconds[1], 20.0, places=2)

    def test_records_audio_to_store(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())

        events = self._run(session.close())
        del events

        audio_path = self.runs / "s1" / "audio.wav"
        self.assertTrue(audio_path.exists())
        import soundfile as sf

        written, _ = sf.read(str(audio_path), dtype="float32")
        self.assertEqual(written.shape[0], 8 * 16000)

    def test_close_flushes_provisional_and_finalizes(self):
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[18][S01]结尾[19]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        self._run(session.run_pending())

        events = self._run(session.close())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual([seg["text"] for seg in committed["segments"]], ["结尾"])
        self.assertTrue(session.closed)

    def test_close_is_idempotent(self):
        transcriber = ScriptedTranscriber([])
        session = self._session(transcriber)

        self._run(session.close())

        self.assertEqual(self._run(session.close()), [])

    def test_push_after_close_is_ignored(self):
        transcriber = ScriptedTranscriber([])
        session = self._session(transcriber)
        self._run(session.close())

        session.push_audio(_speech(8.0))

        self.assertEqual(session.committed, [])

    def test_status_event_reports_transcript_lag(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(30.0))

        events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        # 首次运行的窗口是 [0, 30]：Ruling 15 的读法 B 规定首窗左边界恒为 0，不能套用
        # 后续窗口的 max(0, total - window)。段落局部 1-2 秒即绝对 1-2 秒，低于
        # cutoff = 30 - 6 = 24，故被定稿、水位线推到 2.0。字幕落后 = 缓冲 30 - 定稿 2 = 28。
        self.assertAlmostEqual(status["lag_sec"], 28.0, places=1)
        self.assertAlmostEqual(status["buffered_sec"], 30.0, places=1)
        self.assertIn("rtf", status)

    def test_single_failure_reports_error_and_keeps_going(self):
        transcriber = FailingTranscriber()
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(error["code"], "transcribe_failed")
        self.assertIn("boom", error["detail"])

    def test_repeated_failures_mark_degraded(self):
        transcriber = FailingTranscriber()
        session = self._session(transcriber)
        for _ in range(3):
            session.push_audio(_speech(10.0))
            events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertTrue(status["degraded"])

    def test_a_success_resets_the_failure_counter(self):
        class FlakyTranscriber:
            def __init__(self):
                self.calls = 0

            def transcribe_window(self, audio, *, prompt):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("第一次失败")
                return "[1][S01]恢复[2]"

        session = self._session(FlakyTranscriber())
        session.push_audio(_speech(10.0))
        self._run(session.run_pending())

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertFalse(status["degraded"])

    def test_speaker_event_lists_the_roster(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        speaker = next(event for event in events if event["type"] == "speaker")
        self.assertEqual(speaker["speakers"], [])

    def test_audio_of_uses_window_offsets_not_the_buffer(self):
        """段落音频按窗口内偏移切取，避免环形缓冲回绕后取错位置。"""
        class RecordingSpeakerEmbedder:
            embedding_dim = 2
            seen_sizes: list[int] = []

            def embed(self, audio, sample_rate):
                RecordingSpeakerEmbedder.seen_sizes.append(int(np.asarray(audio).size))
                return np.array([1.0, 0.0], dtype=np.float32)

        RecordingSpeakerEmbedder.seen_sizes = []
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = RealtimeSession(
            _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=RecordingSpeakerEmbedder(),
        )
        session.push_audio(_speech(8.0))

        self._run(session.run_pending())

        self.assertEqual(RecordingSpeakerEmbedder.seen_sizes, [16000])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_session.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'moss_transcribe_diarize.realtime.session'`

- [ ] **Step 3: 实现 `session.py`**

```python
"""Orchestration: buffer audio, run windows, commit stable segments."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np

from moss_transcribe_diarize.prompts import DEFAULT_PROMPT

from .buffer import AudioRingBuffer
from .config import RealtimeConfig
from .speaker import SpeakerEmbedder, SpeakerGallery
from .stitch import Segment, Stitcher
from .store import SessionStore
from .transcriber import WindowTranscriber
from .window import WindowPolicy


@dataclass(frozen=True, slots=True)
class CommittedSegment:
    id: str
    start: float
    end: float
    speaker_id: str
    speaker_name: str
    text: str
    confident: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "speaker_confident": self.confident,
        }


def segment_audio(
    window_audio: np.ndarray,
    window_start: float,
    seg: Segment,
    sample_rate: int,
) -> np.ndarray | None:
    """按窗口内偏移切出段落音频。

    刻意不从环形缓冲重新取——缓冲可能已经回绕并淘汰了这段音频，而窗口副本
    一定是完整的。
    """
    total = int(window_audio.size)
    start = int(round((seg.start - window_start) * sample_rate))
    end = int(round((seg.end - window_start) * sample_rate))
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    if end <= start:
        return None
    return window_audio[start:end]


class RealtimeSession:
    """一个会话的全部状态与行为。不负责网络，只吐事件。"""

    def __init__(
        self,
        config: RealtimeConfig,
        *,
        transcriber: WindowTranscriber,
        store: SessionStore,
        embedder: SpeakerEmbedder | None = None,
        prompt: str = DEFAULT_PROMPT,
    ):
        self.config = config
        self._transcriber = transcriber
        self._store = store
        self._prompt = prompt or DEFAULT_PROMPT
        self._buffer = AudioRingBuffer(config.buffer_capacity, config.sample_rate)
        self._policy = WindowPolicy(
            window=config.window,
            hop=config.hop,
            min_first_window=config.min_first_window,
        )
        self._stitcher = Stitcher(window=config.window, tail=config.tail)
        self._gallery = SpeakerGallery(
            embedder,
            threshold=config.speaker_threshold,
            min_segment_sec=config.min_segment_sec,
            sample_rate=config.sample_rate,
        )
        self._last_run_sec: float | None = None
        self._window_running = False
        self._failures = 0
        self._window_id = 0
        self._segment_counter = 0
        self._last_window_sec = 0.0
        self._closed = False
        self._committed: list[CommittedSegment] = []

    @property
    def committed(self) -> list[CommittedSegment]:
        return list(self._committed)

    @property
    def closed(self) -> bool:
        return self._closed

    def push_audio(self, pcm: np.ndarray) -> None:
        if self._closed:
            return
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._buffer.append(arr)
        self._store.append_audio(arr)

    async def run_pending(self) -> list[dict]:
        if self._closed or self._window_running:
            return []
        total = self._buffer.total_seconds
        decision = self._policy.decide(total_seconds=total, last_run_sec=self._last_run_sec)
        if not decision.should_run:
            return []
        audio = self._buffer.slice(decision.start_sec, decision.end_sec)
        if audio is None or audio.size == 0:
            # 没有可推理的音频。推进 last_run_sec，避免每次轮询都重试。
            self._last_run_sec = decision.end_sec
            return []
        return await self._run_window(decision.start_sec, decision.end_sec, audio)

    async def close(self) -> list[dict]:
        if self._closed:
            return []
        events: list[dict] = []
        total = self._buffer.total_seconds
        window_audio: np.ndarray | None = None
        window_start = 0.0
        if total > 0:
            window_start = max(0.0, total - self.config.window)
            window_audio = self._buffer.slice(window_start, total)
            if window_audio is not None and window_audio.size:
                # 用正常的 tail 再跑一次，让临时区拿到最新推理结果，再整段定稿。
                events.extend(await self._run_window(window_start, total, window_audio))
        remaining = self._stitcher.flush()
        if remaining:
            newly = self._commit(remaining, window_audio, window_start)
            if newly:
                events.append({"type": "committed", "segments": [seg.to_dict() for seg in newly]})
            events.append({"type": "provisional", "segments": []})
        events.append({"type": "speaker", "speakers": self._gallery.speakers()})
        self._closed = True
        self._store.finalize(self._gallery.speakers())
        return events

    # --- 内部 ---

    async def _run_window(self, start: float, end: float, audio: np.ndarray) -> list[dict]:
        window_id = self._window_id
        self._window_id += 1
        self._window_running = True
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            committed, provisional = await loop.run_in_executor(
                None, self._process_window, start, end, window_id, audio
            )
        except Exception as exc:
            self._failures += 1
            return [
                {
                    "type": "error",
                    "code": "transcribe_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "failures": self._failures,
                },
                self._status_event(),
            ]
        finally:
            self._window_running = False
            self._last_run_sec = end
        self._failures = 0
        self._last_window_sec = time.perf_counter() - started

        events: list[dict] = []
        if committed:
            events.append({"type": "committed", "segments": [seg.to_dict() for seg in committed]})
        events.append(
            {
                "type": "provisional",
                "segments": [
                    {
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "speaker": seg.speaker,
                        "text": seg.text,
                    }
                    for seg in provisional
                ],
            }
        )
        events.append({"type": "speaker", "speakers": self._gallery.speakers()})
        events.append(self._status_event())
        return events

    def _process_window(
        self,
        start: float,
        end: float,
        window_id: int,
        audio: np.ndarray,
    ) -> tuple[list[CommittedSegment], list[Segment]]:
        raw = self._transcriber.transcribe_window(audio, prompt=self._prompt)
        result = self._stitcher.ingest(
            window_start=start,
            window_end=end,
            raw_text=raw,
            window_id=window_id,
        )
        committed = self._commit(result.committed, audio, start)
        self._store.write_provisional(result.provisional)
        return committed, result.provisional

    def _commit(
        self,
        segments: list[Segment],
        window_audio: np.ndarray | None,
        window_start: float,
    ) -> list[CommittedSegment]:
        if not segments:
            return []
        if window_audio is None:
            audio_of = lambda seg: None  # noqa: E731
        else:
            audio_of = lambda seg: segment_audio(  # noqa: E731
                window_audio, window_start, seg, self.config.sample_rate
            )
        assignments = self._gallery.assign(segments, audio_of)

        emitted: list[CommittedSegment] = []
        for item in assignments:
            self._segment_counter += 1
            seg = item.segment
            emitted.append(
                CommittedSegment(
                    id=f"seg-{self._segment_counter}",
                    start=seg.start,
                    end=seg.end,
                    speaker_id=item.speaker_id,
                    speaker_name=self._gallery.display_name(item.speaker_id),
                    text=seg.text,
                    confident=item.confident,
                )
            )
        self._store.append_committed(emitted)
        self._committed.extend(emitted)
        return emitted

    def _status_event(self) -> dict:
        hop = self.config.hop
        buffered = self._buffer.total_seconds
        # lag 用"字幕落后实时多少秒"来定义，而不是"缓冲里有多少秒音频没跑过推理"。
        # 后者恒等于零：_last_run_sec 每次都被设成窗口末尾，所以那个差值没有信息量。
        # 定稿水位线才是用户真正感受到的延迟，它天然包含 window + tail 的开销。
        lag = max(0.0, buffered - self._stitcher.committed_until)
        degraded = self._failures >= self.config.max_consecutive_failures
        return {
            "type": "status",
            "state": "degraded" if degraded else "running",
            "buffered_sec": round(buffered, 2),
            "rtf": round(self._last_window_sec / hop, 3) if hop > 0 else 0.0,
            "last_window_ms": int(round(self._last_window_sec * 1000)),
            "lag_sec": round(lag, 2),
            "degraded": degraded,
        }
```

`run_pending` 是唯一的推理入口，静音门控在 Task 11 里加进来。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_session.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add moss_transcribe_diarize/realtime/session.py tests/test_realtime_session.py
git commit -m "feat(realtime): add RealtimeSession orchestration"
```

---

### Task 11: 静音门控

会议里大量时间是静音，而每个窗口都是一次全量推理。静音门控把这段时间的算力直接省掉，是整条管线上性价比最高的一个开关。

**Files:**
- Create: `moss_transcribe_diarize/realtime/energy.py`
- Modify: `moss_transcribe_diarize/realtime/session.py`（`run_pending` 接入门控）
- Test: `tests/test_realtime_energy.py`
- Test: `tests/test_realtime_session.py`（追加 `SilenceGateTest`）

**Interfaces:**
- Consumes: 无
- Produces:
  - `frame_rms_db(audio: np.ndarray, sample_rate: int, frame_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray`
  - `is_silent(audio: np.ndarray, sample_rate: int, threshold_db: float, max_active_ratio: float) -> bool`

门控跳过时必须**照常推进 `last_run_sec`**，否则 `decide` 会在每次轮询都判定该跑，退化成忙等。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_realtime_energy.py`：

```python
from __future__ import annotations

import unittest

import numpy as np

from moss_transcribe_diarize.realtime.energy import frame_rms_db, is_silent


def _tone(seconds: float, *, amplitude: float = 0.3, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(seconds: float, *, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


class FrameRmsDbTest(unittest.TestCase):
    def test_silence_is_very_low(self):
        levels = frame_rms_db(_silence(0.1), 16000)

        self.assertTrue(np.all(levels < -100.0))

    def test_tone_is_loud(self):
        levels = frame_rms_db(_tone(0.2, amplitude=0.5), 16000)

        self.assertTrue(np.all(levels > -20.0))

    def test_frame_count_matches_the_hop(self):
        # 0.1 秒音频 = 1600 样本；20ms 窗 = 320，10ms 移 = 160
        # count = 1 + (1600 - 320) // 160 = 9
        levels = frame_rms_db(_silence(0.1), 16000, frame_ms=20.0, hop_ms=10.0)

        self.assertEqual(levels.size, 9)

    def test_short_audio_still_yields_one_frame(self):
        levels = frame_rms_db(_silence(0.001), 16000)

        self.assertEqual(levels.size, 1)

    def test_empty_audio_yields_one_frame(self):
        levels = frame_rms_db(_silence(0.0), 16000)

        self.assertEqual(levels.size, 1)


class IsSilentTest(unittest.TestCase):
    def test_pure_silence_is_silent(self):
        self.assertTrue(is_silent(_silence(2.0), 16000, -45.0, 0.05))

    def test_continuous_speech_is_not_silent(self):
        self.assertFalse(is_silent(_tone(2.0), 16000, -45.0, 0.05))

    def test_short_burst_in_a_long_window_is_silent(self):
        # 0.5 秒有声落在 20 秒窗口里，活跃帧占比约 2.5%，远低于 5% 门限。
        # 不要用 1.0 秒——那算出来是 5.0025%，正好压在门限上。
        audio = np.concatenate([_silence(19.5), _tone(0.5)])

        self.assertTrue(is_silent(audio, 16000, -45.0, 0.05))

    def test_enough_speech_in_a_long_window_is_not_silent(self):
        audio = np.concatenate([_silence(14.0), _tone(6.0)])

        self.assertFalse(is_silent(audio, 16000, -45.0, 0.05))

    def test_empty_audio_is_silent(self):
        self.assertTrue(is_silent(_silence(0.0), 16000, -45.0, 0.05))


if __name__ == "__main__":
    unittest.main()
```

在 `tests/test_realtime_session.py` 追加：

```python
class SilenceGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, transcriber, **kwargs) -> RealtimeSession:
        # 注意不要把 silence_gate 同时写死再通过 **kwargs 传一次——那会是重复关键字。
        params = {"silence_gate": True}
        params.update(kwargs)
        return RealtimeSession(
            _config(**params),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def test_silent_window_skips_inference(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))

        events = self._run(session.run_pending())

        self.assertEqual(transcriber.window_seconds, [])
        status = next(event for event in events if event["type"] == "status")
        self.assertEqual(status["state"], "running")

    def test_silent_window_still_advances_the_schedule(self):
        """跳过后必须推进 last_run_sec，否则每次轮询都在重算同一个窗口。"""
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))
        self._run(session.run_pending())

        session.push_audio(_silence(2.0))
        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_speech_after_silence_is_transcribed(self):
        transcriber = ScriptedTranscriber(["[1][S01]恢复[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))
        self._run(session.run_pending())

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        self.assertEqual(len(transcriber.window_seconds), 1)
        self.assertIn("committed", [event["type"] for event in events])

    def test_close_runs_the_final_window_even_when_it_is_silent(self):
        # close() 刻意**不**经过静音门控：收尾那一窗是为了让临时区拿到最新结果。
        # 构造一个整段安静的收尾窗口，就能从转写器的调用次数上直接看出来——门控若
        # 在这里生效，最后这一窗根本不会被推理。
        #
        # 时序：前 8 秒有语音（首窗 [0,8] 出一段临时段），之后 22 秒全静音，于是
        # total = 30 时收尾窗口是 [10, 30]，整段落在这段静音里。
        #
        # 早期版本用的是 [18][S01]结尾[19] ×2 + 20 秒语音 + 10 秒静音，那时收尾窗口
        # [10,30] 里含 10 秒语音、is_silent 为 False，**即使 close 被门控也不会跳过**，
        # 于是该用例两种实现下都通过——它命名的正是"close 不被门控"，却钉不住它。
        transcriber = ScriptedTranscriber(["[6][S01]旧[7]", "[1][S01]新[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())
        calls_after_first_run = len(transcriber.window_seconds)

        session.push_audio(_silence(22.0))
        self._run(session.close())

        self.assertEqual(len(transcriber.window_seconds), calls_after_first_run + 1)

    def test_gate_can_be_disabled(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber, silence_gate=False)
        session.push_audio(_silence(10.0))

        self._run(session.run_pending())

        self.assertEqual(len(transcriber.window_seconds), 1)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_realtime_energy.py tests/test_realtime_session.py -v`
Expected: `tests/test_realtime_energy.py` 报 `ModuleNotFoundError`；`SilenceGateTest` 里除 `test_gate_can_be_disabled` 外的用例失败，因为静音窗口现在会被送去推理。

- [ ] **Step 3: 实现 `energy.py`**

```python
"""Cheap frame-energy silence detection to skip inference on quiet windows."""

from __future__ import annotations

import numpy as np

SILENCE_FLOOR = 1e-10


def frame_rms_db(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """逐帧 RMS，单位 dBFS。极短或空音频也至少返回一帧。"""
    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if data.size < frame:
        data = np.pad(data, (0, frame - data.size))
    count = 1 + (data.size - frame) // hop
    indices = np.arange(count)[:, None] * hop + np.arange(frame)[None, :]
    frames = data[indices]
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return (20.0 * np.log10(np.maximum(rms, SILENCE_FLOOR))).astype(np.float32)


def is_silent(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    max_active_ratio: float,
) -> bool:
    """高于阈值的帧占比低于 ``max_active_ratio`` 就判为静音。"""
    levels = frame_rms_db(audio, sample_rate)
    if levels.size == 0:
        return True
    active = float(np.count_nonzero(levels > threshold_db)) / float(levels.size)
    return active < max_active_ratio
```

注意 `frame_rms_db(_silence(0.1), 16000)` 的帧数：0.1 秒 = 1600 样本，窗 320（20ms）、移 160（10ms），故 `count = 1 + (1600 - 320) // 160 = 9`。测试断言的是 **9**，与实现一致。公式是 `1 + (samples - frame) // hop`，不是 `samples // hop`——后者会得 10 并让测试莫名失败。

- [ ] **Step 4: 把门控接进 `run_pending`**

在 `moss_transcribe_diarize/realtime/session.py` 顶部加 import：

```python
from .energy import is_silent
```

把 `run_pending` 替换为：

```python
    async def run_pending(self) -> list[dict]:
        if self._closed or self._window_running:
            return []
        total = self._buffer.total_seconds
        decision = self._policy.decide(total_seconds=total, last_run_sec=self._last_run_sec)
        if not decision.should_run:
            return []
        audio = self._buffer.slice(decision.start_sec, decision.end_sec)
        if audio is None or audio.size == 0:
            # 没有可推理的音频。推进 last_run_sec，避免每次轮询都重试。
            self._last_run_sec = decision.end_sec
            return []
        if self.config.silence_gate and is_silent(
            audio,
            self.config.sample_rate,
            self.config.silence_rms_db,
            self.config.silence_frame_ratio,
        ):
            # 整窗静音：跳过推理，但必须推进 last_run_sec，否则下次轮询会重算
            # 同一个窗口，退化成忙等。
            self._last_run_sec = decision.end_sec
            return [self._status_event()]
        return await self._run_window(decision.start_sec, decision.end_sec, audio)
```

门控只在 `run_pending` 里生效。`close()` 走的是无条件推理路径——会话收尾不能因为最后一段是静音就把临时内容丢掉，所以它不经过门控。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_realtime_energy.py tests/test_realtime_session.py -v`
Expected: 全部 PASS

Run: `python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
git add moss_transcribe_diarize/realtime/energy.py moss_transcribe_diarize/realtime/session.py tests/test_realtime_energy.py tests/test_realtime_session.py
git commit -m "feat(realtime): skip inference on silent windows"
```

---

## 阶段一完成标志

跑一遍全量测试应当全绿：

```bash
python -m pytest tests/ -v
```

此时可以手工验证管线确实端到端可用：

```bash
python - <<'PY'
import asyncio, tempfile
from pathlib import Path
import numpy as np
from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.session import RealtimeSession
from moss_transcribe_diarize.realtime.store import SessionStore

class Fake:
    def transcribe_window(self, audio, *, prompt):
        return "[1.0][S01]这是第一句[3.0][4.0][S02]这是第二句[6.0]"

async def main():
    runs = Path(tempfile.mkdtemp())
    session = RealtimeSession(
        RealtimeConfig(silence_gate=False),
        transcriber=Fake(),
        store=SessionStore(runs, "demo"),
    )
    # 12 秒语音，超过 min_first_window=8
    t = np.arange(12 * 16000, dtype=np.float32) / 16000
    session.push_audio((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
    for event in await session.run_pending():
        print(event["type"], event.get("segments", event.get("state", "")))
    for event in await session.close():
        print(event["type"], event.get("segments", ""))
    print("落盘:", sorted(p.name for p in (runs / "demo").iterdir()))

asyncio.run(main())
PY
```

预期：`committed` 事件里出现 `seg-1` 和 `seg-2` 两段，`speaker` 两份，`status` 的 `rtf` 是个小数，最后打印出 `['audio.wav', 'provisional.json', 'session.json', 'transcript.jsonl']`。

## 后续计划

阶段一的交付物是一个库，没有服务也没有界面。接下来的两个计划：

- **阶段二：服务与协议** —— 抽出 `app/openai_audio_client.py`、实现 `VllmWindowTranscriber` 与 `OnnxCampplusEmbedder`（含模型下载与 SHA-256 校验）、`realtime_server` 的 WebSocket 协议、`mtd-realtime` CLI、无浏览器冒烟客户端。
- **阶段三：浏览器前端** —— AudioWorklet 采集与混音、页面渲染与交互、i18n、README 与真机端到端调参。

阶段一末尾的 `python - <<'PY'` 手工验证脚本是阶段二 `scripts/realtime_client.py` 的雏形。

### 阶段二必须接上的三件事

都源于阶段一的最终整分支审查——**漏掉任何一条都会产生静默错误**（程序照跑、测试照绿、结果错）：

1. **把 `config.effective_max_new_tokens()` 传给转写器。** 阶段一里这个方法**没有任何消费者**，而 `HfWindowTranscriber` 把同一个数硬编码了第二份（`max_new_tokens=1020` = 20×51）。用户用 `--window 40` 时 config 算 2040、HF 后端仍用 1020，**输出被静默截断**。
2. **`session_id` 在拼进路径之前必须校验。** `SessionStore.session_dir` 与 `__init__` 现在都会校验（并显式拒绝 `.`、`..`），但阶段二的 `GET /api/sessions/{id}` 会把 URL 段直接传进来，传输层必须再校验一次——否则是路径穿越。**另注意一处不对称**：`list_sessions` 会返回 `runs_dir` 下**任意**目录名，而 `session_dir`/`load_committed` 对字符集之外的 id 会抛 `ValueError`——所以运维手工建的目录（带空格或非 ASCII）会出现在会话列表里、点进去却抛异常。**传输层必须把这个 `ValueError` 映射成 400/404，否则它会变成 500。**
3. **驱动循环必须"先 await 完 `run_pending` 再 `close()`"。** 二者并发会让两个窗口落在两个 executor 线程里并改 `Stitcher` 的水位线、`SpeakerGallery` 的表、段落计数器与 `SessionStore`，后果是重复定稿或临时区错乱，**不会报错**。

另有两件只有阶段二才能回答的事：`--no-silence-gate` 开关（阶段一里门控无法关闭，而且它的丢失上界是阶段一才加的）；以及 spec §11 列出的实测风险——`rtf` 是否真的小于 1、声纹在真实会议音频上的误合并率。
