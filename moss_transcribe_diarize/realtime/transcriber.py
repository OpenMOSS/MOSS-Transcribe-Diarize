"""Window-level transcription backends."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import soundfile as sf

from moss_transcribe_diarize.app.openai_audio_client import (
    encode_wav_bytes,
    extract_transcription_text,
    transcribe_bytes,
)

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


@runtime_checkable
class WindowTranscriber(Protocol):
    """把一个音频窗口转成模型原始输出文本。"""

    def transcribe_window(self, audio: np.ndarray, *, prompt: str) -> str: ...


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


class HfWindowTranscriber:
    """本地 HF 后端：窗口写临时 wav，交给 ModelRunner。

    ``runner`` 只需要有 ``transcribe(path, *, prompt, max_new_tokens, decoding)``
    方法并返回带 ``.text`` 的对象，所以测试可以注入假实现，不必加载 torch。

    ``max_new_tokens`` 是**可选的硬上限**，不是默认值：默认预算按每次拿到的音频长度
    推算（见 ``token_budget``），因为覆盖性保证会让部分窗口长于 ``window``。
    """

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

    @property
    def window_path(self) -> Path:
        return self._path

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
