"""Tunable parameters for the realtime transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .transcriber import MIN_NEW_TOKENS, TOKENS_PER_AUDIO_SECOND


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
        """按 ``window`` 推算的预算——**只用作下限**。

        真正的预算是按每次拿到的音频长度算的（见 ``transcriber.token_budget``）：
        4.7 的覆盖性保证会让被迫放行的窗口长于 ``window``，按 ``window`` 定死的预算
        会把它们的尾部截掉。这里保留这个方法，是给调用方一个"一个 window 长的窗口
        至少需要多少 token"的下限。
        """
        if self.max_new_tokens is not None:
            return self.max_new_tokens
        return max(MIN_NEW_TOKENS, int(round(self.window * TOKENS_PER_AUDIO_SECOND)))
