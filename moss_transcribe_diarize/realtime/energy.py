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
