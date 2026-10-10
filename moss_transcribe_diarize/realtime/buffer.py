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
        逻辑顺序写入就维持了「逻辑索引 ``i`` 存放在物理位置 ``i % capacity``」这
        一不变量——即便单次追加超过容量、需要多圈回绕也一样。
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
