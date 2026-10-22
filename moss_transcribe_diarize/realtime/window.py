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
            # 首次运行的左边界恒为 0。不能套用后面那条统一公式：当
            # min_first_window > window 时它算出的左边界会落在 0 之后，而后续窗口都是
            # [t - window, t]，永远够不到开头，于是 [0, total - window) 这段音频不会
            # 被任何一次推理覆盖，被静默丢掉。
            return WindowDecision(True, start_sec=0.0, end_sec=float(total_seconds))
        if total_seconds - last_run_sec < self.hop:
            return WindowDecision(False)
        return WindowDecision(
            True,
            start_sec=max(0.0, total_seconds - self.window),
            end_sec=float(total_seconds),
        )
