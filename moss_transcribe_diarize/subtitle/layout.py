from __future__ import annotations

from collections.abc import Iterable

from .models import SubtitleSegment


def assign_overlap_lanes(segments: Iterable[SubtitleSegment]) -> list[int]:
    """Assign vertical lanes for overlapping subtitle segments.

    Lane 0 is the baseline subtitle row nearest the bottom edge. Larger lane
    numbers stack upward when multiple subtitle events are active at once.
    """
    segment_list = list(segments)
    lanes = [0] * len(segment_list)
    lane_ends: list[float] = []

    indexed = sorted(
        enumerate(segment_list),
        key=lambda item: (float(item[1].start), float(item[1].end), item[0]),
    )
    for original_index, segment in indexed:
        start = float(segment.start)
        end = max(start, float(segment.end))
        for lane, lane_end in enumerate(lane_ends):
            if lane_end <= start:
                lanes[original_index] = lane
                lane_ends[lane] = end
                break
        else:
            lanes[original_index] = len(lane_ends)
            lane_ends.append(end)

    return lanes
