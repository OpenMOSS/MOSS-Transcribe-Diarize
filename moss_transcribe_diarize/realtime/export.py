"""把一个会话的定稿段落导成人类与机器都读得懂的几种格式。

``srt`` / ``json`` / ``ass`` 直接复用 ``subtitle/export.py``——那里的实现已经被
字幕工坊用测试钉住，重复一份只会让两边漂移。``md`` / ``txt`` 那个包没有，在这里写。

说话人显示名一律**先查说话人表**：用户把 S01 改成"张总"之后，``transcript.jsonl``
里冻结的还是旧值，导出必须用新的。
"""

from __future__ import annotations

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
