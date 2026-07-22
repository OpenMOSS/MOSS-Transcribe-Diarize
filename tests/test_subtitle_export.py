from __future__ import annotations

import json
import unittest

from moss_transcribe_diarize.subtitle import SubtitleSegment, SubtitleStyle, export_ass, export_json, export_srt
from moss_transcribe_diarize.subtitle.export import format_ass_time, format_srt_time


class SubtitleExportTest(unittest.TestCase):
    def test_time_formatters(self):
        self.assertEqual(format_srt_time(3661.234), "01:01:01,234")
        self.assertEqual(format_ass_time(3661.23), "1:01:01.23")

    def test_export_srt(self):
        text = export_srt([SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")])

        self.assertIn("00:00:00,500 --> 00:00:02,000", text)
        self.assertIn("S01: hello", text)

    def test_export_srt_with_speaker_names(self):
        text = export_srt(
            [SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")],
            speaker_names={"S01": "Alice"},
        )

        self.assertIn("Alice: hello", text)

    def test_export_ass(self):
        text = export_ass(
            [SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")],
            style=SubtitleStyle(font_size=42, show_speaker=False),
            video_width=1280,
            video_height=720,
        )

        self.assertIn("PlayResX: 1280", text)
        self.assertIn("Style: Speaker_S01,Noto Sans CJK SC,42", text)
        self.assertIn("Dialogue: 0,0:00:00.50,0:00:02.00,Speaker_S01", text)
        self.assertIn("hello", text)

    def test_export_ass_with_speaker_names(self):
        text = export_ass(
            [SubtitleSegment("seg_0001", 0.5, 2.0, "S01", "hello")],
            style=SubtitleStyle(font_size=42, speaker_names={"S01": "Alice"}),
            video_width=1280,
            video_height=720,
        )

        self.assertIn("Alice: hello", text)

    def test_export_ass_offsets_overlapping_segments(self):
        text = export_ass(
            [
                SubtitleSegment("seg_0001", 0.0, 5.0, "S01", "one"),
                SubtitleSegment("seg_0002", 1.0, 3.0, "S02", "two"),
                SubtitleSegment("seg_0003", 5.0, 6.0, "S01", "three"),
            ],
            style=SubtitleStyle(font_size=50, margin_v=60, speaker_colors=False),
            video_width=1280,
            video_height=720,
        )

        self.assertIn("Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,60,,S01: one", text)
        self.assertIn("Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,110,,S02: two", text)
        self.assertIn("Dialogue: 0,0:00:05.00,0:00:06.00,Default,,0,0,60,,S01: three", text)

    def test_export_json(self):
        data = json.loads(export_json([SubtitleSegment("seg_0001", 0, 1, "S01", "hello")]))

        self.assertEqual(data[0]["id"], "seg_0001")
        self.assertEqual(data[0]["text"], "hello")


if __name__ == "__main__":
    unittest.main()
