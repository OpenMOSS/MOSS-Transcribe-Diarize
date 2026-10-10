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

    def test_rows_without_a_speaker_are_kept(self):
        """缺 speaker 的行照常导出，只是说话人一栏为空——丢内容比丢标签更糟。"""
        rows = ROWS + [{"id": "seg-3", "start": 1.0, "end": 2.0, "text": "无"}]

        segments = to_subtitle_segments(rows)

        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[2].speaker, "")


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
        # 两段之间恰好一个空行（实现是 "\n\n".join，另加结尾的 "\n"）。
        self.assertEqual(export_text(ROWS).count("\n\n"), 1)
        self.assertTrue(export_text(ROWS).endswith("\n"))


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
