from __future__ import annotations

import unittest

from moss_transcribe_diarize.subtitle import (
    SubtitleSegment,
    coerce_subtitle_segments,
    normalize_segments,
    subtitle_segments_from_transcript,
)


class SubtitlePostprocessTest(unittest.TestCase):
    def test_builds_subtitle_segments_from_transcript(self):
        segments = subtitle_segments_from_transcript("[0][S01]你好[1.5][2][S02]开始[3.5]")

        self.assertEqual([segment.speaker for segment in segments], ["S01", "S02"])
        self.assertEqual([segment.text for segment in segments], ["你好", "开始"])
        self.assertEqual(segments[0].id, "seg_0001")

    def test_recovers_unlabelled_single_speaker_segments(self):
        segments = subtitle_segments_from_transcript(
            "[0.0]連續單一說話者內容[2.0][2.0]下一段[3.0]",
            postprocess=False,
        )

        self.assertEqual(
            [(segment.start, segment.end, segment.speaker, segment.text) for segment in segments],
            [
                (0.0, 2.0, "S01", "連續單一說話者內容"),
                (2.0, 3.0, "S01", "下一段"),
            ],
        )

    def test_can_build_raw_subtitle_segments_without_postprocess(self):
        segments = subtitle_segments_from_transcript(
            "[0][S01]短[0.4][0.2][S01]重叠但保留[0.8]",
            postprocess=False,
        )

        self.assertEqual([(s.start, s.end, s.text) for s in segments], [(0.0, 0.4, "短"), (0.2, 0.8, "重叠但保留")])

    def test_coerce_payload_does_not_reorder_or_fix_times(self):
        segments = coerce_subtitle_segments(
            [
                {"id": "b", "start": 3, "end": 2, "speaker": "S02", "text": ""},
                {"id": "a", "start": 0.2, "end": 0.1, "speaker": "S01", "text": "x"},
            ]
        )

        self.assertEqual([segment.id for segment in segments], ["b", "a"])
        self.assertEqual([(segment.start, segment.end) for segment in segments], [(3.0, 2.0), (0.2, 0.1)])
        self.assertEqual(segments[0].text, "")

    def test_merges_adjacent_same_speaker_short_gap(self):
        segments = normalize_segments(
            [
                SubtitleSegment("a", 0, 1.2, "S01", "你好"),
                SubtitleSegment("b", 1.3, 2.4, "S01", "世界"),
            ],
            merge_gap=0.3,
            max_chars=24,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "你好世界")
        self.assertEqual(segments[0].end, 2.4)

    def test_fixes_overlaps_and_min_duration(self):
        segments = normalize_segments(
            [
                SubtitleSegment("a", 0, 0.4, "S01", "a"),
                SubtitleSegment("b", 0.2, 0.6, "S02", "b"),
            ],
            min_duration=1.0,
            merge_gap=0,
        )

        self.assertEqual([(s.start, s.end) for s in segments], [(0.0, 1.0), (1.0, 2.0)])

    def test_splits_long_segments(self):
        segments = normalize_segments(
            [
                SubtitleSegment("a", 0, 12, "S01", "第一句很长，需要切开。第二句也很长，需要继续切开。"),
            ],
            max_duration=6.0,
            max_chars=10,
            merge_gap=0,
        )

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(segment.end > segment.start for segment in segments))
        self.assertEqual(segments[0].start, 0.0)


if __name__ == "__main__":
    unittest.main()
