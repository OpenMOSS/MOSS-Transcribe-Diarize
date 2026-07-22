from __future__ import annotations

import unittest

from moss_transcribe_diarize.subtitle import SubtitleSegment, assign_overlap_lanes


class SubtitleLayoutTest(unittest.TestCase):
    def test_assign_overlap_lanes_stacks_concurrent_segments(self):
        segments = [
            SubtitleSegment("a", 0.0, 5.0, "S01", "one"),
            SubtitleSegment("b", 1.0, 3.0, "S02", "two"),
            SubtitleSegment("c", 2.0, 6.0, "S03", "three"),
            SubtitleSegment("d", 6.0, 7.0, "S01", "four"),
        ]

        self.assertEqual(assign_overlap_lanes(segments), [0, 1, 2, 0])

    def test_assign_overlap_lanes_preserves_input_order(self):
        segments = [
            SubtitleSegment("late", 5.0, 6.0, "S01", "late"),
            SubtitleSegment("early", 0.0, 3.0, "S01", "early"),
            SubtitleSegment("overlap", 1.0, 2.0, "S02", "overlap"),
        ]

        self.assertEqual(assign_overlap_lanes(segments), [0, 0, 1])

    def test_touching_segments_share_a_lane(self):
        segments = [
            SubtitleSegment("a", 0.0, 1.0, "S01", "one"),
            SubtitleSegment("b", 1.0, 2.0, "S02", "two"),
        ]

        self.assertEqual(assign_overlap_lanes(segments), [0, 0])

    def test_sub_centisecond_overlap_uses_another_lane(self):
        segments = [
            SubtitleSegment("a", 0.0, 1.004, "S01", "one"),
            SubtitleSegment("b", 1.003, 2.0, "S02", "two"),
        ]

        self.assertEqual(assign_overlap_lanes(segments), [0, 1])


if __name__ == "__main__":
    unittest.main()
