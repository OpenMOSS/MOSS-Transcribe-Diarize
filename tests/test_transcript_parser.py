from __future__ import annotations

import unittest

from moss_transcribe_diarize.transcript_parser import (
    TranscriptParseError,
    TranscriptSegment,
    TranscriptStreamParser,
    iter_transcript_segments,
    parse_transcript,
)


class TranscriptParserTest(unittest.TestCase):
    def test_parse_compact_transcript(self):
        text = "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]"

        self.assertEqual(
            parse_transcript(text),
            [
                TranscriptSegment(0.48, 1.66, "S01", "Welcome"),
                TranscriptSegment(12.26, 13.81, "S02", "Ready"),
            ],
        )

    def test_streaming_with_single_character_chunks(self):
        text = "[0.48][S01]你好[1.66][12.26][S02]可以开始[13.81]"
        parser = TranscriptStreamParser()
        segments = []

        for ch in text:
            segments.extend(parser.feed(ch))
        segments.extend(parser.close())

        self.assertEqual(
            segments,
            [
                TranscriptSegment(0.48, 1.66, "S01", "你好"),
                TranscriptSegment(12.26, 13.81, "S02", "可以开始"),
            ],
        )

    def test_iter_transcript_segments_accepts_arbitrary_chunks(self):
        chunks = ["[0", ".48][", "S01]", "hello[", "1.66]", "[2.0][S02]", "bye", "[3.0]"]

        self.assertEqual(
            list(iter_transcript_segments(chunks)),
            [
                TranscriptSegment(0.48, 1.66, "S01", "hello"),
                TranscriptSegment(2.0, 3.0, "S02", "bye"),
            ],
        )

    def test_numeric_brackets_inside_text_are_preserved(self):
        text = "[0][S01]第[2024]年，编号[001]继续[4]"

        self.assertEqual(
            parse_transcript(text),
            [TranscriptSegment(0.0, 4.0, "S01", "第[2024]年，编号[001]继续")],
        )

    def test_missing_speaker_uses_default_speaker(self):
        text = "[0.0]單一說話者片段[2.0][2.0]下一段[3.0]"

        self.assertEqual(
            parse_transcript(text),
            [
                TranscriptSegment(0.0, 2.0, "S01", "單一說話者片段"),
                TranscriptSegment(2.0, 3.0, "S01", "下一段"),
            ],
        )

    def test_missing_speaker_can_be_parsed_from_arbitrary_chunks(self):
        chunks = ["[0.0]hel", "lo[1.0]", "[1.0]world[2.0]"]

        self.assertEqual(
            list(iter_transcript_segments(chunks)),
            [
                TranscriptSegment(0.0, 1.0, "S01", "hello"),
                TranscriptSegment(1.0, 2.0, "S01", "world"),
            ],
        )

    def test_strict_mode_keeps_unlabelled_segments_ignored(self):
        self.assertEqual(parse_transcript("[0]unlabelled[1]", default_speaker=None), [])

    def test_default_speaker_must_be_valid(self):
        with self.assertRaises(TranscriptParseError):
            TranscriptStreamParser(default_speaker="speaker-1")

    def test_noise_before_first_segment_is_ignored(self):
        text = "noise [bad][0.1][S01]hello[0.9]"

        self.assertEqual(
            parse_transcript(text),
            [TranscriptSegment(0.1, 0.9, "S01", "hello")],
        )

    def test_whitespace_between_segments_is_ignored(self):
        text = "[0][S01]a[1]\n [2][S02]b[3]"

        self.assertEqual(
            parse_transcript(text),
            [
                TranscriptSegment(0.0, 1.0, "S01", "a"),
                TranscriptSegment(2.0, 3.0, "S02", "b"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
