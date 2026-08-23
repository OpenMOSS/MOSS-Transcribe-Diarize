import unittest
from unittest.mock import patch

import numpy as np
import torch

from moss_transcribe_diarize.inference_utils import (
    DEFAULT_PROMPT,
    build_transcription_messages,
    load_audio_item,
    process_audio_info,
)


class InferenceUtilsTest(unittest.TestCase):
    @patch("moss_transcribe_diarize.inference_utils.load_audio_item")
    def test_process_audio_info_accepts_numpy_waveform(self, load_audio_item):
        waveform = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        load_audio_item.return_value = waveform
        messages = [{"role": "user", "content": [{"type": "audio", "audio": waveform}]}]

        audios = process_audio_info(messages, sampling_rate=16_000)

        self.assertEqual(audios, [waveform])
        load_audio_item.assert_called_once_with(waveform, sampling_rate=16_000)

    @patch("moss_transcribe_diarize.inference_utils.load_audio_item")
    def test_process_audio_info_accepts_tensor_without_truth_value_error(self, load_audio_item):
        waveform = torch.tensor([0.1, -0.2, 0.3])
        load_audio_item.return_value = waveform.numpy()
        messages = [{"role": "user", "content": [{"type": "audio", "audio": waveform}]}]

        process_audio_info(messages, sampling_rate=16_000)

        load_audio_item.assert_called_once_with(waveform, sampling_rate=16_000)

    @patch("moss_transcribe_diarize.inference_utils.load_audio")
    def test_load_audio_item_converts_tensor_to_numpy(self, load_audio):
        waveform = torch.tensor([0.1, -0.2, 0.3])
        load_audio.return_value = waveform.numpy()

        load_audio_item(waveform, sampling_rate=16_000)

        loaded_audio = load_audio.call_args.args[0]
        self.assertIsInstance(loaded_audio, np.ndarray)
        np.testing.assert_array_equal(loaded_audio, waveform.numpy())

    def test_build_transcription_messages_preserves_in_memory_audio(self):
        waveform = np.zeros(8, dtype=np.float32)

        messages = build_transcription_messages(waveform, prompt=" ")

        self.assertIs(messages[0]["content"][0]["audio"], waveform)
        self.assertEqual(messages[0]["content"][1]["text"], DEFAULT_PROMPT)


if __name__ == "__main__":
    unittest.main()
