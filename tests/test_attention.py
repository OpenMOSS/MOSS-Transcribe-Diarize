from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from moss_transcribe_diarize.attention import (
    _candidate_list,
    load_model_with_attention_fallback,
    normalize_attention_implementation,
)


class AttentionBackendTest(unittest.TestCase):
    def test_normalize_attention_aliases_and_rejects_unknown(self):
        self.assertEqual(normalize_attention_implementation(None), "auto")
        self.assertEqual(normalize_attention_implementation("default"), "auto")
        self.assertEqual(normalize_attention_implementation("flash3"), "flash_attention_3")
        with self.assertRaises(ValueError):
            normalize_attention_implementation("not-a-backend")

    def test_auto_candidates_keep_sdpa_before_eager(self):
        candidates, attempts = _candidate_list("auto", device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(candidates, ["sdpa", "eager"])
        self.assertEqual([item["backend"] for item in attempts], [
            "flash_attention_4",
            "flash_attention_3",
            "flash_attention_2",
        ])

    def test_hidden_eager_resolution_is_rejected_before_final_fallback(self):
        calls = []

        def loader(_path, **kwargs):
            calls.append(kwargs["attn_implementation"])
            return SimpleNamespace(
                config=SimpleNamespace(
                    _attn_implementation="eager",
                    text_config=SimpleNamespace(_attn_implementation="eager"),
                    audio_config=SimpleNamespace(_attn_implementation="eager"),
                )
            )

        with self.assertLogs("moss_transcribe_diarize.attention", level="WARNING") as captured:
            model, report = load_model_with_attention_fallback(
                "fake-model",
                device=torch.device("cpu"),
                dtype=torch.float32,
                model_loader=loader,
            )

        self.assertIsNotNone(model)
        self.assertEqual(calls, ["sdpa", "eager"])
        self.assertEqual(report["selected"], "eager")
        self.assertTrue(any("selected eager" in line for line in captured.output))

    def test_auto_falls_from_flash_candidate_to_sdpa(self):
        calls = []
        sdpa_model = SimpleNamespace(
            config=SimpleNamespace(
                _attn_implementation="sdpa",
                text_config=SimpleNamespace(_attn_implementation="sdpa"),
                audio_config=SimpleNamespace(_attn_implementation="sdpa"),
            )
        )

        def fake_preflight(implementation, _device, _dtype):
            return None if implementation == "flash_attention_3" else "unavailable in test"

        def loader(_path, **kwargs):
            implementation = kwargs["attn_implementation"]
            calls.append(implementation)
            if implementation == "flash_attention_3":
                raise RuntimeError("simulated flash load failure")
            return sdpa_model

        with patch("moss_transcribe_diarize.attention._flash_preflight", side_effect=fake_preflight):
            with patch("moss_transcribe_diarize.attention.probe_sdpa_kernels", return_value=("flash", "math")):
                loaded, report = load_model_with_attention_fallback(
                    "fake-model",
                    device=torch.device("cuda"),
                    dtype=torch.bfloat16,
                    model_loader=loader,
                )

        self.assertIs(loaded, sdpa_model)
        self.assertEqual(calls, ["flash_attention_3", "sdpa"])
        self.assertEqual(report["selected"], "sdpa")

    def test_sdpa_selection_is_reported(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                _attn_implementation="sdpa",
                text_config=SimpleNamespace(_attn_implementation="sdpa"),
                audio_config=SimpleNamespace(_attn_implementation="sdpa"),
            )
        )

        with patch(
            "moss_transcribe_diarize.attention.probe_sdpa_kernels",
            return_value=("flash", "math"),
        ):
            loaded, report = load_model_with_attention_fallback(
                "fake-model",
                device=torch.device("cpu"),
                dtype=torch.float32,
                model_loader=lambda _path, **_kwargs: model,
            )

        self.assertIs(loaded, model)
        self.assertEqual(report["selected"], "sdpa")
        self.assertEqual(report["config"]["text"], "sdpa")
        self.assertEqual(report["sdpa_kernels"], ["flash", "math"])

    def test_probe_failure_does_not_force_eager(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                _attn_implementation="sdpa",
                text_config=SimpleNamespace(_attn_implementation="sdpa"),
                audio_config=SimpleNamespace(_attn_implementation="sdpa"),
            )
        )
        with patch(
            "moss_transcribe_diarize.attention.probe_sdpa_kernels",
            side_effect=RuntimeError("probe unavailable"),
        ):
            loaded, report = load_model_with_attention_fallback(
                "fake-model",
                device=torch.device("cpu"),
                dtype=torch.float32,
                model_loader=lambda _path, **_kwargs: model,
            )

        self.assertIs(loaded, model)
        self.assertEqual(report["selected"], "sdpa")
        self.assertEqual(report["sdpa_kernels"], [])


if __name__ == "__main__":
    unittest.main()
