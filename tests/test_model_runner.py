from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from moss_transcribe_diarize.app.model_runner import ModelRunner


def load_with(*, cuda: bool, flash_installed: bool, side_effect=None) -> MagicMock:
    with (
        patch(
            "moss_transcribe_diarize.app.model_runner.AutoModelForCausalLM"
        ) as model_cls,
        patch("moss_transcribe_diarize.app.model_runner.AutoProcessor"),
        patch("torch.cuda.is_available", return_value=cuda),
        patch(
            "importlib.util.find_spec",
            return_value=MagicMock() if flash_installed else None,
        ),
    ):
        if side_effect is not None:
            model_cls.from_pretrained.side_effect = side_effect
        else:
            model_cls.from_pretrained.return_value = MagicMock()
        runner = ModelRunner(
            "unused-local-model", device="cuda:0" if cuda else "cpu", dtype="bf16"
        )
        runner._ensure_loaded()
        return model_cls.from_pretrained


class ModelRunnerTest(unittest.TestCase):
    def test_uses_sdpa_without_flash_attn(self):
        from_pretrained = load_with(cuda=False, flash_installed=False)
        self.assertEqual(
            from_pretrained.call_args.kwargs["attn_implementation"], "sdpa"
        )

    def test_prefers_flash_attention_2_on_cuda(self):
        from_pretrained = load_with(cuda=True, flash_installed=True)
        self.assertEqual(
            from_pretrained.call_args.kwargs["attn_implementation"], "flash_attention_2"
        )

    def test_falls_back_to_sdpa_when_flash_load_fails(self):
        from_pretrained = load_with(
            cuda=True,
            flash_installed=True,
            side_effect=[RuntimeError("flash-attn build mismatch"), MagicMock()],
        )
        requested = [
            call.kwargs["attn_implementation"]
            for call in from_pretrained.call_args_list
        ]
        self.assertEqual(requested, ["flash_attention_2", "sdpa"])

    def test_falls_back_to_eager_as_last_resort(self):
        from_pretrained = load_with(
            cuda=False,
            flash_installed=False,
            side_effect=[RuntimeError("no sdpa kernel"), MagicMock()],
        )
        requested = [
            call.kwargs["attn_implementation"]
            for call in from_pretrained.call_args_list
        ]
        self.assertEqual(requested, ["sdpa", "eager"])

    def test_last_resort_failure_propagates(self):
        with self.assertRaises(RuntimeError):
            load_with(
                cuda=False,
                flash_installed=False,
                side_effect=[
                    RuntimeError("no sdpa kernel"),
                    RuntimeError("no eager either"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
