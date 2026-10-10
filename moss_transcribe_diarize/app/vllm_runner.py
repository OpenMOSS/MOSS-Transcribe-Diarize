from __future__ import annotations

import time
from pathlib import Path

from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, load_audio_item

from .model_runner import StatusCallback, TranscriptionResult, generation_progress
from .openai_audio_client import encode_wav_bytes, extract_transcription_text, transcribe_bytes


class VllmRunner:
    """Remote vLLM OpenAI-compatible audio transcription runner."""

    device_name = "vllm-api"
    dtype_name = "remote"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_path = model
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout

    @property
    def is_loaded(self) -> bool:
        return True

    def runtime_info(self) -> dict:
        return {
            "backend": "vllm",
            "path": self.model_path,
            "device": self.device_name,
            "dtype": self.dtype_name,
            "base_url": self.base_url,
        }

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_length: int = 131072,
        max_new_tokens: int = 2048,
        decoding: str = "greedy",
        temperature: float | None = None,
        status_callback: StatusCallback | None = None,
    ) -> TranscriptionResult:
        del max_length
        started = time.time()
        if status_callback is not None:
            status_callback("loading_model", 0.05, None)
        wav_bytes = _media_to_wav_bytes(audio_path)
        if status_callback is not None:
            status_callback("transcribing", 0.25, None)

        response = transcribe_bytes(
            base_url=self.base_url,
            model=self.model_path,
            prompt=prompt.strip() or DEFAULT_PROMPT,
            file_bytes=wav_bytes,
            filename="audio.wav",
            api_key=self.api_key,
            timeout=self.timeout,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
            on_progress=(
                None
                if status_callback is None
                else lambda tokens: status_callback(
                    "transcribing", generation_progress(tokens, max_new_tokens), tokens
                )
            ),
        )
        text = extract_transcription_text(response)
        usage = response.get("usage") or {}
        generated_tokens = int(usage.get("completion_tokens") or 0)
        prompt_len = int(usage.get("prompt_tokens") or 0)
        if status_callback is not None:
            status_callback("transcribing", 0.85, generated_tokens)
        return TranscriptionResult(
            text=text,
            prompt_len=prompt_len,
            generated_tokens=generated_tokens,
            elapsed_sec=time.time() - started,
            model=self.model_path,
            audio=str(Path(audio_path).expanduser()),
            decoding=decoding,
            temperature=temperature if decoding == "sample" else None,
        )


def _media_to_wav_bytes(path: str | Path) -> bytes:
    path = Path(path).expanduser()
    audio = load_audio_item(str(path), sampling_rate=16000)
    return encode_wav_bytes(audio, 16000)
