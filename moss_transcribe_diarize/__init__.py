"""MOSS-Transcribe-Diarize: inference code and remote-code model implementation.

Public names are resolved lazily (PEP 562). Importing a lightweight submodule
such as ``moss_transcribe_diarize.prompts`` must not pull in torch and
transformers through this module, so the heavy names are imported only when
they are actually accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRIBUTES: dict[str, str] = {
    # torch-free
    "DEFAULT_PROMPT": "prompts",
    "TranscriptParseError": "transcript_parser",
    "TranscriptSegment": "transcript_parser",
    "TranscriptStreamParser": "transcript_parser",
    "iter_transcript_segments": "transcript_parser",
    "parse_transcript": "transcript_parser",
    "SubtitleSegment": "subtitle",
    "SubtitleStyle": "subtitle",
    "coerce_subtitle_segments": "subtitle",
    "export_ass": "subtitle",
    "export_json": "subtitle",
    "export_srt": "subtitle",
    "normalize_segments": "subtitle",
    "subtitle_segments_from_transcript": "subtitle",
    # requires torch
    "MossTranscribeDiarizeConfig": "configuration_moss_transcribe_diarize",
    "MossTranscribeDiarizeForConditionalGeneration": "modeling_moss_transcribe_diarize",
    "MossTranscribeDiarizeModel": "modeling_moss_transcribe_diarize",
    "MossTranscribeDiarizePreTrainedModel": "modeling_moss_transcribe_diarize",
    "MossTranscribeDiarizeProcessor": "processing_moss_transcribe_diarize",
    "VQAdaptor": "modeling_moss_transcribe_diarize",
}

__all__ = sorted(_LAZY_ATTRIBUTES)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
