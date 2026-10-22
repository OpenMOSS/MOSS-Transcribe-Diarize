"""Subtitle and realtime applications.

``create_app`` is resolved lazily (PEP 562) so that importing a lightweight
submodule does not drag in ``server`` and, through it, torch.
``app.openai_audio_client`` in particular has to stay importable in a process
that has never loaded torch — that is the whole point of the realtime service's
slim deployment.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_ATTRIBUTES = {"create_app": "server"}

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
