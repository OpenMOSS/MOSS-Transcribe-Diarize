"""Attention implementation selection for Hugging Face inference.

Transformers can silently choose ``eager`` when a requested attention
implementation is unavailable.  That is particularly expensive for the
long audio prompts used by MOSS-Transcribe-Diarize, so model loading goes
through the explicit policy in this module and records every fallback.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


LOGGER = logging.getLogger(__name__)

AUTO_ATTENTION_IMPLEMENTATION = "auto"
ATTENTION_IMPLEMENTATIONS = (
    "flash_attention_4",
    "flash_attention_3",
    "flash_attention_2",
    "sdpa",
    "eager",
)
_FLASH_IMPLEMENTATIONS = ATTENTION_IMPLEMENTATIONS[:3]


def normalize_attention_implementation(value: str | None) -> str:
    """Normalize and validate the user-facing attention selector."""
    normalized = (value or AUTO_ATTENTION_IMPLEMENTATION).strip().lower()
    aliases = {
        "default": AUTO_ATTENTION_IMPLEMENTATION,
        "flash3": "flash_attention_3",
        "flash2": "flash_attention_2",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in (AUTO_ATTENTION_IMPLEMENTATION, *ATTENTION_IMPLEMENTATIONS):
        choices = ", ".join((AUTO_ATTENTION_IMPLEMENTATION, *ATTENTION_IMPLEMENTATIONS))
        raise ValueError(f"Unsupported attention implementation {value!r}; choose one of: {choices}.")
    return normalized


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _transformers_flash_available(implementation: str) -> bool | None:
    """Use Transformers' distribution-aware availability check when present."""
    version = implementation.rsplit("_", 1)[-1]
    try:
        from transformers import utils as transformers_utils

        checker = getattr(transformers_utils, f"is_flash_attn_{version}_available", None)
        if checker is None:
            return None
        return bool(checker())
    except Exception:  # noqa: BLE001 - older Transformers may not expose the checker
        return None


def _device_capability(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability(device))
    except (RuntimeError, AssertionError, ValueError):
        return None


def _flash_preflight(implementation: str, device: torch.device, dtype: torch.dtype) -> str | None:
    """Return a reason when an external FlashAttention candidate is unavailable."""
    if device.type != "cuda":
        return "requires a CUDA device"
    if dtype not in (torch.float16, torch.bfloat16):
        return f"requires float16/bfloat16, got {dtype}"

    capability = _device_capability(device)
    if capability is None:
        return "CUDA device capability could not be determined"
    major, _ = capability
    if implementation == "flash_attention_4" and major < 9:
        return f"requires compute capability >= 9.x, got {capability}"
    if implementation == "flash_attention_3" and major < 8:
        return f"requires compute capability >= 8.x, got {capability}"

    module_name = {
        "flash_attention_4": "flash_attn",
        "flash_attention_3": "flash_attn_interface",
        "flash_attention_2": "flash_attn",
    }[implementation]
    if not _module_available(module_name):
        return f"optional package/module {module_name!r} is not installed"
    reported_available = _transformers_flash_available(implementation)
    if reported_available is False:
        return f"Transformers reports {implementation} unavailable in this environment"
    return None


def _sdpa_available() -> bool:
    return callable(getattr(F, "scaled_dot_product_attention", None))


def _candidate_list(
    requested: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[str], list[dict[str, str]]]:
    """Build candidates and explain candidates skipped before model loading."""
    if requested != AUTO_ATTENTION_IMPLEMENTATION:
        return [requested], []

    candidates: list[str] = []
    attempts: list[dict[str, str]] = []
    for implementation in _FLASH_IMPLEMENTATIONS:
        reason = _flash_preflight(implementation, device, dtype)
        if reason is None:
            candidates.append(implementation)
            continue
        attempts.append({"backend": implementation, "status": "skipped", "reason": reason})
        LOGGER.info("[MOSS attention] skip %s: %s", implementation, reason)

    if _sdpa_available():
        candidates.append("sdpa")
    else:
        reason = "torch.nn.functional.scaled_dot_product_attention is unavailable"
        attempts.append({"backend": "sdpa", "status": "skipped", "reason": reason})
        LOGGER.warning("[MOSS attention] skip sdpa: %s", reason)

    # Eager is intentionally last.  It is retained as a compatibility escape
    # hatch, but selecting it for this long-context model is a warning-worthy
    # event rather than a silent Transformers default.
    candidates.append("eager")
    return candidates, attempts


def _config_attention_values(model: Any) -> dict[str, str | None]:
    config = getattr(model, "config", None)
    values: dict[str, str | None] = {}
    for name, item in (
        ("model", config),
        ("text", getattr(config, "text_config", None)),
        ("audio", getattr(config, "audio_config", None)),
    ):
        value = None
        if item is not None:
            value = getattr(item, "_attn_implementation", None)
            if value is None:
                value = getattr(item, "_attn_implementation_internal", None)
        values[name] = None if value is None else str(value)
    return values


def _contains_eager(values: dict[str, str | None]) -> bool:
    return any(value == "eager" or (value or "").endswith("|eager") for value in values.values())


def _attention_family(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.lower()
    if value == "eager" or value.endswith("|eager"):
        return "eager"
    for version in ("4", "3", "2"):
        if f"flash_attention_{version}" in value or f"flash-attn{version}" in value:
            return f"flash_attention_{version}"
    if value == "sdpa" or value.endswith("|sdpa"):
        return "sdpa"
    return None


def _resolution_mismatch(requested: str, values: dict[str, str | None]) -> str | None:
    """Return a mismatch description for a silently rewritten candidate."""
    families = {_attention_family(value) for value in values.values() if value is not None}
    families.discard(None)
    if not families:
        return None
    expected = _attention_family(requested)
    if expected is not None and families != {expected}:
        return f"requested {requested}, resolved families={sorted(families)} ({values})"
    return None


def _release_failed_model(model: Any, device: torch.device) -> None:
    if model is not None:
        del model
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must not hide the load error
            pass


def probe_sdpa_kernels(device: torch.device, dtype: torch.dtype) -> tuple[str, ...]:
    """Probe native SDPA kernels and return names that can execute.

    This is deliberately a small representative probe.  It tells the log
    which native kernels are available, while the SDPA dispatcher still owns
    the final per-shape decision during the real model forward.
    """
    if device.type != "cuda" or not _sdpa_available():
        return ("math",) if _sdpa_available() else ()
    if dtype not in (torch.float16, torch.bfloat16):
        return ("math",)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError):
        return ("math",)

    backends = (
        ("flash", getattr(SDPBackend, "FLASH_ATTENTION", None)),
        ("cudnn", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        ("efficient", getattr(SDPBackend, "EFFICIENT_ATTENTION", None)),
        ("math", getattr(SDPBackend, "MATH", None)),
    )
    query = key = value = None
    available: list[str] = []
    try:
        # Qwen3 uses 16 query heads and 8 KV heads.  Probe that GQA shape so
        # the report does not claim that a kernel works merely because it can
        # handle a same-head toy tensor.
        query = torch.randn((1, 16, 128, 128), device=device, dtype=dtype)
        key = torch.randn((1, 8, 128, 128), device=device, dtype=dtype)
        value = torch.randn_like(key)
        with torch.inference_mode():
            for name, backend in backends:
                if backend is None:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with sdpa_kernel(backend):
                            F.scaled_dot_product_attention(
                                query,
                                key,
                                value,
                                is_causal=True,
                                enable_gqa=True,
                            )
                    torch.cuda.synchronize(device)
                    available.append(name)
                except Exception:  # noqa: BLE001 - this is a capability probe
                    continue
    finally:
        del query, key, value
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must not hide a probe result
            pass
    return tuple(available)


@contextmanager
def attention_execution_context(report: dict[str, Any] | None):
    """Apply the probed SDPA kernel priority during an actual model forward."""
    if not report or report.get("selected") != "sdpa":
        with nullcontext():
            yield
        return

    device_name = report.get("device_type")
    kernels = tuple(report.get("sdpa_kernels") or ())
    if device_name != "cuda" or not kernels:
        with nullcontext():
            yield
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except (ImportError, AttributeError):
        with nullcontext():
            yield
        return

    backend_names = {
        "flash": "FLASH_ATTENTION",
        "cudnn": "CUDNN_ATTENTION",
        "efficient": "EFFICIENT_ATTENTION",
        "math": "MATH",
    }
    backends = [
        getattr(SDPBackend, backend_names[name])
        for name in kernels
        if name in backend_names and getattr(SDPBackend, backend_names[name], None) is not None
    ]
    if not backends:
        with nullcontext():
            yield
        return

    try:
        context = sdpa_kernel(backends, set_priority=True)
    except TypeError:  # Older torch versions do not expose set_priority.
        try:
            context = sdpa_kernel(backends)
        except Exception as exc:  # noqa: BLE001 - retain the default dispatcher
            LOGGER.warning("[MOSS attention] could not apply SDPA priority (%s); using the default dispatcher", exc)
            with nullcontext():
                yield
            return
    except Exception as exc:  # noqa: BLE001 - retain the default dispatcher
        LOGGER.warning("[MOSS attention] could not apply SDPA priority (%s); using the default dispatcher", exc)
        with nullcontext():
            yield
        return
    with context:
        yield


def load_model_with_attention_fallback(
    model_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requested: str = AUTO_ATTENTION_IMPLEMENTATION,
    model_loader: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a model with an explicit, logged attention fallback policy."""
    requested = normalize_attention_implementation(requested)
    candidates, attempts = _candidate_list(requested, device=device, dtype=dtype)
    loader = model_loader or AutoModelForCausalLM.from_pretrained
    selected_model = None

    for implementation in candidates:
        model = None
        try:
            model = loader(
                str(model_path),
                trust_remote_code=True,
                dtype="auto",
                attn_implementation=implementation,
            )
            config_values = _config_attention_values(model)
            # Transformers may silently turn an unavailable SDPA/Flash request
            # into eager.  Reject that candidate so the fallback is explicit.
            if implementation != "eager" and _contains_eager(config_values):
                raise RuntimeError(f"Transformers resolved {implementation} to eager: {config_values}")
            mismatch = _resolution_mismatch(implementation, config_values)
            if mismatch is not None:
                raise RuntimeError(f"Transformers rewrote the requested attention implementation: {mismatch}")

            selected_model = model
            report: dict[str, Any] = {
                "requested": requested,
                "selected": implementation,
                "config": config_values,
                "attempts": [*attempts, {"backend": implementation, "status": "selected"}],
            }
            if implementation == "sdpa":
                try:
                    kernels = probe_sdpa_kernels(device, dtype)
                except Exception as exc:  # noqa: BLE001 - probing must not force an eager fallback
                    LOGGER.warning(
                        "[MOSS attention] SDPA kernel probe failed (%s: %s); "
                        "keeping the SDPA dispatcher without forcing a kernel",
                        type(exc).__name__,
                        str(exc).splitlines()[0],
                    )
                    kernels = ()
                report["sdpa_kernels"] = list(kernels)
                report["device_type"] = device.type
                if device.type == "cuda" and (kernels == ("math",) or not any(kernel != "math" for kernel in kernels)):
                    LOGGER.warning(
                        "[MOSS attention] selected sdpa, but no fused CUDA kernel passed the probe; "
                        "long prompts may use quadratic math attention"
                    )
                else:
                    LOGGER.info("[MOSS attention] SDPA native kernel capability probe (preference order): %s", kernels)
            if implementation == "eager":
                LOGGER.warning(
                    "[MOSS attention] selected eager attention as the final fallback; "
                    "long audio can require quadratic attention memory"
                )
            else:
                LOGGER.info(
                    "[MOSS attention] selected %s (requested=%s, config=%s)",
                    implementation,
                    requested,
                    config_values,
                )
            return selected_model, report
        except Exception as exc:  # noqa: BLE001 - each candidate must be isolated
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            attempts.append({"backend": implementation, "status": "failed", "reason": reason})
            LOGGER.warning("[MOSS attention] %s unavailable; falling back: %s", implementation, reason)
            _release_failed_model(model, device)

    details = "; ".join(f"{item['backend']}: {item.get('reason', item['status'])}" for item in attempts)
    raise RuntimeError(f"No usable attention implementation was found ({details})")


__all__ = [
    "ATTENTION_IMPLEMENTATIONS",
    "AUTO_ATTENTION_IMPLEMENTATION",
    "attention_execution_context",
    "load_model_with_attention_fallback",
    "normalize_attention_implementation",
    "probe_sdpa_kernels",
]
