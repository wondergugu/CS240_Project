"""Resolve a requested compute device, falling back to CPU when no GPU is present.

A device string like "cuda:0" is honored when CUDA is available; otherwise the
run transparently falls back to CPU instead of crashing. All inference stays
local regardless of where it runs.
"""

from __future__ import annotations

import warnings


def resolve_device(device: str | None = None) -> str:
    """Return a concrete torch device string, falling back to CPU if needed.

    None                          -> "cuda:0" when a GPU is available, else "cpu".
    "cuda" / "cuda:N" with no GPU  -> "cpu" (with a warning).
    anything else                  -> returned unchanged.
    """
    import torch

    available = torch.cuda.is_available()
    if device is None:
        return "cuda:0" if available else "cpu"
    if device.startswith("cuda") and not available:
        warnings.warn(
            f"requested device {device!r} but CUDA is unavailable; using CPU"
        )
        return "cpu"
    return device
