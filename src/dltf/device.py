"""Device selection: CUDA > Apple MPS > CPU, with an AMP capability flag.

The framework never assumes a GPU exists; ``auto`` always resolves to something
usable and the resolved device is recorded in every checkpoint and artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute device plus its capabilities."""

    device: torch.device
    kind: str
    amp_dtype: torch.dtype | None
    notes: str = ""

    @property
    def supports_amp(self) -> bool:
        """True when autocast is safe (CUDA bf16/fp16, or CPU bf16)."""
        return self.amp_dtype is not None

    def as_dict(self) -> dict[str, str | bool]:
        """JSON-friendly description stored in run metadata."""
        return {
            "device": str(self.device),
            "kind": self.kind,
            "amp_dtype": str(self.amp_dtype) if self.amp_dtype else None,
            "supports_amp": self.supports_amp,
            "notes": self.notes,
        }


def cuda_available() -> bool:
    """True when CUDA is present and usable."""
    return torch.cuda.is_available()


def mps_available() -> bool:
    """True when the Apple Metal backend is present and usable."""
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def get_device(preference: str = "auto", amp: bool = True) -> DeviceInfo:
    """Resolve the compute device.

    Args:
        preference: ``auto``, ``cpu``, ``cuda`` or ``mps``.
        amp: Whether the caller wants mixed precision when available. The flag is
            reported back but the device is still returned when AMP is off.

    Raises:
        RuntimeError: if an explicitly requested device is unavailable.
    """
    pref = (preference or "auto").lower()
    if pref not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"device preference must be auto|cpu|cuda|mps, got {preference!r}")

    if pref == "cuda" or (pref == "auto" and cuda_available()):
        if not cuda_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        return DeviceInfo(torch.device("cuda", index), "cuda", torch.float16 if amp else None, name)

    if pref == "mps" or (pref == "auto" and mps_available()):
        if not mps_available():
            raise RuntimeError("MPS was requested but the Metal backend is unavailable")
        # AMP on MPS is not supported by this framework, so no autocast dtype.
        return DeviceInfo(torch.device("mps"), "mps", None, "Apple Silicon GPU (AMP disabled)")

    return DeviceInfo(torch.device("cpu"), "cpu", None, "CPU only run")
