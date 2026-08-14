"""Public exception hierarchy."""

from __future__ import annotations


class RapidsChoirError(Exception):
    """Base class for package errors."""


class InvalidConfigurationError(RapidsChoirError, ValueError):
    """A requested configuration violates the frozen contract."""


class GPUUnavailableError(RapidsChoirError, RuntimeError):
    """The requested GPU runtime cannot be initialized."""


class UnsupportedHardwareError(GPUUnavailableError):
    """A CUDA device is present but below the supported capability."""


class GraphSemanticError(RapidsChoirError, ValueError):
    """A graph is missing, misaligned, or has ambiguous semantics."""


class InputResidencyError(RapidsChoirError, TypeError):
    """An input violates its declared host/device residency policy."""


class InsufficientDeviceMemoryError(RapidsChoirError, MemoryError):
    """Fixed-mode allocation would violate the configured VRAM headroom."""

    def __init__(
        self,
        *,
        estimated_bytes: int,
        free_bytes: int,
        headroom_fraction: float,
        required_bytes: int,
    ) -> None:
        self.estimated_bytes = estimated_bytes
        self.free_bytes = free_bytes
        self.headroom_fraction = headroom_fraction
        self.required_bytes = required_bytes
        message = (
            "comparison allocation preflight failed: "
            f"estimated={estimated_bytes} bytes, free={free_bytes} bytes, "
            f"headroom={headroom_fraction:.3f}, required={required_bytes} bytes"
        )
        super().__init__(message)


class RecordsUnavailableError(RapidsChoirError, LookupError):
    """Full comparison records cannot be resolved from the supplied object."""


class MissingCalibrationError(RapidsChoirError, RuntimeError):
    """No valid frozen RF calibration profile is available."""
