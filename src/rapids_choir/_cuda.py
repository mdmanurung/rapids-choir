"""Lazy CUDA/RMM initialization and device assertions."""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from .exceptions import (
    GPUUnavailableError,
    InputResidencyError,
    UnsupportedHardwareError,
)

MINIMUM_COMPUTE_CAPABILITY = (7, 0)
_INITIALIZATION_LOCK = threading.Lock()
_CONTEXTS: dict[int, CUDAContext] = {}


@dataclass(frozen=True, slots=True)
class CUDAContext:
    device_index: int
    cupy: ModuleType
    rmm: ModuleType
    compute_capability: tuple[int, int]
    device_name: str

    def memory_info(self) -> tuple[int, int]:
        with self.cupy.cuda.Device(self.device_index):
            free_bytes, total_bytes = self.cupy.cuda.runtime.memGetInfo()
        return int(free_bytes), int(total_bytes)


def parse_device(device: str) -> int:
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("device must have the form 'cuda:<index>'")
    return int(device[5:])


def _load_gpu_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except Exception as error:
        raise GPUUnavailableError(
            f"could not import GPU dependency {name!r}: {error}"
        ) from error


def initialize_cuda(device: str = "cuda:0") -> CUDAContext:
    """Initialize the requested CUDA device and the RMM-backed CuPy allocator."""

    device_index = parse_device(device)
    with _INITIALIZATION_LOCK:
        if device_index in _CONTEXTS:
            return _CONTEXTS[device_index]

        cp = _load_gpu_module("cupy")
        rmm = _load_gpu_module("rmm")
        try:
            cp.cuda.Device(device_index).use()
            properties: dict[str | bytes, Any] = cp.cuda.runtime.getDeviceProperties(
                device_index
            )
            major_value = properties.get("major", properties.get(b"major"))
            minor_value = properties.get("minor", properties.get(b"minor"))
            if major_value is None or minor_value is None:
                raise RuntimeError("CUDA device properties omit compute capability")
            major = int(major_value)
            minor = int(minor_value)
            raw_name = properties.get("name", properties.get(b"name", "unknown"))
            device_name = (
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            capability = (major, minor)
            if capability < MINIMUM_COMPUTE_CAPABILITY:
                required_capability = ".".join(
                    str(component) for component in MINIMUM_COMPUTE_CAPABILITY
                )
                raise UnsupportedHardwareError(
                    f"{device_name} has compute capability {major}.{minor}; "
                    f">={required_capability} is required"
                )

            rmm.reinitialize(
                managed_memory=False,
                pool_allocator=True,
                devices=[device_index],
            )
            allocator_module = _load_gpu_module("rmm.allocators.cupy")
            cp.cuda.set_allocator(allocator_module.rmm_cupy_allocator)
            probe = cp.empty(1, dtype=cp.uint8)
            del probe
        except UnsupportedHardwareError:
            raise
        except Exception as error:
            raise GPUUnavailableError(
                f"failed to initialize CUDA device {device_index}: {error}"
            ) from error

        context = CUDAContext(
            device_index=device_index,
            cupy=cp,
            rmm=rmm,
            compute_capability=capability,
            device_name=device_name,
        )
        _CONTEXTS[device_index] = context
        return context


def assert_device_array(
    value: Any,
    *,
    name: str,
    dtype: Any | None = None,
    ndim: int | None = None,
    contiguous: bool = False,
) -> None:
    """Require a CuPy dense array without copying it."""

    cp = _load_gpu_module("cupy")
    if not isinstance(value, cp.ndarray):
        raise InputResidencyError(f"{name} must be a CuPy device array")
    if dtype is not None and value.dtype != dtype:
        raise InputResidencyError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if ndim is not None and value.ndim != ndim:
        raise InputResidencyError(f"{name} must have {ndim} dimensions")
    if contiguous and not value.flags.c_contiguous:
        raise InputResidencyError(f"{name} must be C-contiguous")


def require_cupy_csr(value: Any, *, name: str) -> None:
    sparse = _load_gpu_module("cupyx.scipy.sparse")
    if not sparse.isspmatrix_csr(value):
        raise InputResidencyError(f"{name} must be a CuPy CSR matrix")
