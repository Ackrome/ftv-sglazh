"""CPU and GPU backend selection."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, ContextManager

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Backend:
    """Array backend used by the iterative optimizer."""

    name: str
    xp: Any
    gpu: bool
    device_id: int | None = None

    def _device_context(self) -> ContextManager[Any]:
        if not self.gpu or self.device_id is None:
            return nullcontext()
        return self.xp.cuda.Device(self.device_id)

    def asarray(self, array: np.ndarray) -> Any:
        """Move or convert an array to this backend."""

        with self._device_context():
            return self.xp.asarray(array, dtype=self.xp.float32)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Move an array to host memory as ``float32``."""

        if self.gpu:
            with self._device_context():
                return self.xp.asnumpy(array).astype(np.float32, copy=False)
        return np.asarray(array, dtype=np.float32)

    def synchronize(self) -> None:
        """Synchronize GPU work before timing or host transfers."""

        if self.gpu:
            with self._device_context():
                self.xp.cuda.Stream.null.synchronize()

    def free_memory_pool(self) -> None:
        """Release cached GPU allocator blocks."""

        if self.gpu:
            with self._device_context():
                self.xp.get_default_memory_pool().free_all_blocks()

    def allocation_context(self, estimated_bytes: int) -> ContextManager[Any]:
        """Use CUDA managed memory when estimated allocations exceed VRAM."""

        if not self.gpu:
            return nullcontext()

        with self._device_context():
            free_bytes, total_bytes = self.xp.cuda.runtime.memGetInfo()
        LOGGER.info(
            "GPU memory before FTV: %.2f GiB free / %.2f GiB total",
            free_bytes / 2**30,
            total_bytes / 2**30,
        )
        if estimated_bytes <= int(free_bytes * 0.8):
            return nullcontext()

        LOGGER.warning(
            "Estimated GPU allocation %.2f GiB exceeds the VRAM safety budget; "
            "switching to CUDA managed memory",
            estimated_bytes / 2**30,
        )
        managed_pool = self.xp.cuda.MemoryPool(self.xp.cuda.malloc_managed)
        return self.xp.cuda.using_allocator(managed_pool.malloc)


@dataclass(frozen=True, slots=True)
class GPUDeviceInfo:
    """Runtime probe result for one CUDA device."""

    id: int
    name: str
    compute_capability: str | None
    memory_free_bytes: int | None
    memory_total_bytes: int | None
    usable: bool
    reason: str
    warning: str | None = None


@dataclass(frozen=True, slots=True)
class GPUDiagnostics:
    """CUDA/CuPy availability and backend selection diagnostics."""

    cupy_available: bool
    cupy_version: str | None
    cuda_runtime_version: str | None
    cuda_driver_version: str | None
    devices: tuple[GPUDeviceInfo, ...]
    selected_device_id: int | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostics payload."""

        return {
            "cupy_available": self.cupy_available,
            "cupy_version": self.cupy_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "devices": [asdict(device) for device in self.devices],
            "selected_device_id": self.selected_device_id,
            "reason": self.reason,
        }


def _cpu_backend() -> Backend:
    return Backend(name="numba-cpu", xp=np, gpu=False)


def _exception_text(exc: Exception) -> str:
    text = str(exc).strip()
    return f"{exc.__class__.__name__}: {text}" if text else exc.__class__.__name__


def _format_cuda_version(value: int | None) -> str | None:
    if value is None or value <= 0:
        return None
    major = value // 1000
    minor = (value % 1000) // 10
    patch = value % 10
    if patch:
        return f"{major}.{minor}.{patch}"
    return f"{major}.{minor}"


def _runtime_version(cp: Any, function_name: str) -> str | None:
    try:
        raw = getattr(cp.cuda.runtime, function_name)()
    except Exception:
        return None
    return _format_cuda_version(int(raw))


def _property(props: Any, name: str, default: Any = None) -> Any:
    if isinstance(props, dict):
        return props.get(name, default)
    return getattr(props, name, default)


def _device_name(raw: Any, fallback: str) -> str:
    if isinstance(raw, bytes):
        return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace") or fallback
    text = str(raw or fallback).strip("\0")
    return text or fallback


def _probe_device(cp: Any, device_id: int) -> GPUDeviceInfo:
    name = f"cuda:{device_id}"
    compute_capability = None
    memory_free_bytes: int | None = None
    memory_total_bytes: int | None = None
    warning = None

    try:
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = _device_name(_property(props, "name"), name)
        major = int(_property(props, "major", 0) or 0)
        minor = int(_property(props, "minor", 0) or 0)
        if major:
            compute_capability = f"{major}.{minor}"
        total_global = _property(props, "totalGlobalMem")
        if total_global is not None:
            memory_total_bytes = int(total_global)
        if major and major < 5:
            warning = (
                "Legacy compute capability; current CUDA 12+/13 CuPy wheels "
                "usually cannot run this device"
            )
    except Exception as exc:
        return GPUDeviceInfo(
            id=device_id,
            name=name,
            compute_capability=compute_capability,
            memory_free_bytes=memory_free_bytes,
            memory_total_bytes=memory_total_bytes,
            usable=False,
            reason=f"Device properties unavailable: {_exception_text(exc)}",
            warning=warning,
        )

    try:
        with cp.cuda.Device(device_id):
            memory_free_bytes, memory_total_bytes = (
                int(value) for value in cp.cuda.runtime.memGetInfo()
            )
            probe = cp.arange(8, dtype=cp.float32)
            probe = probe + cp.float32(1)
            float(cp.asnumpy(probe[:1])[0])
            cp.cuda.Stream.null.synchronize()
            del probe
            cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        return GPUDeviceInfo(
            id=device_id,
            name=name,
            compute_capability=compute_capability,
            memory_free_bytes=memory_free_bytes,
            memory_total_bytes=memory_total_bytes,
            usable=False,
            reason=f"CuPy runtime probe failed: {_exception_text(exc)}",
            warning=warning,
        )

    return GPUDeviceInfo(
        id=device_id,
        name=name,
        compute_capability=compute_capability,
        memory_free_bytes=memory_free_bytes,
        memory_total_bytes=memory_total_bytes,
        usable=True,
        reason="CuPy runtime probe succeeded",
        warning=warning,
    )


def _best_usable_device(devices: tuple[GPUDeviceInfo, ...]) -> GPUDeviceInfo | None:
    usable = [device for device in devices if device.usable]
    if not usable:
        return None
    return max(
        usable,
        key=lambda device: (
            device.memory_free_bytes or 0,
            device.memory_total_bytes or 0,
            _compute_capability_key(device.compute_capability),
        ),
    )


def _compute_capability_key(value: str | None) -> tuple[int, int]:
    if not value:
        return (0, 0)
    try:
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except ValueError:
        return (0, 0)


def diagnose_gpu_stack() -> GPUDiagnostics:
    """Probe CuPy and visible CUDA devices without requiring a GPU."""

    try:
        import cupy as cp
    except ModuleNotFoundError:
        return GPUDiagnostics(
            cupy_available=False,
            cupy_version=None,
            cuda_runtime_version=None,
            cuda_driver_version=None,
            devices=(),
            selected_device_id=None,
            reason="CuPy is not installed",
        )
    except Exception as exc:
        return GPUDiagnostics(
            cupy_available=False,
            cupy_version=None,
            cuda_runtime_version=None,
            cuda_driver_version=None,
            devices=(),
            selected_device_id=None,
            reason=f"CuPy import failed: {_exception_text(exc)}",
        )

    cupy_version = str(getattr(cp, "__version__", "unknown"))
    runtime_version = _runtime_version(cp, "runtimeGetVersion")
    driver_version = _runtime_version(cp, "driverGetVersion")

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return GPUDiagnostics(
            cupy_available=True,
            cupy_version=cupy_version,
            cuda_runtime_version=runtime_version,
            cuda_driver_version=driver_version,
            devices=(),
            selected_device_id=None,
            reason=f"CUDA device query failed: {_exception_text(exc)}",
        )

    if device_count < 1:
        return GPUDiagnostics(
            cupy_available=True,
            cupy_version=cupy_version,
            cuda_runtime_version=runtime_version,
            cuda_driver_version=driver_version,
            devices=(),
            selected_device_id=None,
            reason="No CUDA devices are visible",
        )

    devices = tuple(_probe_device(cp, index) for index in range(device_count))
    selected = _best_usable_device(devices)
    if selected is None:
        reason = "No visible CUDA device passed the CuPy runtime probe"
        if devices:
            reason = f"{reason}: {devices[0].reason}"
        selected_device_id = None
    else:
        reason = f"Selected {selected.name} on cuda:{selected.id}"
        selected_device_id = selected.id
    return GPUDiagnostics(
        cupy_available=True,
        cupy_version=cupy_version,
        cuda_runtime_version=runtime_version,
        cuda_driver_version=driver_version,
        devices=devices,
        selected_device_id=selected_device_id,
        reason=reason,
    )


def backend_diagnostics(requested: str = "auto") -> dict[str, Any]:
    """Return public diagnostics for the requested compute mode."""

    if requested not in {"auto", "gpu", "cpu"}:
        raise ValueError("requested backend must be auto, gpu, or cpu")

    gpu = diagnose_gpu_stack()
    selected_device = _best_usable_device(gpu.devices)
    selected_backend = "numba-cpu"
    selected_gpu = False
    selected_device_id = None

    if requested == "cpu":
        reason = "CPU backend requested"
    elif selected_device is not None:
        selected_backend = f"cupy-cuda:{selected_device.id}"
        selected_gpu = True
        selected_device_id = selected_device.id
        reason = gpu.reason
    elif requested == "gpu":
        reason = f"GPU backend requested but unavailable: {gpu.reason}"
    else:
        reason = f"GPU unavailable; using CPU backend: {gpu.reason}"

    return {
        "requested": requested,
        "selected_backend": selected_backend,
        "selected_gpu": selected_gpu,
        "selected_device_id": selected_device_id,
        "reason": reason,
        "gpu": gpu.to_dict(),
    }


def _load_gpu_backend(device_id: int) -> Backend:
    """Import CuPy and force a tiny operation on the selected device."""

    import cupy as cp

    with cp.cuda.Device(device_id):
        probe = cp.arange(2, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        del probe
    return Backend(name=f"cupy-cuda:{device_id}", xp=cp, gpu=True, device_id=device_id)


def resolve_backend(requested: str = "auto") -> Backend:
    """Resolve ``auto``, ``gpu``, or ``cpu`` to a working array backend."""

    if requested not in {"auto", "gpu", "cpu"}:
        raise ValueError("requested backend must be auto, gpu, or cpu")
    if requested == "cpu":
        return _cpu_backend()

    diagnostics = diagnose_gpu_stack()
    selected = _best_usable_device(diagnostics.devices)
    if selected is None:
        if requested == "gpu":
            raise RuntimeError(
                f"GPU backend requested but no usable CUDA device is available: "
                f"{diagnostics.reason}"
            )
        LOGGER.warning(
            "GPU backend unavailable (%s); falling back to Numba CPU",
            diagnostics.reason,
        )
        return _cpu_backend()

    try:
        return _load_gpu_backend(selected.id)
    except Exception as exc:  # pragma: no cover - machine-specific fallback
        if requested == "gpu":
            raise RuntimeError(
                f"GPU backend requested but cuda:{selected.id} failed to initialize: "
                f"{_exception_text(exc)}"
            ) from exc
        LOGGER.warning(
            "Selected GPU cuda:%d failed to initialize (%s); falling back to Numba CPU",
            selected.id,
            _exception_text(exc),
        )
        return _cpu_backend()
