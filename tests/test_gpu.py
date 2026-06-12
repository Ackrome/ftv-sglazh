import numpy as np
import pytest
from numpy.testing import assert_allclose

from ftv_smoothing import backend as backend_module
from ftv_smoothing.backend import (
    GPUDeviceInfo,
    GPUDiagnostics,
    backend_diagnostics,
    resolve_backend,
)
from ftv_smoothing.operators import FractionalOperator


def test_backend_diagnostics_has_stable_public_shape() -> None:
    diagnostics = backend_diagnostics("auto")

    assert diagnostics["requested"] == "auto"
    assert diagnostics["selected_backend"]
    assert isinstance(diagnostics["selected_gpu"], bool)
    assert isinstance(diagnostics["reason"], str)
    assert isinstance(diagnostics["gpu"]["devices"], list)


def test_auto_diagnostics_prefers_usable_gpu_with_most_free_memory(monkeypatch) -> None:
    devices = (
        GPUDeviceInfo(
            id=0,
            name="Small GPU",
            compute_capability="8.0",
            memory_free_bytes=2,
            memory_total_bytes=8,
            usable=True,
            reason="ok",
        ),
        GPUDeviceInfo(
            id=1,
            name="Blocked GPU",
            compute_capability="8.0",
            memory_free_bytes=32,
            memory_total_bytes=32,
            usable=False,
            reason="probe failed",
        ),
        GPUDeviceInfo(
            id=2,
            name="Large GPU",
            compute_capability="8.9",
            memory_free_bytes=16,
            memory_total_bytes=24,
            usable=True,
            reason="ok",
        ),
    )

    def fake_diagnostics() -> GPUDiagnostics:
        return GPUDiagnostics(
            cupy_available=True,
            cupy_version="test",
            cuda_runtime_version="12.0",
            cuda_driver_version="12.0",
            devices=devices,
            selected_device_id=2,
            reason="Selected Large GPU on cuda:2",
        )

    monkeypatch.setattr(backend_module, "diagnose_gpu_stack", fake_diagnostics)

    diagnostics = backend_diagnostics("auto")

    assert diagnostics["selected_backend"] == "cupy-cuda:2"
    assert diagnostics["selected_gpu"] is True
    assert diagnostics["selected_device_id"] == 2


def test_explicit_gpu_does_not_fallback_to_cpu_silently() -> None:
    try:
        backend = resolve_backend("gpu")
    except RuntimeError as exc:
        assert "GPU backend requested" in str(exc)
    else:
        assert backend.gpu
        assert backend.name.startswith("cupy-cuda:")


def test_gpu_direct_operator_matches_cpu_when_cuda_is_available() -> None:
    try:
        gpu_backend = resolve_backend("gpu")
    except Exception as exc:  # pragma: no cover - machine-specific skip
        pytest.skip(f"CuPy unavailable: {exc}")
    if not gpu_backend.gpu:
        pytest.skip("CuPy unavailable")

    rng = np.random.default_rng(8)
    array = rng.normal(size=(47, 53)).astype(np.float32)
    px = rng.normal(size=array.shape).astype(np.float32)
    py = rng.normal(size=array.shape).astype(np.float32)
    cpu = FractionalOperator(alpha=1.5, k_size=12, backend="cpu")
    gpu = FractionalOperator(alpha=1.5, k_size=12, backend=gpu_backend)
    cpu_gx, cpu_gy = cpu.gradient(array)
    cpu_divergence = cpu.divergence(px, py)
    gpu_gx, gpu_gy = gpu.gradient(gpu_backend.asarray(array))
    gpu_divergence = gpu.divergence(gpu_backend.asarray(px), gpu_backend.asarray(py))
    assert_allclose(gpu_backend.to_numpy(gpu_gx), cpu_gx, atol=2e-5)
    assert_allclose(gpu_backend.to_numpy(gpu_gy), cpu_gy, atol=2e-5)
    assert_allclose(gpu_backend.to_numpy(gpu_divergence), cpu_divergence, atol=2e-5)
