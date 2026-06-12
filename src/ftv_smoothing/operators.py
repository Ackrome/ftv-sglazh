"""Grunwald-Letnikov fractional gradient and divergence operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit, prange

from .backend import Backend, resolve_backend


def compute_gl_coefficients(alpha: float, k_size: int) -> np.ndarray:
    """Compute generalized GL binomial coefficients recursively.

    The returned coefficients are the exact truncated sequence
    ``(-1)**k * binomial(alpha, k)``. The operator applies a zero-sum short
    memory correction separately so that constants remain invariant.
    """

    if not 1.0 <= alpha <= 2.0:
        raise ValueError("alpha must be in [1.0, 2.0]")
    if k_size < 2:
        raise ValueError("k_size must be >= 2")
    weights = np.empty(k_size, dtype=np.float32)
    weights[0] = 1.0
    for k in range(1, k_size):
        weights[k] = weights[k - 1] * (1.0 - (alpha + 1.0) / k)
    return weights


def _zero_sum_short_memory(weights: np.ndarray) -> np.ndarray:
    """Remove finite-window constant bias while preserving GL tail shape."""

    corrected = np.asarray(weights, dtype=np.float32).copy()
    corrected[0] -= corrected.sum(dtype=np.float32)
    return corrected


@njit(parallel=True, fastmath=True, cache=True)
def _gradient_wrap_cpu(
    array: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = array.shape
    gx = np.empty_like(array)
    gy = np.empty_like(array)
    for row in prange(rows):
        for col in range(cols):
            sum_x = 0.0
            sum_y = 0.0
            for k in range(weights.size):
                sum_x += weights[k] * array[row, (col - k) % cols]
                sum_y += weights[k] * array[(row - k) % rows, col]
            gx[row, col] = sum_x
            gy[row, col] = sum_y
    return gx, gy


@njit(parallel=True, fastmath=True, cache=True)
def _divergence_wrap_cpu(
    px: np.ndarray,
    py: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    rows, cols = px.shape
    result = np.empty_like(px)
    for row in prange(rows):
        for col in range(cols):
            value = 0.0
            for k in range(weights.size):
                value -= weights[k] * (
                    px[row, (col + k) % cols] + py[(row + k) % rows, col]
                )
            result[row, col] = value
    return result


def _correlation_kernel(weights: np.ndarray, forward: bool) -> np.ndarray:
    """Build an odd kernel for centered backend correlation calls."""

    k_size = int(weights.size)
    center = k_size - 1
    kernel = np.zeros(2 * k_size - 1, dtype=np.float32)
    for k, weight in enumerate(weights):
        kernel[center + k if forward else center - k] = weight
    return kernel


@dataclass(slots=True)
class FractionalOperator:
    """Fractional GL gradient and negative-adjoint divergence."""

    alpha: float = 1.5
    k_size: int = 12
    backend: Backend | str = "cpu"
    method: str = "auto"
    preserve_constants: bool = True
    weights: np.ndarray = field(init=False)
    _backward_kernel: Any = field(init=False, default=None)
    _forward_kernel: Any = field(init=False, default=None)
    _symbols: dict[tuple[int, int], tuple[Any, Any]] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            self.backend = resolve_backend(self.backend)
        if self.method not in {"auto", "direct", "fft"}:
            raise ValueError("method must be auto, direct, or fft")
        raw_weights = compute_gl_coefficients(self.alpha, self.k_size)
        self.weights = (
            _zero_sum_short_memory(raw_weights) if self.preserve_constants else raw_weights
        )
        if self.backend.gpu:
            self._backward_kernel = self.backend.asarray(
                _correlation_kernel(self.weights, forward=False)
            )
            self._forward_kernel = self.backend.asarray(
                _correlation_kernel(self.weights, forward=True)
            )

    @property
    def effective_method(self) -> str:
        """Return the chosen direct or FFT implementation."""

        if self.method == "auto":
            return "direct" if self.k_size <= 32 else "fft"
        return self.method

    @property
    def norm_bound(self) -> float:
        """Conservative spectral norm bound for PDHG step sizing."""

        return float(np.sqrt(2.0) * np.abs(self.weights).sum(dtype=np.float64))

    def gradient(self, array: Any) -> tuple[Any, Any]:
        """Apply the two-dimensional fractional gradient."""

        if self.effective_method == "fft":
            return self._gradient_fft(array)
        if not self.backend.gpu:
            return _gradient_wrap_cpu(array, self.weights)

        from cupyx.scipy import ndimage as cndi

        gx = cndi.correlate1d(array, self._backward_kernel, axis=1, mode="wrap")
        gy = cndi.correlate1d(array, self._backward_kernel, axis=0, mode="wrap")
        return gx, gy

    def divergence(self, px: Any, py: Any) -> Any:
        """Apply negative adjoint divergence: ``-K.T @ (px, py)``."""

        if self.effective_method == "fft":
            return self._divergence_fft(px, py)
        if not self.backend.gpu:
            return _divergence_wrap_cpu(px, py, self.weights)

        from cupyx.scipy import ndimage as cndi

        dx = cndi.correlate1d(px, self._forward_kernel, axis=1, mode="wrap")
        dy = cndi.correlate1d(py, self._forward_kernel, axis=0, mode="wrap")
        return -(dx + dy)

    def _get_symbols(self, shape: tuple[int, int]) -> tuple[Any, Any]:
        xp = self.backend.xp
        if shape not in self._symbols:
            rows, cols = shape
            kernel_x = xp.zeros(cols, dtype=xp.float32)
            kernel_y = xp.zeros(rows, dtype=xp.float32)
            weights = xp.asarray(self.weights)
            kernel_x[: self.k_size] = weights
            kernel_y[: self.k_size] = weights
            self._symbols[shape] = (
                xp.fft.rfft(kernel_x),
                xp.fft.rfft(kernel_y),
            )
        return self._symbols[shape]

    def _gradient_fft(self, array: Any) -> tuple[Any, Any]:
        xp = self.backend.xp
        symbol_x, symbol_y = self._get_symbols(array.shape)
        gx = xp.fft.irfft(
            xp.fft.rfft(array, axis=1) * symbol_x[None, :],
            n=array.shape[1],
            axis=1,
        ).astype(xp.float32)
        gy = xp.fft.irfft(
            xp.fft.rfft(array, axis=0) * symbol_y[:, None],
            n=array.shape[0],
            axis=0,
        ).astype(xp.float32)
        return gx, gy

    def _divergence_fft(self, px: Any, py: Any) -> Any:
        xp = self.backend.xp
        symbol_x, symbol_y = self._get_symbols(px.shape)
        dx = xp.fft.irfft(
            xp.fft.rfft(px, axis=1) * xp.conj(symbol_x)[None, :],
            n=px.shape[1],
            axis=1,
        )
        dy = xp.fft.irfft(
            xp.fft.rfft(py, axis=0) * xp.conj(symbol_y)[:, None],
            n=px.shape[0],
            axis=0,
        )
        return -(dx + dy).astype(xp.float32)


def apply_fractional_gradient(
    u: np.ndarray,
    gl_weights_x: np.ndarray,
    gl_weights_y: np.ndarray | None = None,
    backend: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a direct GL gradient using explicit weights.

    This compatibility API keeps the signature described in the specification.
    The optimized solver uses :class:`FractionalOperator`.
    """

    if gl_weights_y is not None and not np.array_equal(gl_weights_x, gl_weights_y):
        raise ValueError("Anisotropic GL weights are not supported")
    resolved = resolve_backend(backend)
    operator = FractionalOperator(
        alpha=1.0,
        k_size=len(gl_weights_x),
        backend=resolved,
        preserve_constants=False,
    )
    operator.weights = np.asarray(gl_weights_x, dtype=np.float32)
    if resolved.gpu:
        operator._backward_kernel = resolved.asarray(
            _correlation_kernel(operator.weights, forward=False)
        )
        operator._forward_kernel = resolved.asarray(
            _correlation_kernel(operator.weights, forward=True)
        )
        gx, gy = operator.gradient(resolved.asarray(u))
        return resolved.to_numpy(gx), resolved.to_numpy(gy)
    return operator.gradient(np.asarray(u, dtype=np.float32))


def apply_fractional_divergence(
    px: np.ndarray,
    py: np.ndarray,
    gl_weights_x: np.ndarray,
    gl_weights_y: np.ndarray | None = None,
    backend: str = "cpu",
) -> np.ndarray:
    """Apply the negative-adjoint GL divergence using explicit weights."""

    if gl_weights_y is not None and not np.array_equal(gl_weights_x, gl_weights_y):
        raise ValueError("Anisotropic GL weights are not supported")
    resolved = resolve_backend(backend)
    operator = FractionalOperator(
        alpha=1.0,
        k_size=len(gl_weights_x),
        backend=resolved,
        preserve_constants=False,
    )
    operator.weights = np.asarray(gl_weights_x, dtype=np.float32)
    if resolved.gpu:
        operator._backward_kernel = resolved.asarray(
            _correlation_kernel(operator.weights, forward=False)
        )
        operator._forward_kernel = resolved.asarray(
            _correlation_kernel(operator.weights, forward=True)
        )
        result = operator.divergence(resolved.asarray(px), resolved.asarray(py))
        return resolved.to_numpy(result)
    return operator.divergence(
        np.asarray(px, dtype=np.float32),
        np.asarray(py, dtype=np.float32),
    )

