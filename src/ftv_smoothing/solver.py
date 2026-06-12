"""Chambolle-Pock optimizer for fractional total variation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .artifacts import generate_sarp_weights
from .backend import resolve_backend
from .config import FTVConfig
from .operators import FractionalOperator

LOGGER = logging.getLogger(__name__)

SnapshotCallback = Callable[[int, np.ndarray, float], None]
OptimizerProgressCallback = Callable[[int, int, float | None], None]


@dataclass(slots=True)
class FTVResult:
    """Optimizer output and diagnostics."""

    corrected_dem: np.ndarray
    fidelity_weight: np.ndarray
    backend: str
    iterations: int
    converged: bool
    relative_errors: list[tuple[int, float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    operator_norm_bound: float = 0.0
    tau: float = 0.0
    sigma: float = 0.0


def _relative_error(xp: object, current: object, previous: object) -> float:
    numerator = xp.linalg.norm(current - previous)
    denominator = xp.maximum(xp.linalg.norm(previous), xp.asarray(1e-6, dtype=xp.float32))
    return float(numerator / denominator)


def _estimate_gpu_bytes(dem: np.ndarray) -> int:
    """Estimate persistent and temporary peak storage for one optimizer step."""

    return int(dem.nbytes * 18)


def _run_pdhg(
    dem: np.ndarray,
    fidelity_weight: np.ndarray,
    config: FTVConfig,
    snapshot_callback: SnapshotCallback | None,
    progress_callback: OptimizerProgressCallback | None,
) -> FTVResult:
    backend = resolve_backend(config.backend)
    operator = FractionalOperator(
        alpha=config.alpha,
        k_size=config.k_size,
        backend=backend,
        method=config.convolution_method,
    )
    norm_bound = max(operator.norm_bound, 1e-6)
    tau = float(config.tau if config.tau is not None else 0.95 / norm_bound)
    sigma = float(config.sigma if config.sigma is not None else 0.95 / norm_bound)
    if tau * sigma * norm_bound * norm_bound >= 1.0:
        raise ValueError("PDHG step sizes violate tau * sigma * ||K||^2 < 1")

    LOGGER.info(
        "FTV PDHG started on backend %s (%s convolution), iterations=%d, "
        "alpha=%.3f, k=%d, tau=%.6f, sigma=%.6f",
        backend.name,
        operator.effective_method,
        config.max_iter,
        config.alpha,
        config.k_size,
        tau,
        sigma,
    )
    errors: list[tuple[int, float]] = []
    converged = False
    completed_iterations = 0
    start = time.perf_counter()

    with backend.allocation_context(_estimate_gpu_bytes(dem)):
        xp = backend.xp
        source = backend.asarray(dem)
        u = source.copy()
        u_bar = u.copy()
        px = xp.zeros_like(u)
        py = xp.zeros_like(u)
        fidelity = backend.asarray(fidelity_weight)
        denominator = 1.0 + tau * fidelity
        progress_every = max(1, config.max_iter // 100)

        for iteration in range(1, config.max_iter + 1):
            gx, gy = operator.gradient(u_bar)
            px += sigma * gx
            py += sigma * gy
            dual_norm = xp.maximum(1.0, xp.sqrt(px * px + py * py))
            px /= dual_norm
            py /= dual_norm
            del gx, gy, dual_norm

            divergence = operator.divergence(px, py)
            previous = u
            u = (u + tau * divergence + tau * fidelity * source) / denominator
            u_bar = u + config.theta * (u - previous)
            del divergence
            completed_iterations = iteration

            should_check = iteration == 1 or iteration % config.log_every == 0
            should_snapshot = (
                snapshot_callback is not None
                and config.snapshot_every > 0
                and iteration % config.snapshot_every == 0
            )
            should_report_progress = (
                progress_callback is not None
                and (
                    iteration == 1
                    or iteration % progress_every == 0
                    or iteration == config.max_iter
                )
            )
            progress_error: float | None = None
            if should_check or should_snapshot:
                backend.synchronize()
                error = _relative_error(xp, u, previous)
                progress_error = error
                if should_check:
                    errors.append((iteration, error))
                    LOGGER.debug("FTV iteration %d relative error %.8e", iteration, error)
                if should_snapshot:
                    snapshot_callback(iteration, backend.to_numpy(u), error)
                if should_check and error < config.tol:
                    converged = True
                    LOGGER.info("FTV converged at iteration %d (%.8e)", iteration, error)
                    if should_report_progress:
                        progress_callback(iteration, config.max_iter, progress_error)
                    break

            if should_report_progress:
                progress_callback(iteration, config.max_iter, progress_error)

            if (
                backend.gpu
                and config.memory_cleanup_every > 0
                and iteration % config.memory_cleanup_every == 0
            ):
                backend.free_memory_pool()

        backend.synchronize()
        corrected = backend.to_numpy(u)

    elapsed = time.perf_counter() - start
    backend.free_memory_pool()
    LOGGER.info("FTV PDHG finished in %.2f seconds", elapsed)
    return FTVResult(
        corrected_dem=corrected,
        fidelity_weight=fidelity_weight,
        backend=backend.name,
        iterations=completed_iterations,
        converged=converged,
        relative_errors=errors,
        elapsed_seconds=elapsed,
        operator_norm_bound=norm_bound,
        tau=tau,
        sigma=sigma,
    )


def fractional_total_variation_denoise(
    dem: np.ndarray,
    artifact_mask: np.ndarray,
    slope: np.ndarray,
    alpha: float = 1.5,
    lambda_base: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    use_gpu: bool = True,
    *,
    max_curvature: np.ndarray | None = None,
    config: FTVConfig | None = None,
    valid_mask: np.ndarray | None = None,
    snapshot_callback: SnapshotCallback | None = None,
    progress_callback: OptimizerProgressCallback | None = None,
    return_result: bool = False,
) -> np.ndarray | FTVResult:
    """Denoise a DEM using fractional-order Chambolle-Pock optimization.

    Args:
        dem: Interpolated DEM as a two-dimensional float32 array.
        artifact_mask: MSA mask of local defects.
        slope: Precomputed slope layer.
        alpha: Fractional GL derivative order.
        lambda_base: Base data-fidelity strength.
        max_iter: Maximum PDHG iterations.
        tol: Relative early-stop threshold.
        use_gpu: Prefer CuPy when available.
        max_curvature: Optional curvature layer for SARP.
        config: Full runtime configuration. Overrides scalar compatibility args.
        valid_mask: Source valid-data mask for SARP percentile calibration.
        snapshot_callback: Optional callback for intermediate comparisons.
        return_result: Return diagnostics instead of only the corrected DEM.

    Returns:
        Corrected DEM or an :class:`FTVResult`.
    """

    dem = np.asarray(dem, dtype=np.float32)
    slope = np.asarray(slope, dtype=np.float32)
    artifact_mask = np.asarray(artifact_mask, dtype=np.bool_)
    if dem.shape != slope.shape or dem.shape != artifact_mask.shape:
        raise ValueError("dem, slope, and artifact_mask shapes must match")
    if max_curvature is None:
        max_curvature = np.zeros_like(dem)
    if config is None:
        config = FTVConfig(
            alpha=alpha,
            lambda_base=lambda_base,
            max_iter=max_iter,
            tol=tol,
            backend="auto" if use_gpu else "cpu",
        )
    config.validate()
    if valid_mask is None:
        valid_mask = np.ones(dem.shape, dtype=np.bool_)

    fidelity_weight = generate_sarp_weights(
        slope,
        np.asarray(max_curvature, dtype=np.float32),
        artifact_mask,
        lambda_base=config.lambda_base,
        slope_gain=config.slope_gain,
        curvature_gain=config.curvature_gain,
        artifact_fidelity_scale=config.artifact_fidelity_scale,
        clip_percentile=config.sarp_clip_percentile,
        valid_mask=valid_mask,
    )
    result = _run_pdhg(dem, fidelity_weight, config, snapshot_callback, progress_callback)
    return result if return_result else result.corrected_dem
