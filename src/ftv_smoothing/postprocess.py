"""Iterative morphological and edge-aware postprocessing."""

from __future__ import annotations

import logging

import numpy as np
from numba import njit, prange
from scipy import ndimage as ndi
from skimage import measure, morphology

LOGGER = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True, cache=True)
def _anisotropic_step(
    image: np.ndarray,
    update_mask: np.ndarray,
    kappa: float,
    gamma: float,
) -> np.ndarray:
    rows, cols = image.shape
    output = image.copy()
    inv_kappa_sq = 1.0 / max(kappa * kappa, 1e-6)
    for row in prange(1, rows - 1):
        for col in range(1, cols - 1):
            if not update_mask[row, col]:
                continue
            center = image[row, col]
            north = image[row - 1, col] - center
            south = image[row + 1, col] - center
            west = image[row, col - 1] - center
            east = image[row, col + 1] - center
            flux = (
                np.exp(-(north * north) * inv_kappa_sq) * north
                + np.exp(-(south * south) * inv_kappa_sq) * south
                + np.exp(-(west * west) * inv_kappa_sq) * west
                + np.exp(-(east * east) * inv_kappa_sq) * east
            )
            output[row, col] = center + gamma * flux
    return output


def morphological_postprocess(
    dem_corrected: np.ndarray,
    original_dem: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    max_component_size: int = 24,
    diffusion_iterations: int = 2,
    kappa: float = 8.0,
    gamma: float = 0.12,
) -> np.ndarray:
    """Remove isolated residual artifacts and diffuse only non-edge zones."""

    corrected = np.asarray(dem_corrected, dtype=np.float32).copy()
    original = np.asarray(original_dem, dtype=np.float32)
    if corrected.shape != original.shape:
        raise ValueError("dem_corrected and original_dem shapes must match")
    if valid_mask is None:
        valid_mask = np.ones(corrected.shape, dtype=np.bool_)
    else:
        valid_mask = np.asarray(valid_mask, dtype=np.bool_)

    LOGGER.info("Running IMF morphological postprocessing")
    residual = np.abs(corrected - original)
    values = residual[valid_mask]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + 4.0 * 1.4826 * max(mad, 1e-6)
    suspects = (residual > threshold) & valid_mask

    opened = suspects.copy()
    for radius in range(1, 4):
        selem = morphology.disk(radius)
        opened = ndi.binary_opening(opened, structure=selem)
        opened = ndi.binary_closing(opened, structure=selem)

    labels = measure.label(suspects, connectivity=2)
    counts = np.bincount(labels.ravel())
    small = counts[labels] <= max_component_size
    isolated = suspects & (small | ~opened)
    if isolated.any():
        local_median = ndi.median_filter(corrected, size=3, mode="nearest")
        corrected[isolated] = local_median[isolated]
    LOGGER.info("IMF replaced %d isolated residual cells", int(isolated.sum()))

    gx = ndi.sobel(original, axis=1, mode="nearest")
    gy = ndi.sobel(original, axis=0, mode="nearest")
    edge_strength = np.hypot(gx, gy)
    edge_limit = float(np.percentile(edge_strength[valid_mask], 75.0))
    non_edge_mask = (edge_strength <= edge_limit) & valid_mask
    for _ in range(diffusion_iterations):
        corrected = _anisotropic_step(corrected, non_edge_mask, kappa, gamma)
    return corrected.astype(np.float32, copy=False)

