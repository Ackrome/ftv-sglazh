"""NetCDF ingestion and NaN-safe interpolation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage as ndi

LOGGER = logging.getLogger(__name__)

DERIVATIVE_NAMES = (
    "slope",
    "hillshade",
    "aspect",
    "curvature",
    "planform_curvature",
    "profile_curvature",
    "max_curvature",
    "topographic_position_index",
    "terrain_ruggedness_index",
    "roughness",
    "rugosity",
)


def _to_float32(array: np.ndarray) -> np.ndarray:
    """Convert masked or regular arrays to a writable ``float32`` array."""

    return np.asarray(np.ma.filled(array, np.nan), dtype=np.float32).copy()


def interpolate_nan_diffusion(
    array: np.ndarray,
    iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill NaN values using nearest initialization followed by diffusion.

    Nearest-neighbor values initialize the large exterior masked region. A
    four-neighbor diffusion pass then makes the working field continuous for
    GL convolutions. The original invalid mask is returned for exact
    restoration after optimization.

    Args:
        array: Two-dimensional DEM array.
        iterations: Number of diffusion passes over invalid pixels.

    Returns:
        Filled DEM, original invalid mask, and nearest-valid index tensor.
    """

    filled = _to_float32(array)
    if filled.ndim != 2:
        raise ValueError("DEM array must be two-dimensional")

    invalid_mask = ~np.isfinite(filled)
    if not invalid_mask.any():
        empty_indices = np.indices(filled.shape, dtype=np.int32)
        return filled, invalid_mask, empty_indices
    if invalid_mask.all():
        raise ValueError("DEM array contains no finite values")

    LOGGER.info(
        "Interpolating %d NaN DEM cells (%.2f%%)",
        int(invalid_mask.sum()),
        float(invalid_mask.mean() * 100),
    )
    nearest_indices = ndi.distance_transform_edt(
        invalid_mask,
        return_distances=False,
        return_indices=True,
    )
    filled[invalid_mask] = filled[tuple(nearest_indices[:, invalid_mask])]

    if iterations > 0:
        kernel = np.array(
            [[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]],
            dtype=np.float32,
        )
        workspace = np.empty_like(filled)
        for _ in range(iterations):
            ndi.correlate(filled, kernel, output=workspace, mode="nearest")
            filled[invalid_mask] = workspace[invalid_mask]

    return filled, invalid_mask, nearest_indices


def _fill_derivative(
    array: np.ndarray,
    dem_invalid_mask: np.ndarray,
    nearest_indices: np.ndarray,
) -> np.ndarray:
    """Fill derivative gaps without running a second expensive distance map."""

    result = _to_float32(array)
    result[dem_invalid_mask] = result[tuple(nearest_indices[:, dem_invalid_mask])]
    remaining = ~np.isfinite(result)
    if remaining.any():
        finite_values = result[np.isfinite(result)]
        fallback = np.float32(np.median(finite_values)) if finite_values.size else np.float32(0)
        result[remaining] = fallback
    return result


def load_and_prepare_ds(
    ds: xr.Dataset,
    interpolation_iterations: int = 12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Extract DEM data, fill gaps, and prepare morphometric derivatives.

    Args:
        ds: Dataset containing ``reprojected_dem`` and derivative layers.
        interpolation_iterations: Number of NaN diffusion passes.

    Returns:
        Interpolated DEM, original invalid mask, and float32 derivatives.
    """

    if "reprojected_dem" not in ds:
        raise KeyError("Dataset must contain 'reprojected_dem'")

    LOGGER.info("Loading and preparing NetCDF layers")
    dem, invalid_mask, nearest_indices = interpolate_nan_diffusion(
        ds["reprojected_dem"].values,
        iterations=interpolation_iterations,
    )
    derivatives: dict[str, np.ndarray] = {}
    for name in DERIVATIVE_NAMES:
        if name in ds:
            derivatives[name] = _fill_derivative(
                ds[name].values,
                invalid_mask,
                nearest_indices,
            )
    if "slope" not in derivatives:
        raise KeyError("Dataset must contain 'slope'")
    if "max_curvature" not in derivatives:
        LOGGER.warning("max_curvature missing; using zeros for SARP")
        derivatives["max_curvature"] = np.zeros_like(dem)
    LOGGER.info("Prepared DEM and %d derivative layers", len(derivatives))
    return dem, invalid_mask, derivatives


def restore_nan_topology(array: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    """Restore source NaNs exactly at their original locations."""

    restored = np.asarray(array, dtype=np.float32).copy()
    restored[invalid_mask] = np.nan
    return restored


def open_dataset(path: str | Path) -> xr.Dataset:
    """Open a NetCDF dataset without converting float32 layers to float64."""

    return xr.open_dataset(Path(path), engine="h5netcdf", mask_and_scale=True)
