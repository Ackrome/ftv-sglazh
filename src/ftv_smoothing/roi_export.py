"""Export changed-area ROI samples for the web application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from .io import open_dataset
from .webgl_export import WebGLTerrainConfig, export_webgl_model


@dataclass(frozen=True, slots=True)
class ROISample:
    """One changed-area sample selected from the corrected DEM."""

    id: str
    rank: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    center_row: int
    center_col: int
    mean_abs_change_m: float
    max_abs_change_m: float
    valid_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rank": self.rank,
            "row_start": self.row_start,
            "row_stop": self.row_stop,
            "col_start": self.col_start,
            "col_stop": self.col_stop,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "mean_abs_change_m": self.mean_abs_change_m,
            "max_abs_change_m": self.max_abs_change_m,
            "valid_fraction": self.valid_fraction,
        }


def _window_sums(values: np.ndarray, window_size: int) -> np.ndarray:
    padded = np.pad(values, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    size = int(window_size)
    return (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )


def select_changed_rois(
    before: np.ndarray,
    after: np.ndarray,
    *,
    count: int,
    window_size: int = 96,
    min_valid_fraction: float = 0.55,
) -> list[ROISample]:
    """Select non-overlapping windows with the largest mean absolute change."""

    if count <= 0:
        return []
    original = np.asarray(before, dtype=np.float32)
    corrected = np.asarray(after, dtype=np.float32)
    if original.ndim != 2 or original.shape != corrected.shape:
        raise ValueError("ROI selection requires paired 2D arrays with the same shape")
    rows, cols = original.shape
    size = max(2, min(int(window_size), rows, cols))
    valid = np.isfinite(original) & np.isfinite(corrected)
    if not valid.any():
        return []

    residual = np.where(valid, np.abs(corrected - original), 0.0).astype(np.float64)
    valid_f = valid.astype(np.float64)
    change_sum = _window_sums(residual, size)
    valid_sum = _window_sums(valid_f, size)
    valid_fraction = valid_sum / float(size * size)
    score = np.divide(
        change_sum,
        valid_sum,
        out=np.full_like(change_sum, -np.inf),
        where=valid_sum > 0,
    )
    score[valid_fraction < min_valid_fraction] = -np.inf

    samples: list[ROISample] = []
    for rank in range(1, count + 1):
        flat_index = int(np.argmax(score))
        best = float(score.flat[flat_index])
        if not np.isfinite(best) or best <= 0:
            break
        row_start, col_start = (int(value) for value in np.unravel_index(flat_index, score.shape))
        row_stop = row_start + size
        col_stop = col_start + size
        patch_residual = residual[row_start:row_stop, col_start:col_stop]
        patch_valid = valid[row_start:row_stop, col_start:col_stop]
        sample_id = f"roi_{rank:02d}"
        samples.append(
            ROISample(
                id=sample_id,
                rank=rank,
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
                center_row=row_start + size // 2,
                center_col=col_start + size // 2,
                mean_abs_change_m=best,
                max_abs_change_m=float(np.max(patch_residual[patch_valid])),
                valid_fraction=float(patch_valid.mean()),
            )
        )

        block_row_start = max(0, row_start - size + 1)
        block_row_stop = min(score.shape[0], row_stop)
        block_col_start = max(0, col_start - size + 1)
        block_col_stop = min(score.shape[1], col_stop)
        score[block_row_start:block_row_stop, block_col_start:block_col_stop] = -np.inf

    return samples


def _save_roi_png(
    data: np.ndarray,
    output: Path,
    *,
    vmin: float,
    vmax: float,
    title: str,
    segment_mask: np.ndarray | None = None,
) -> None:
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=130)
    ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        interpolation="nearest",
    )
    if segment_mask is not None and np.any(segment_mask):
        mask = np.asarray(segment_mask, dtype=np.float32)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 1] = 0.16
        overlay[..., 2] = 0.12
        overlay[..., 3] = np.where(mask > 0, 0.22, 0.0)
        ax.imshow(overlay, origin="upper", interpolation="nearest")
        ax.contour(mask, levels=[0.5], colors=["#ffdd55"], linewidths=1.15, origin="upper")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _shared_limits(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]])
    if values.size == 0:
        return 0.0, 1.0
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def _change_segments(
    before_patch: np.ndarray,
    after_patch: np.ndarray,
    *,
    top_fraction: float = 0.15,
    min_pixels: int = 6,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    valid = np.isfinite(before_patch) & np.isfinite(after_patch)
    residual = np.where(valid, np.abs(after_patch - before_patch), np.nan)
    values = residual[np.isfinite(residual)]
    if values.size == 0 or float(np.max(values)) <= 0:
        return np.zeros(before_patch.shape, dtype=np.bool_), []

    threshold = float(np.nanpercentile(values, max(0.0, min(100.0, (1.0 - top_fraction) * 100.0))))
    mask = np.isfinite(residual) & (residual >= threshold) & (residual > 0)
    labeled, count = ndi.label(mask)
    segments: list[dict[str, Any]] = []
    clean = np.zeros(mask.shape, dtype=np.bool_)
    for index, region in enumerate(ndi.find_objects(labeled), start=1):
        if region is None:
            continue
        component = labeled[region] == index
        pixels = int(component.sum())
        if pixels < min_pixels:
            continue
        row_slice, col_slice = region
        clean[region] |= component
        component_values = residual[region][component]
        segments.append(
            {
                "row_start": int(row_slice.start),
                "row_stop": int(row_slice.stop),
                "col_start": int(col_slice.start),
                "col_stop": int(col_slice.stop),
                "pixels": pixels,
                "mean_abs_change_m": float(np.mean(component_values)),
                "max_abs_change_m": float(np.max(component_values)),
            }
        )

    segments.sort(key=lambda item: item["mean_abs_change_m"], reverse=True)
    return clean, segments


def export_changed_roi_samples(
    input_nc: str | Path,
    output_dir: str | Path,
    *,
    count: int,
    window_size: int = 96,
    before_var: str = "reprojected_dem",
    after_var: str = "reprojected_dem_ftv",
    config: WebGLTerrainConfig | None = None,
) -> dict[str, Any]:
    """Export 2D and 3D examples from the most changed DEM areas."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if count <= 0:
        metadata = {
            "schema_version": 1,
            "sample_count": 0,
            "window_size": int(window_size),
            "samples": [],
        }
        (output / "index.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    with open_dataset(input_nc) as ds:
        before = np.asarray(ds[before_var].values, dtype=np.float32)
        after = np.asarray(ds[after_var].values, dtype=np.float32)
        resolution_m = float(ds.attrs.get("resolution_m", 30.0))

    if config is None:
        config = WebGLTerrainConfig(resolution_m=resolution_m)
    else:
        config = WebGLTerrainConfig(
            max_side=config.max_side,
            vertical_exaggeration=config.vertical_exaggeration,
            resolution_m=resolution_m,
        )
    samples = select_changed_rois(
        before,
        after,
        count=count,
        window_size=window_size,
    )

    sample_payloads: list[dict[str, Any]] = []
    for sample in samples:
        sample_dir = output / sample.id
        sample_dir.mkdir(parents=True, exist_ok=True)
        rows = slice(sample.row_start, sample.row_stop)
        cols = slice(sample.col_start, sample.col_stop)
        before_patch = before[rows, cols]
        after_patch = after[rows, cols]
        vmin, vmax = _shared_limits(before_patch, after_patch)
        segment_mask, segments = _change_segments(before_patch, after_patch)
        before_png = sample_dir / "before.png"
        after_png = sample_dir / "after.png"
        _save_roi_png(
            before_patch,
            before_png,
            vmin=vmin,
            vmax=vmax,
            title="Before FTV",
            segment_mask=segment_mask,
        )
        _save_roi_png(
            after_patch,
            after_png,
            vmin=vmin,
            vmax=vmax,
            title="After FTV",
            segment_mask=segment_mask,
        )
        webgl = export_webgl_model(
            before_patch,
            after_patch,
            sample_dir / "webgl",
            config=config,
            source_label=Path(input_nc).name,
            scene_label=f"{sample.id}: rows {sample.row_start}:{sample.row_stop}, "
            f"cols {sample.col_start}:{sample.col_stop}",
            segment_mask=segment_mask,
        )
        payload = sample.to_dict()
        payload["change_segments"] = segments
        payload["change_segment_count"] = len(segments)
        payload["files"] = {
            "before_png": f"{sample.id}/before.png",
            "after_png": f"{sample.id}/after.png",
            "webgl_model": f"{sample.id}/webgl/{webgl.model_json.name}",
        }
        payload["webgl"] = webgl.metadata
        sample_payloads.append(payload)

    metadata = {
        "schema_version": 1,
        "sample_count": len(sample_payloads),
        "requested_count": int(count),
        "window_size": int(window_size),
        "ranking": "mean absolute elevation correction over non-overlapping windows",
        "samples": sample_payloads,
    }
    (output / "index.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata
