"""Export a full-area DEM model for the browser WebGL terrain viewer."""

from __future__ import annotations

import argparse
import importlib.resources
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .forge3d_visualize import parse_crop
from .io import open_dataset


@dataclass(frozen=True, slots=True)
class WebGLTerrainConfig:
    """Controls for a browser-native full-area terrain mesh."""

    max_side: int = 512
    vertical_exaggeration: float = 2.4
    resolution_m: float = 30.0

    def validate(self) -> None:
        """Reject model settings that cannot produce a useful mesh."""

        if self.max_side < 64:
            raise ValueError("WebGL terrain max_side must be at least 64 pixels")
        if self.vertical_exaggeration <= 0:
            raise ValueError("WebGL terrain vertical_exaggeration must be positive")
        if self.resolution_m <= 0:
            raise ValueError("WebGL terrain resolution_m must be positive")


@dataclass(frozen=True, slots=True)
class WebGLTerrainArtifacts:
    """Files emitted for the browser-native full-area terrain model."""

    output_dir: Path
    model_json: Path
    before_f32: Path
    after_f32: Path
    mask_u8: Path
    viewer_html: Path
    metadata: dict[str, Any]


def _sample_indices(size: int, target_size: int) -> np.ndarray:
    return np.linspace(0, size - 1, target_size).round().astype(np.intp)


def _downsample_full_extent(
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_side: int,
    segment_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original = np.asarray(before, dtype=np.float32)
    corrected = np.asarray(after, dtype=np.float32)
    if original.ndim != 2 or corrected.ndim != 2:
        raise ValueError("WebGL terrain inputs must be two-dimensional")
    if original.shape != corrected.shape:
        raise ValueError("WebGL terrain before and after shapes must match")
    valid = np.isfinite(original) & np.isfinite(corrected)
    if not valid.any():
        raise ValueError("WebGL terrain export requires finite paired DEM cells")

    scale = min(1.0, max_side / float(max(original.shape)))
    target_rows = max(2, int(round(original.shape[0] * scale)))
    target_cols = max(2, int(round(original.shape[1] * scale)))
    rows = _sample_indices(original.shape[0], target_rows)
    cols = _sample_indices(original.shape[1], target_cols)
    sampled_before = original[np.ix_(rows, cols)]
    sampled_after = corrected[np.ix_(rows, cols)]
    sampled_valid = valid[np.ix_(rows, cols)]
    if segment_mask is None:
        sampled_segments = np.zeros(sampled_valid.shape, dtype=np.uint8)
    else:
        segments = np.asarray(segment_mask, dtype=np.bool_)
        if segments.shape != original.shape:
            raise ValueError("segment_mask shape must match terrain inputs")
        sampled_segments = (segments[np.ix_(rows, cols)] & sampled_valid).astype(np.uint8) * 255

    minimum = float(min(np.nanmin(original), np.nanmin(corrected)))
    sampled_before = np.where(sampled_valid, sampled_before, minimum).astype("<f4")
    sampled_after = np.where(sampled_valid, sampled_after, minimum).astype("<f4")
    return sampled_before, sampled_after, sampled_valid.astype(np.uint8) * 255, sampled_segments


def _copy_viewer_assets(output_dir: Path) -> Path:
    resource_dir = importlib.resources.files("ftv_smoothing").joinpath("viewer_assets")
    for filename in ("index.html", "styles.css", "app.js"):
        source = resource_dir.joinpath(filename)
        with importlib.resources.as_file(source) as source_path:
            shutil.copyfile(source_path, output_dir / filename)
    return output_dir / "index.html"


def export_webgl_model(
    before: np.ndarray,
    after: np.ndarray,
    output_dir: str | Path,
    *,
    config: WebGLTerrainConfig | None = None,
    source_label: str = "DEM",
    scene_label: str = "Full study area",
    segment_mask: np.ndarray | None = None,
) -> WebGLTerrainArtifacts:
    """Export a real WebGL heightfield model covering the complete input extent."""

    if config is None:
        config = WebGLTerrainConfig()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original = np.asarray(before, dtype=np.float32)
    corrected = np.asarray(after, dtype=np.float32)
    valid = np.isfinite(original) & np.isfinite(corrected)
    sampled_before, sampled_after, sampled_mask, sampled_segments = _downsample_full_extent(
        original,
        corrected,
        max_side=config.max_side,
        segment_mask=segment_mask,
    )

    before_path = output / "terrain_before.f32"
    after_path = output / "terrain_after.f32"
    mask_path = output / "terrain_mask.u8"
    segments_path = output / "terrain_segments.u8"
    sampled_before.tofile(before_path)
    sampled_after.tofile(after_path)
    sampled_mask.tofile(mask_path)
    sampled_segments.tofile(segments_path)

    source_rows, source_cols = original.shape
    grid_rows, grid_cols = sampled_before.shape
    valid_before = original[valid]
    valid_after = corrected[valid]
    minimum = float(min(np.min(valid_before), np.min(valid_after)))
    maximum = float(max(np.max(valid_before), np.max(valid_after)))
    span_x_m = float(max(1, source_cols - 1) * config.resolution_m)
    span_z_m = float(max(1, source_rows - 1) * config.resolution_m)
    metadata: dict[str, Any] = {
        "viewer_type": "webgl-terrain-model",
        "schema_version": 3,
        "title": "FTV Terrain Viewer",
        "asset_revision": datetime.now(timezone.utc).isoformat(),
        "source": source_label,
        "scene": scene_label,
        "whole_study_area": True,
        "axis_convention": "X=east-west, Y=elevation, Z=north-south",
        "base_surface": "horizontal XZ plane",
        "files": {
            "before": before_path.name,
            "after": after_path.name,
            "mask": mask_path.name,
            "segments": segments_path.name,
        },
        "source_shape": [source_rows, source_cols],
        "grid_shape": [grid_rows, grid_cols],
        "source_valid_fraction": float(valid.mean()),
        "grid_valid_fraction": float(np.count_nonzero(sampled_mask) / sampled_mask.size),
        "grid_segment_fraction": float(np.count_nonzero(sampled_segments) / sampled_segments.size),
        "resolution_m": config.resolution_m,
        "span_x_m": span_x_m,
        "span_z_m": span_z_m,
        "terrain_span_m": max(span_x_m, span_z_m),
        "elevation_min_m": minimum,
        "elevation_max_m": maximum,
        "vertical_exaggeration_default": config.vertical_exaggeration,
    }
    model_json = output / "terrain-model.json"
    model_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    viewer_html = _copy_viewer_assets(output)
    return WebGLTerrainArtifacts(
        output_dir=output,
        model_json=model_json,
        before_f32=before_path,
        after_f32=after_path,
        mask_u8=mask_path,
        viewer_html=viewer_html,
        metadata=metadata,
    )


def export_dataset_webgl_model(
    input_nc: str | Path,
    output_dir: str | Path,
    *,
    before_var: str = "reprojected_dem",
    after_var: str = "reprojected_dem_ftv",
    crop: tuple[slice, slice] | None = None,
    config: WebGLTerrainConfig | None = None,
) -> WebGLTerrainArtifacts:
    """Export an existing FTV NetCDF artifact as a browser-native terrain model."""

    path = Path(input_nc)
    with open_dataset(path) as ds:
        if before_var not in ds or after_var not in ds:
            raise KeyError(f"NetCDF must contain {before_var!r} and {after_var!r}")
        before = ds[before_var]
        after = ds[after_var]
        scene_label = "Full study area"
        if crop is not None:
            row_slice, col_slice = crop
            before = before.isel(latitude=row_slice, longitude=col_slice)
            after = after.isel(latitude=row_slice, longitude=col_slice)
            scene_label = "Selected study area crop"
        if config is None:
            config = WebGLTerrainConfig(resolution_m=float(ds.attrs.get("resolution_m", 30.0)))
        return export_webgl_model(
            before.values,
            after.values,
            output_dir,
            config=config,
            source_label=path.name,
            scene_label=scene_label,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create the browser-native terrain export CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_nc", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--before-var", default="reprojected_dem")
    parser.add_argument("--after-var", default="reprojected_dem_ftv")
    parser.add_argument("--crop", type=parse_crop)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--resolution-m", type=float)
    parser.add_argument("--vertical-exaggeration", type=float, default=2.4)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Export a full-area browser-native terrain model."""

    args = build_parser().parse_args(argv)
    if args.resolution_m is None:
        with open_dataset(args.input_nc) as ds:
            resolution_m = float(ds.attrs.get("resolution_m", 30.0))
    else:
        resolution_m = args.resolution_m
    export_dataset_webgl_model(
        args.input_nc,
        args.output_dir,
        before_var=args.before_var,
        after_var=args.after_var,
        crop=args.crop,
        config=WebGLTerrainConfig(
            max_side=args.max_side,
            resolution_m=resolution_m,
            vertical_exaggeration=args.vertical_exaggeration,
        ),
    )


if __name__ == "__main__":
    main()
