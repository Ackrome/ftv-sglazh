"""Command-line interface for FTV DEM smoothing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import FTVConfig
from .pipeline import orchestrate_file


def _crop(value: str) -> tuple[slice, slice]:
    try:
        row_text, col_text = value.split(",")
        row_start, row_stop = (int(part) for part in row_text.split(":"))
        col_start, col_stop = (int(part) for part in col_text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "crop must use row_start:row_stop,col_start:col_stop"
        ) from exc
    return slice(row_start, row_stop), slice(col_start, col_stop)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_nc", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--diagnostics-png", type=Path)
    parser.add_argument("--slope-comparison-png", type=Path)
    parser.add_argument("--forge3d-png", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--snapshot-every", type=int, default=25)
    parser.add_argument("--crop", type=_crop)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--k-size", type=int, default=12)
    parser.add_argument("--lambda-base", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--convolution-method",
        choices=("auto", "direct", "fft"),
        default="auto",
    )
    parser.add_argument("--msa-threshold", type=float, default=5.0)
    parser.add_argument("--interpolation-iterations", type=int, default=12)
    parser.add_argument("--visualization-dpi", type=int, default=600)
    parser.add_argument("--save-sarp", action="store_true")
    parser.add_argument("--skip-postprocess", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Keep --log-level DEBUG useful for FTV convergence rather than flooding
    # the console with Matplotlib font matching and Numba compiler internals.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    config = FTVConfig(
        alpha=args.alpha,
        k_size=args.k_size,
        lambda_base=args.lambda_base,
        max_iter=args.max_iter,
        tol=args.tol,
        backend=args.backend,
        convolution_method=args.convolution_method,
        msa_threshold=args.msa_threshold,
        interpolation_iterations=args.interpolation_iterations,
        snapshot_every=args.snapshot_every,
        visualization_dpi=args.visualization_dpi,
        save_sarp=args.save_sarp,
        postprocess=not args.skip_postprocess,
    )
    orchestrate_file(
        args.input_nc,
        args.output,
        args.png,
        diagnostics_png=args.diagnostics_png,
        slope_comparison_png=args.slope_comparison_png,
        forge3d_png=args.forge3d_png,
        snapshot_dir=args.snapshot_dir,
        config=config,
        crop=args.crop,
    )


if __name__ == "__main__":
    main()
