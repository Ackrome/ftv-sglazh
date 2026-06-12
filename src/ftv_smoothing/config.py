"""Configuration objects for the FTV pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class FTVConfig:
    """Runtime parameters for FTV smoothing.

    The default GL memory window is intentionally inside the 10-15 range from
    the specification. Direct separable convolution is faster than FFT for
    windows this short; FFT remains available for experimentation.
    """

    alpha: float = 1.5
    k_size: int = 12
    lambda_base: float = 1.0
    max_iter: int = 200
    tol: float = 1e-5
    theta: float = 1.0
    tau: float | None = None
    sigma: float | None = None
    backend: str = "auto"
    convolution_method: str = "auto"
    interpolation_iterations: int = 12
    msa_threshold: float = 5.0
    msa_window: int = 7
    msa_robust_sigma: float = 5.0
    msa_dilation_iterations: int = 2
    slope_gain: float = 3.0
    curvature_gain: float = 2.0
    artifact_fidelity_scale: float = 0.25
    sarp_clip_percentile: float = 95.0
    log_every: int = 10
    memory_cleanup_every: int = 25
    snapshot_every: int = 25
    snapshot_dpi: int = 180
    visualization_dpi: int = 600
    postprocess: bool = True
    postprocess_component_size: int = 24
    postprocess_diffusion_iterations: int = 2
    postprocess_kappa: float = 8.0
    postprocess_gamma: float = 0.12
    save_sarp: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` for unsupported parameter combinations."""

        if not 1.0 <= self.alpha <= 2.0:
            raise ValueError("alpha must be in [1.0, 2.0]")
        if not 2 <= self.k_size <= 128:
            raise ValueError("k_size must be in [2, 128]")
        if self.lambda_base <= 0:
            raise ValueError("lambda_base must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.backend not in {"auto", "cpu", "gpu"}:
            raise ValueError("backend must be auto, cpu, or gpu")
        if self.convolution_method not in {"auto", "direct", "fft"}:
            raise ValueError("convolution_method must be auto, direct, or fft")
        if self.msa_window < 3 or self.msa_window % 2 == 0:
            raise ValueError("msa_window must be an odd integer >= 3")
        if not 0 < self.artifact_fidelity_scale <= 1:
            raise ValueError("artifact_fidelity_scale must be in (0, 1]")
        if not 0 < self.postprocess_gamma <= 0.25:
            raise ValueError("postprocess_gamma must be in (0, 0.25]")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        return asdict(self)

