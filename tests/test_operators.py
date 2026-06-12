import numpy as np
from numpy.testing import assert_allclose

from ftv_smoothing.operators import FractionalOperator, compute_gl_coefficients


def test_gl_coefficients_match_analytic_sequence() -> None:
    actual = compute_gl_coefficients(1.5, 5)
    expected = np.array([1.0, -1.5, 0.375, 0.0625, 0.0234375], dtype=np.float32)
    assert_allclose(actual, expected, atol=1e-7)


def test_alpha_one_is_standard_backward_difference() -> None:
    array = np.arange(20, dtype=np.float32).reshape(4, 5)
    operator = FractionalOperator(alpha=1.0, k_size=5, backend="cpu")
    gx, gy = operator.gradient(array)
    assert_allclose(gx, array - np.roll(array, 1, axis=1))
    assert_allclose(gy, array - np.roll(array, 1, axis=0))


def test_short_memory_correction_preserves_constants() -> None:
    array = np.full((8, 9), 123.0, dtype=np.float32)
    operator = FractionalOperator(alpha=1.5, k_size=12, backend="cpu")
    gx, gy = operator.gradient(array)
    assert_allclose(gx, 0, atol=1e-5)
    assert_allclose(gy, 0, atol=1e-5)


def test_divergence_is_negative_adjoint() -> None:
    rng = np.random.default_rng(4)
    array = rng.normal(size=(17, 19)).astype(np.float32)
    px = rng.normal(size=array.shape).astype(np.float32)
    py = rng.normal(size=array.shape).astype(np.float32)
    operator = FractionalOperator(alpha=1.45, k_size=11, backend="cpu")
    gx, gy = operator.gradient(array)
    divergence = operator.divergence(px, py)
    left = np.sum(gx * px + gy * py, dtype=np.float64)
    right = np.sum(-array * divergence, dtype=np.float64)
    assert_allclose(left, right, atol=2e-5)


def test_fft_matches_direct_operator() -> None:
    rng = np.random.default_rng(2)
    array = rng.normal(size=(16, 18)).astype(np.float32)
    direct = FractionalOperator(alpha=1.5, k_size=12, backend="cpu", method="direct")
    fft = FractionalOperator(alpha=1.5, k_size=12, backend="cpu", method="fft")
    expected = direct.gradient(array)
    actual = fft.gradient(array)
    assert_allclose(actual[0], expected[0], atol=2e-5)
    assert_allclose(actual[1], expected[1], atol=2e-5)

