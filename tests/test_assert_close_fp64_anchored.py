"""Host-only pins for `assert_close_fp64_anchored`."""

import pytest
import torch

from torch_mojo_backend.testing import assert_close_fp64_anchored

# Small magnitudes, so atol (1e-5) rather than rtol sets the default bar.
_REFERENCE = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64).reshape(8, 8)


def _rounded_with_noise(noise: float) -> torch.Tensor:
    """torch's-result stand-in: the exact answer rounded to fp32 plus one
    element perturbed by `noise` (fp32 summation-order error)."""
    out = _REFERENCE.float()
    out[3, 3] += noise
    return out


def test_identical_results_pass():
    torch_result = _rounded_with_noise(2e-5)
    assert_close_fp64_anchored(torch_result.clone(), torch_result, _REFERENCE)


def test_error_comparable_to_torchs_passes_beyond_the_default_bar():
    torch_result = _rounded_with_noise(2e-5)
    ours = _REFERENCE.float()
    ours[5, 1] -= 3e-5  # > atol 1e-5, < 2x torch's own 2e-5 miss
    with pytest.raises(AssertionError):
        torch.testing.assert_close(ours, torch_result)
    assert_close_fp64_anchored(ours, torch_result, _REFERENCE)


def test_error_far_beyond_torchs_fails():
    torch_result = _rounded_with_noise(2e-5)
    ours = _REFERENCE.float()
    ours[5, 1] -= 5e-4
    with pytest.raises(AssertionError, match="1 / 64 elements"):
        assert_close_fp64_anchored(ours, torch_result, _REFERENCE)


def test_where_torch_is_exact_the_default_bar_is_the_floor():
    torch_result = _REFERENCE.float()  # exact, up to fp32 rounding
    ours = torch_result.clone()
    ours[0, 0] += 5e-6  # inside atol 1e-5
    assert_close_fp64_anchored(ours, torch_result, _REFERENCE)
    ours[0, 0] += 5e-5  # outside it, and torch itself is (nearly) exact
    with pytest.raises(AssertionError):
        assert_close_fp64_anchored(ours, torch_result, _REFERENCE)


def test_dtype_and_shape_must_match():
    torch_result = _REFERENCE.float()
    with pytest.raises(AssertionError):
        assert_close_fp64_anchored(torch_result.double(), torch_result, _REFERENCE)
    with pytest.raises(AssertionError):
        assert_close_fp64_anchored(torch_result.flatten(), torch_result, _REFERENCE)


def test_a_nan_where_the_reference_is_finite_fails():
    torch_result = _rounded_with_noise(2e-5)
    ours = torch_result.clone()
    ours[2, 2] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        assert_close_fp64_anchored(ours, torch_result, _REFERENCE)


def test_non_finite_reference_positions_take_the_plain_bar():
    """A masked -inf (or a NaN) in the reference is compared as such, and the
    finite elements next to it are still anchored."""
    reference = _REFERENCE.clone()
    reference[0, 0] = float("-inf")
    reference[0, 1] = float("nan")
    torch_result = reference.float()
    torch_result[3, 3] += 2e-5
    ours = reference.float()
    ours[5, 1] -= 3e-5  # anchored: within 2x torch's 2e-5 miss
    assert_close_fp64_anchored(ours, torch_result, reference)
    ours[0, 0] = 1.0  # finite where torch says -inf: the plain bar rejects it
    with pytest.raises(AssertionError):
        assert_close_fp64_anchored(ours, torch_result, reference)


def test_a_non_finite_torch_result_at_a_finite_reference_takes_the_plain_bar():
    torch_result = _REFERENCE.float()
    torch_result[1, 1] = float(
        "inf"
    )  # torch overflowed where the exact answer is finite
    ours = torch_result.clone()
    assert_close_fp64_anchored(
        ours, torch_result, _REFERENCE
    )  # we overflow the same way
    ours[1, 1] = 1.0
    with pytest.raises(AssertionError):
        assert_close_fp64_anchored(ours, torch_result, _REFERENCE)


def test_an_empty_tensor_passes():
    empty = torch.empty(0, 3, dtype=torch.float64)
    assert_close_fp64_anchored(empty.float(), empty.float(), empty)
