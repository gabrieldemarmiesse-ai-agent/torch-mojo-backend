"""Native backend: the compare group (eq/ne/lt/le/gt/ge, isin, where,
masked_fill(_), searchsorted, bucketize). Public torch API on the mojo
device only -- see docs/native_backend.md and AGENT_BRIEF.md.
"""

import pytest
import torch

from torch_mojo_backend import aten_functions, register_mojo_devices
from torch_mojo_backend.testing import CallChecker


@pytest.fixture(autouse=True)
def _ensure_registered():
    """`mojo_device` (unlike `mojo_gpu`) doesn't register the device itself;
    do it here so this file runs standalone in any collection order."""
    register_mojo_devices()


# ---------------------------------------------------------------------------
# eq / ne / lt / le / gt / ge
# ---------------------------------------------------------------------------

_COMPARE_OPS = {
    "eq": (torch.eq, aten_functions.aten_eq),
    "ne": (torch.ne, aten_functions.aten_ne),
    "lt": (torch.lt, aten_functions.aten_lt),
    "le": (torch.le, aten_functions.aten_le),
    "gt": (torch.gt, aten_functions.aten_gt),
    "ge": (torch.ge, aten_functions.aten_ge),
}


@pytest.mark.parametrize("name", list(_COMPARE_OPS))
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
def test_compare_tensor(
    mojo_device: str, call_checker: CallChecker, name: str, dtype: torch.dtype
):
    fn, twin = _COMPARE_OPS[name]
    call_checker.register(twin)
    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    b = torch.tensor([[1, 0, 3], [7, 5, 2]], dtype=dtype)
    expected = fn(a, b)
    got = fn(a.to(mojo_device), b.to(mojo_device))
    assert got.dtype == torch.bool
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("name", list(_COMPARE_OPS))
def test_compare_scalar(mojo_device: str, call_checker: CallChecker, name: str):
    fn, twin = _COMPARE_OPS[name]
    call_checker.register(twin)
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    expected = fn(a, 3.0)
    got = fn(a.to(mojo_device), 3.0)
    torch.testing.assert_close(got.cpu(), expected)


def test_compare_broadcast(mojo_device: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_lt)
    a = torch.randn(3, 1, 4)
    b = torch.randn(1, 5, 4)
    expected = torch.lt(a, b)
    got = torch.lt(a.to(mojo_device), b.to(mojo_device))
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize(
    ("a_dtype", "b_dtype"),
    [
        (torch.bool, torch.int32),
        (torch.int32, torch.int64),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.bfloat16),
    ],
)
def test_compare_dtype_promotion(
    mojo_device: str,
    call_checker: CallChecker,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
):
    call_checker.register(aten_functions.aten_eq)
    a = torch.tensor([0, 1, 1, 0], dtype=a_dtype)
    b = torch.tensor([0, 0, 1, 1], dtype=b_dtype)
    expected = torch.eq(a, b)
    got = torch.eq(a.to(mojo_device), b.to(mojo_device))
    torch.testing.assert_close(got.cpu(), expected)


def test_compare_out_tensor_and_scalar(mojo_gpu: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_eq)
    a = torch.tensor([1.0, 2.0, 3.0]).to(mojo_gpu)
    b = torch.tensor([1.0, 0.0, 3.0]).to(mojo_gpu)
    out = torch.empty(3, dtype=torch.bool, device=mojo_gpu)
    assert torch.eq(a, b, out=out) is out
    torch.testing.assert_close(out.cpu(), torch.tensor([True, False, True]))

    out2 = torch.empty(3, dtype=torch.bool, device=mojo_gpu)
    assert torch.eq(a, 2.0, out=out2) is out2
    torch.testing.assert_close(out2.cpu(), torch.tensor([False, True, False]))

    # A mis-shaped out= tensor forces the compute-then-copy fallback path.
    out3 = torch.empty(0, dtype=torch.bool, device=mojo_gpu)
    assert torch.eq(a, b, out=out3) is out3
    torch.testing.assert_close(out3.cpu(), torch.tensor([True, False, True]))


def test_compare_device_mismatch_raises(mojo_gpu: str):
    a = torch.tensor([1.0]).to(mojo_gpu)
    with pytest.raises(RuntimeError):
        torch.eq(a, torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# isin.Tensor_Tensor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_isin_basic(mojo_device: str, call_checker: CallChecker, dtype: torch.dtype):
    call_checker.register(aten_functions.aten_isin)
    elements = torch.tensor([1, 2, 3, 4, 5], dtype=dtype)
    test_elements = torch.tensor([2, 4], dtype=dtype)
    expected = torch.isin(elements, test_elements)
    got = torch.isin(elements.to(mojo_device), test_elements.to(mojo_device))
    torch.testing.assert_close(got.cpu(), expected)


def test_isin_invert(mojo_device: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_isin)
    elements = torch.tensor([1, 2, 3, 4, 5])
    test_elements = torch.tensor([2, 4])
    expected = torch.isin(elements, test_elements, invert=True)
    got = torch.isin(
        elements.to(mojo_device), test_elements.to(mojo_device), invert=True
    )
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("invert", [False, True])
def test_isin_empty_test_elements(
    mojo_device: str, call_checker: CallChecker, invert: bool
):
    call_checker.register(aten_functions.aten_isin)
    elements = torch.tensor([1, 2, 3])
    test_elements = torch.empty(0, dtype=torch.int64)
    expected = torch.isin(elements, test_elements, invert=invert)
    got = torch.isin(
        elements.to(mojo_device), test_elements.to(mojo_device), invert=invert
    )
    torch.testing.assert_close(got.cpu(), expected)


def test_isin_out(mojo_gpu: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_isin)
    elements = torch.tensor([1, 2, 3, 4]).to(mojo_gpu)
    test_elements = torch.tensor([2, 4]).to(mojo_gpu)
    out = torch.empty(4, dtype=torch.bool, device=mojo_gpu)
    assert torch.isin(elements, test_elements, out=out) is out
    torch.testing.assert_close(out.cpu(), torch.tensor([False, True, False, True]))


def test_isin_unsupported_dtype_raises(mojo_gpu: str):
    elements = torch.tensor([1.0, 2.0]).to(mojo_gpu)
    test_elements = torch.tensor([1.0]).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.isin(elements, test_elements)


# ---------------------------------------------------------------------------
# where.self
# ---------------------------------------------------------------------------


def test_where_self(mojo_device: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_where)
    cond = torch.tensor([[True, False], [False, True]])
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    expected = torch.where(cond, a, b)
    got = torch.where(cond.to(mojo_device), a.to(mojo_device), b.to(mojo_device))
    torch.testing.assert_close(got.cpu(), expected)


def test_where_self_broadcast(mojo_device: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_where)
    cond = torch.tensor([True, False, True])
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.tensor(0.0)
    expected = torch.where(cond, a, b)
    got = torch.where(cond.to(mojo_device), a.to(mojo_device), b.to(mojo_device))
    torch.testing.assert_close(got.cpu(), expected)


def test_where_self_dtype_promotion(mojo_gpu: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_where)
    cond = torch.tensor([True, False, True, False]).to(mojo_gpu)
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(mojo_gpu)
    b = torch.tensor([10, 20, 30, 40], dtype=torch.int64).to(mojo_gpu)
    got = torch.where(cond, a, b)
    assert got.dtype == torch.int64
    torch.testing.assert_close(
        got.cpu(), torch.tensor([1, 20, 3, 40], dtype=torch.int64)
    )


# ---------------------------------------------------------------------------
# masked_fill(.Scalar/.Tensor) and masked_fill_
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_masked_fill_scalar_fast_path(
    mojo_device: str, call_checker: CallChecker, dtype: torch.dtype
):
    """self is a float dtype: routes through MaskedFillScalar."""
    call_checker.register(aten_functions.aten_masked_fill)
    a = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    mask = torch.tensor([True, False, True, False])
    expected = a.masked_fill(mask, -1.0)
    got = a.to(mojo_device).masked_fill(mask.to(mojo_device), -1.0)
    torch.testing.assert_close(got.cpu().float(), expected.float())


def test_masked_fill_scalar_int_dtype(mojo_device: str, call_checker: CallChecker):
    """self is int64: falls back to the generic WhereSelect route."""
    call_checker.register(aten_functions.aten_masked_fill)
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    mask = torch.tensor([True, False, True, False])
    expected = a.masked_fill(mask, -7)
    got = a.to(mojo_device).masked_fill(mask.to(mojo_device), -7)
    torch.testing.assert_close(got.cpu(), expected)


def test_masked_fill_scalar_broadcast_mask(mojo_device: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_masked_fill)
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    mask = torch.tensor([True, False, True])
    expected = a.masked_fill(mask, 9.0)
    got = a.to(mojo_device).masked_fill(mask.to(mojo_device), 9.0)
    torch.testing.assert_close(got.cpu(), expected)


def test_masked_fill_tensor_value(mojo_device: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_masked_fill)
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, False])
    value = torch.tensor(-3.0)
    expected = a.masked_fill(mask, value)
    got = a.to(mojo_device).masked_fill(mask.to(mojo_device), value.to(mojo_device))
    torch.testing.assert_close(got.cpu(), expected)


def test_masked_fill_tensor_value_wrong_rank_raises(mojo_gpu: str):
    a = torch.tensor([1.0, 2.0]).to(mojo_gpu)
    mask = torch.tensor([True, False]).to(mojo_gpu)
    value = torch.tensor([1.0, 2.0]).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="0-dimensional"):
        a.masked_fill(mask, value)


def test_masked_fill_out(mojo_gpu: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_masked_fill)
    a = torch.tensor([1.0, 2.0, 3.0]).to(mojo_gpu)
    mask = torch.tensor([True, False, True]).to(mojo_gpu)
    out = torch.empty(3, device=mojo_gpu)
    assert torch.ops.aten.masked_fill.Scalar_out(a, mask, 5.0, out=out) is out
    torch.testing.assert_close(out.cpu(), torch.tensor([5.0, 2.0, 5.0]))


def test_masked_fill__inplace_scalar_contiguous(mojo_gpu: str):
    a = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(mojo_gpu)
    mask = torch.tensor([True, False, True, False]).to(mojo_gpu)
    ret = a.masked_fill_(mask, -1.0)
    assert ret is a
    torch.testing.assert_close(a.cpu(), torch.tensor([-1.0, 2.0, -1.0, 4.0]))


def test_masked_fill__inplace_scalar_noncontiguous(mojo_gpu: str):
    # `.cpu()` on a non-contiguous mojo tensor hits an unrelated, pre-existing
    # gap in `_copy_from` (ops_core.mojo, outside this group), so this reads
    # results back element-by-element through `.item()` instead (already
    # exercised by tests/native/test_bringup.py).
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(mojo_gpu)
    a = base.t()  # non-contiguous view
    mask = torch.zeros_like(a, dtype=torch.bool)
    mask[0, 0] = True
    ret = a.masked_fill_(mask, -9.0)
    assert ret is a
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4).t().clone()
    expected[0, 0] = -9.0
    for i in range(4):
        for j in range(3):
            assert a[i, j].item() == expected[i, j].item()


def test_masked_fill__inplace_tensor(mojo_gpu: str):
    a = torch.tensor([1.0, 2.0, 3.0]).to(mojo_gpu)
    mask = torch.tensor([False, True, False]).to(mojo_gpu)
    value = torch.tensor(42.0).to(mojo_gpu)
    ret = a.masked_fill_(mask, value)
    assert ret is a
    torch.testing.assert_close(a.cpu(), torch.tensor([1.0, 42.0, 3.0]))


# ---------------------------------------------------------------------------
# searchsorted / bucketize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
def test_searchsorted_tensor_batched(
    mojo_device: str, call_checker: CallChecker, dtype: torch.dtype
):
    call_checker.register(aten_functions.aten_searchsorted)
    boundaries = torch.tensor([[-3, 0, 0, 7], [1, 4, 8, 12]], dtype=dtype)
    values = torch.tensor([[-4, 0, 6], [1, 9, 20]], dtype=dtype)
    expected = torch.searchsorted(boundaries, values, side="right")
    got = torch.searchsorted(
        boundaries.to(mojo_device), values.to(mojo_device), side="right"
    )
    torch.testing.assert_close(got.cpu(), expected)


def test_searchsorted_scalar(mojo_device: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_searchsorted)
    boundaries = torch.tensor([-5.0, 0.0, 2.0, 2.0, 9.0])
    expected = torch.searchsorted(boundaries, 2.0, side="right")
    got = torch.searchsorted(boundaries.to(mojo_device), 2.0, side="right")
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize(
    ("boundary_dtype", "value_dtype"),
    [(torch.float16, torch.bfloat16), (torch.int32, torch.int64)],
)
def test_searchsorted_dtype_promotion(
    mojo_device: str,
    call_checker: CallChecker,
    boundary_dtype: torch.dtype,
    value_dtype: torch.dtype,
):
    call_checker.register(aten_functions.aten_searchsorted)
    boundaries = torch.tensor([-5, 0, 2, 9], dtype=boundary_dtype)
    values = torch.tensor([-6, 0, 1, 10], dtype=value_dtype)
    expected = torch.searchsorted(boundaries, values, right=True)
    got = torch.searchsorted(
        boundaries.to(mojo_device), values.to(mojo_device), right=True
    )
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("right", [False, True])
def test_bucketize_tensor(mojo_device: str, call_checker: CallChecker, right: bool):
    call_checker.register(aten_functions.aten_bucketize)
    boundaries = torch.tensor([-5.0, -1.0, 0.0, 2.0, 2.0, 9.0])
    values = torch.tensor([[-6.0, -1.0, 1.0], [2.0, 8.0, 10.0]])
    expected = torch.bucketize(values, boundaries, right=right)
    got = torch.bucketize(
        values.to(mojo_device), boundaries.to(mojo_device), right=right
    )
    torch.testing.assert_close(got.cpu(), expected)


def test_bucketize_scalar(mojo_device: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_bucketize)
    boundaries = torch.tensor([-5.0, 0.0, 2.0, 9.0])
    expected = torch.bucketize(3.0, boundaries)
    got = torch.bucketize(3.0, boundaries.to(mojo_device))
    torch.testing.assert_close(got.cpu(), expected)


def test_searchsorted_and_bucketize_out_variants(mojo_gpu: str):
    boundaries = torch.tensor([1.0, 2.0, 4.0]).to(mojo_gpu)
    values = torch.tensor([0.5, 2.0, 3.0, 5.0]).to(mojo_gpu)
    tensor_out = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
    scalar_out = torch.empty((), dtype=torch.int64, device=mojo_gpu)

    expected_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    expected_scalar = torch.tensor(2, dtype=torch.int64)
    assert torch.searchsorted(boundaries, values, out=tensor_out) is tensor_out
    torch.testing.assert_close(tensor_out.cpu(), expected_tensor)
    assert (
        torch.ops.aten.searchsorted.Scalar_out(boundaries, 3.0, out=scalar_out)
        is scalar_out
    )
    torch.testing.assert_close(scalar_out.cpu(), expected_scalar)

    tensor_out = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
    scalar_out = torch.empty((), dtype=torch.int64, device=mojo_gpu)
    assert torch.bucketize(values, boundaries, out=tensor_out) is tensor_out
    torch.testing.assert_close(tensor_out.cpu(), expected_tensor)
    assert (
        torch.ops.aten.bucketize.Scalar_out(3.0, boundaries, out=scalar_out)
        is scalar_out
    )
    torch.testing.assert_close(scalar_out.cpu(), expected_scalar)


def test_searchsorted_errors(mojo_gpu: str):
    boundaries = torch.tensor([1.0, 3.0, 7.0]).to(mojo_gpu)
    values = torch.tensor([0.0, 4.0]).to(mojo_gpu)

    with pytest.raises(RuntimeError, match="side can only be 'left' or 'right'"):
        torch.searchsorted(boundaries, values, side="middle")
    with pytest.raises(RuntimeError, match="side and right can't be set to opposites"):
        torch.searchsorted(boundaries, values, right=True, side="left")
    with pytest.raises(RuntimeError, match="should have same device type"):
        torch.searchsorted(boundaries, values.cpu())
    with pytest.raises(RuntimeError, match="boundaries tensor must be 1 dimension"):
        torch.bucketize(values, boundaries.unsqueeze(0))


def test_searchsorted_sorter_declined(mojo_gpu: str):
    """The native port declines `sorter` (needs a min/max reduction the
    native reductions group doesn't implement yet -- see the module
    docstring in ops_compare.mojo); this documents the current behavior."""
    boundaries = torch.tensor([1.0, 2.0, 3.0]).to(mojo_gpu)
    values = torch.tensor([0.5, 2.5]).to(mojo_gpu)
    sorter = torch.tensor([0, 1, 2], dtype=torch.int64).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.searchsorted(boundaries, values, sorter=sorter)
