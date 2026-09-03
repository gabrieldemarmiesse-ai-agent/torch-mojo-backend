"""The direct-impl table `__torch_dispatch__` uses instead of redispatching.

`deferred_compile._direct` calls the PrivateUse1 callable straight out of
`deferred_compile.DIRECT_IMPLS` rather than re-entering the C++ dispatcher.
That is only sound if a library kernel and `__torch_dispatch__` hand an impl
the same `(args, kwargs)`, so the probe library below registers one kernel per
argument shape most likely to diverge -- a defaulted keyword-only scalar, an
optional dtype, an `int[]?`, an `out=` overload, a TensorList -- and records
what it receives on each path.
"""

import contextlib

import pytest
import torch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.mojo_device import deferred_compile
from torch_mojo_backend.mojo_device.mojo_device_aten_ops import (
    EAGER_CALL_COUNTERS,
    _aten_ops_registry,
)
from torch_mojo_backend.mojo_device.register import _resolve_overload

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_mojo_device():
    register_mojo_devices()


def test_registered_ops_resolve_into_the_direct_table():
    names = [name for name, _ in _aten_ops_registry]
    unresolved = [name for name in names if _resolve_overload(name) is None]
    # A name may only miss the table because this torch build has no such
    # overload (`aten::empty_strided.memory_format` today). It keeps its
    # library registration and goes on redispatching through C++.
    for name in unresolved:
        packet_name, _, overload = name.removeprefix("aten::").partition(".")
        packet = getattr(torch.ops.aten, packet_name, None)
        assert packet is None or (overload or "default") not in packet.overloads()
    assert len(deferred_compile.DIRECT_IMPLS) == len(set(names)) - len(unresolved)


def test_table_holds_the_call_counted_callable():
    """`CallChecker` observes ops through EAGER_CALL_COUNTERS, so the table has
    to hold the wrapper that increments them, not the bare impl."""
    counted = EAGER_CALL_COUNTERS["aten::add.Tensor"]
    before = counted.call_count
    x = torch.ones(4, device="mojo")
    (x + x).cpu()
    assert counted.call_count > before
    table_entry = deferred_compile.DIRECT_IMPLS[torch.ops.aten.add.Tensor]
    assert getattr(table_entry, "__wrapped__", None) is counted


@contextlib.contextmanager
def _table_disabled(overload: torch._ops.OpOverload):
    """Force `overload` back onto the C++ redispatch path."""
    entry = deferred_compile.DIRECT_IMPLS.pop(overload, None)
    try:
        yield
    finally:
        if entry is not None:
            deferred_compile.DIRECT_IMPLS[overload] = entry


# ---------------------------------------------------------------------------
# Argument-convention probe
# ---------------------------------------------------------------------------

_PROBE_SCHEMAS = (
    "probe_alpha(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "probe_dtype(Tensor self, *, ScalarType? dtype=None, bool non_blocking=False)"
    " -> Tensor",
    "probe_dims(Tensor self, int[1]? dim=None, bool keepdim=False, *,"
    " ScalarType? dtype=None) -> Tensor",
    "probe_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)",
    "probe_list(Tensor[] self, Scalar other) -> ()",
)


@pytest.fixture(scope="module")
def probe():
    """A private library whose PrivateUse1 kernels only record their call.

    Yields `(ops_namespace, seen, kernels)`; `kernels[name]` is the very
    callable the library holds, so putting it in DIRECT_IMPLS reproduces what
    `register_mojo_devices` does for the aten ops.
    """
    library = torch.library.Library("torch_mojo_probe", "DEF")
    seen = []
    kernels = {}

    def make(name):
        def kernel(*args, **kwargs):
            seen.append((args, dict(kwargs)))
            if name == "probe_list":
                return None
            if name == "probe_out":
                return kwargs["out"]
            return args[0]

        return kernel

    for schema in _PROBE_SCHEMAS:
        library.define(schema)
        name = schema.split("(")[0]
        kernels[name] = make(name)
        library.impl(name, kernels[name], "PrivateUse1")
    try:
        yield torch.ops.torch_mojo_probe, seen, kernels
    finally:
        library._destroy()


def _both_paths(probe, name, call):
    """Run `call()` once through the C++ redispatch and once through the
    table, returning the `(args, kwargs)` the kernel saw each time."""
    ops, seen, kernels = probe
    overload = getattr(ops, name).default
    seen.clear()
    with _table_disabled(overload):
        call()
        assert len(seen) == 1, f"{name} never reached its PrivateUse1 kernel"
        through_cpp = seen.pop()
    deferred_compile.DIRECT_IMPLS[overload] = kernels[name]
    try:
        call()
        assert len(seen) == 1
        direct = seen.pop()
    finally:
        deferred_compile.DIRECT_IMPLS.pop(overload, None)
    return through_cpp, direct


def _same(left, right):
    """Structural equality that compares tensors by identity."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return left is right
    if isinstance(left, tuple | list) and isinstance(right, tuple | list):
        return type(left) is type(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same(left[k], right[k]) for k in left
        )
    return type(left) is type(right) and left == right


def _probe_alpha_given(ops):
    tensor = torch.ones(4, device="mojo")
    return lambda: ops.probe_alpha(tensor, tensor, alpha=2)


def _probe_alpha_defaulted(ops):
    tensor = torch.ones(4, device="mojo")
    return lambda: ops.probe_alpha(tensor, tensor)


def _probe_dtype_given(ops):
    tensor = torch.ones(4, device="mojo")
    return lambda: ops.probe_dtype(tensor, dtype=torch.float16)


def _probe_dtype_defaulted(ops):
    tensor = torch.ones(4, device="mojo")
    return lambda: ops.probe_dtype(tensor)


def _probe_dims_positional(ops):
    tensor = torch.ones(4, device="mojo")
    return lambda: ops.probe_dims(tensor, [0], True)


def _probe_dims_keyword(ops):
    tensor = torch.ones(4, device="mojo")
    return lambda: ops.probe_dims(tensor, dim=[0], keepdim=True)


def _probe_dims_defaulted(ops):
    tensor = torch.ones(4, device="mojo")
    return lambda: ops.probe_dims(tensor)


def _probe_out(ops):
    tensor = torch.ones(4, device="mojo")
    out = torch.empty(4, device="mojo")
    return lambda: ops.probe_out(tensor, 3, out=out)


def _probe_list(ops):
    tensor = torch.ones(4, device="mojo")
    return lambda: ops.probe_list([tensor, tensor], 2.5)


@pytest.mark.parametrize(
    "name,build",
    [
        # A keyword-only Scalar, passed and left defaulted.
        ("probe_alpha", _probe_alpha_given),
        ("probe_alpha", _probe_alpha_defaulted),
        # Optional ScalarType, passed and left defaulted.
        ("probe_dtype", _probe_dtype_given),
        ("probe_dtype", _probe_dtype_defaulted),
        # int[1]? plus a positional bool, spelled three ways.
        ("probe_dims", _probe_dims_positional),
        ("probe_dims", _probe_dims_keyword),
        ("probe_dims", _probe_dims_defaulted),
        # out= overload.
        ("probe_out", _probe_out),
        # TensorList.
        ("probe_list", _probe_list),
    ],
)
def test_both_paths_see_the_same_arguments(probe, name, build):
    ops, _, _ = probe
    through_cpp, direct = _both_paths(probe, name, build(ops))
    assert _same(through_cpp, direct), f"{name}: {through_cpp} != {direct}"


# ---------------------------------------------------------------------------
# Real aten ops: same result either way.
# ---------------------------------------------------------------------------


def _add_alpha(device):
    a = torch.arange(6.0, device=device).reshape(2, 3)
    return torch.ops.aten.add.Tensor(a, a, alpha=3)


def _to_copy(device):
    a = torch.arange(6.0, device=device)
    return torch.ops.aten._to_copy.default(a, dtype=torch.float16)


def _sum_keepdim(device):
    a = torch.arange(6.0, device=device).reshape(2, 3)
    return torch.ops.aten.sum.dim_IntList(a, [1], True)


def _addcmul_out(device):
    a = torch.arange(6.0, device=device)
    out = torch.empty(6, device=device)
    return torch.ops.aten.addcmul.out(a, a, a, value=2, out=out)


def _foreach_mul_(device):
    tensors = [torch.arange(4.0, device=device), torch.arange(3.0, device=device)]
    torch.ops.aten._foreach_mul_.Scalar(tensors, 2.5)
    return tensors


@pytest.mark.parametrize(
    "overload,run",
    [
        (torch.ops.aten.add.Tensor, _add_alpha),
        (torch.ops.aten._to_copy.default, _to_copy),
        (torch.ops.aten.sum.dim_IntList, _sum_keepdim),
        (torch.ops.aten.addcmul.out, _addcmul_out),
        (torch.ops.aten._foreach_mul_.Scalar, _foreach_mul_),
    ],
)
def test_real_ops_agree_on_both_paths(overload, run):
    def to_cpu(value):
        if isinstance(value, list):
            return [t.cpu() for t in value]
        return value.cpu()

    direct = to_cpu(run("mojo"))
    with _table_disabled(overload):
        assert overload not in deferred_compile.DIRECT_IMPLS
        redispatched = to_cpu(run("mojo"))
    expected = to_cpu(run("cpu"))
    if isinstance(expected, list):
        for a, b, c in zip(direct, redispatched, expected, strict=True):
            torch.testing.assert_close(a, c)
            torch.testing.assert_close(b, c)
    else:
        torch.testing.assert_close(direct, expected)
        torch.testing.assert_close(redispatched, expected)
