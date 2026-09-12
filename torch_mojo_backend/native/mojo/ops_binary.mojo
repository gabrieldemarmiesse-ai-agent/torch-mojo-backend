"""aten ops: binary arithmetic (add/sub/mul/div/... and their in-place and
out= variants). The bring-up add/mul below go through the logic_ops spec
kernels; the full dispatch cascade of the old fast path follows."""
from std.utils import IndexList

from abi import (
    T,
    Values,
    new_tensor,
    own,
    ret_owned,
    unsupported,
    v_f64,
    v_tensor,
)
from device import ctx_for, ctx_ptr
from kernels import KernelCall
from op_utils import MAX_RANK
from registry import Lib, impl


def _binary_spec(op: StaticString, a: T, b: T, dst: T) raises:
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("logic_ops", String(op))
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, b.dtype)
    call.out_dtype(dst.dtype)
    call.spec(a.spec(cp))
    call.spec(b.spec(cp))
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def _broadcast_shape(a: T, b: T) raises -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    for i in range(MAX_RANK):
        var x = a.shape[i]
        var y = b.shape[i]
        if x == y or y == 1:
            shape[i] = x
        elif x == 1:
            shape[i] = y
        else:
            raise Error("shapes are not broadcastable")
    return shape


def _binary(
    op: StaticString, args: Values, rets: Values, want_alpha: Bool
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    if want_alpha and v_f64(args[unsafe_offset=2]) != 1.0:
        unsupported("alpha != 1")
    if a.stype != b.stype:
        unsupported("mixed dtypes")
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        raise Error("expected both operands on the same mojo device")
    var shape = _broadcast_shape(a, b)
    var rank = max(a.rank, b.rank)
    var out = own(new_tensor(shape, rank, a.stype, a.device))
    _binary_spec(op, a, b, out.t)
    ret_owned(rets, 0, out)


# aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
def op_add_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _binary("AddSpec", args, rets, True)


# aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
def op_mul_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _binary("MulSpec", args, rets, False)


def register_binary(lib: Lib) raises:
    impl[op_add_tensor](lib, "add.Tensor")
    impl[op_mul_tensor](lib, "mul.Tensor")
