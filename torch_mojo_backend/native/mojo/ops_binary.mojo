"""add.Tensor / mul.Tensor through the logic_ops spec kernels (bring-up set;
the full dispatch cascade of the old fast path follows)."""
from std.utils import IndexList

from abi import (
    Values,
    Value,
    T,
    IntList,
    DoubleList,
    v_is_none,
    v_int,
    v_int_or,
    v_f64,
    v_f64_or,
    v_bool,
    v_bool_or,
    v_scalar_is_integral,
    v_scalar_is_bool,
    v_dtype_or,
    v_device_index,
    v_memory_format_or,
    v_generator,
    v_string,
    v_tensor,
    v_opt_tensor,
    v_tensor_list,
    ret_tensor,
    ret_ref,
    ret_int,
    ret_bool,
    ret_f64,
    ret_scalar_int,
    ret_scalar_f64,
    ret_scalar_bool,
    ret_tensor_list,
    contiguous_strides,
    new_strided,
    new_tensor,
    new_like,
    new_like_dtype,
    new_scalar,
    view_strided,
    set_sizes_strides,
    retain,
    release,
    cpu_empty,
    default_dtype,
    unsupported,
    check,
    max_dtype,
    torch_dtype,
    is_floating,
    MEMORY_FORMAT_CONTIGUOUS,
    MEMORY_FORMAT_CHANNELS_LAST,
    TAG_NONE,
    TAG_TENSOR,
    TAG_TENSOR_REF,
)
from device import ctx_for, ctx_ptr
from kernels import KernelCall
from op_utils import MAX_RANK, Arg, TensorSpec


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
    var shape = _broadcast_shape(a, b)
    var rank = max(a.rank, b.rank)
    var out = new_tensor(shape, rank, a.stype, a.device)
    _binary_spec(op, a, b, out)
    ret_tensor(rets, 0, out)


# aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
def op_add_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _binary("AddSpec", args, rets, True)


# aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
def op_mul_tensor(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _binary("MulSpec", args, rets, False)
