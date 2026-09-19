# C entry of native_dropout / native_dropout_backward (kernels:
# native_dropout_kernels.mojo). Slots are unpacked here; nothing is read from
# the host or synchronized.

from native_dropout_kernels import (
    I64x8,
    enqueue_native_dropout,
    enqueue_native_dropout_backward,
)
from op_utils import (
    Arg,
    Argv,
    _raw_ctx,
    _raw_f64,
    _raw_int,
    _raw_tuple_int,
    _raw_tuple_len,
    _spec_dispatcher6,
    _spec_dispatcher16,
)

from variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime DROPOUT_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]


@always_inline
def _join_u64(lo: Int, hi: Int) -> UInt64:
    return UInt64(lo) | (UInt64(hi) << 32)


@always_inline
def _tuple_i64x8(t: Arg) -> I64x8:
    var out = I64x8(fill=Int64(0))
    var n = _raw_tuple_len(t)
    for i in range(min(n, 8)):
        out[i] = Int64(_raw_tuple_int(t, i))
    return out^


def _native_dropout_go(
    out_obj: Arg,
    mask_obj: Arg,
    in_obj: Arg,
    numel_obj: Arg,
    ndim_obj: Arg,
    sizes_obj: Arg,
    in_strides_obj: Arg,
    out_strides_obj: Arg,
    vec_obj: Arg,
    grid_obj: Arg,
    keep_p_obj: Arg,
    seed_lo_obj: Arg,
    seed_hi_obj: Arg,
    offset_lo_obj: Arg,
    offset_hi_obj: Arg,
    ctx_obj: Arg,
) raises:
    var seed = _join_u64(_raw_int(seed_lo_obj), _raw_int(seed_hi_obj))
    var offset = _join_u64(_raw_int(offset_lo_obj), _raw_int(offset_hi_obj))
    var handled = False

    comptime for dt in DROPOUT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            enqueue_native_dropout[dt](
                _raw_ctx(ctx_obj),
                _raw_int(out_obj),
                _raw_int(mask_obj),
                _raw_int(in_obj),
                _raw_int(numel_obj),
                _raw_int(ndim_obj),
                _tuple_i64x8(sizes_obj),
                _tuple_i64x8(in_strides_obj),
                _tuple_i64x8(out_strides_obj),
                _raw_int(vec_obj),
                _raw_int(grid_obj),
                _raw_f64(keep_p_obj),
                seed,
                offset,
            )
            handled = True
    if not handled:
        raise Error("native_dropout: no dtype compiled into this module")


def _native_dropout_backward_go(
    grad_input_obj: Arg,
    grad_obj: Arg,
    mask_obj: Arg,
    numel_obj: Arg,
    scale_obj: Arg,
    ctx_obj: Arg,
) raises:
    var handled = False

    comptime for dt in DROPOUT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            enqueue_native_dropout_backward[dt](
                _raw_ctx(ctx_obj),
                _raw_int(grad_input_obj),
                _raw_int(grad_obj),
                _raw_int(mask_obj),
                _raw_int(numel_obj),
                _raw_f64(scale_obj),
            )
            handled = True
    if not handled:
        raise Error(
            "native_dropout_backward: no dtype compiled into this module"
        )


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime if _op_on["NativeDropout"]():
            _spec_dispatcher16[_native_dropout_go, "NativeDropout"](argv, argc)
            return 0
        comptime if _op_on["NativeDropoutBackward"]():
            _spec_dispatcher6[
                _native_dropout_backward_go, "NativeDropoutBackward"
            ](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
