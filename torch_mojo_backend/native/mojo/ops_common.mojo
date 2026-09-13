"""Helpers every op group shares: materializing a contiguous copy, strided
copies and fills, dtype casts, and two generic device-level primitives
(Philox reservation, calling any aten op through the real dispatcher) that
more than one group's RNG/host-fallback ops need."""
from std.ffi import external_call
from std.utils import IndexList

from abi import (
    T,
    Value,
    check,
    contiguous_strides,
    dtype_code,
    dtype_itemsize,
    max_dtype,
    new_like,
    new_like_dtype,
    new_tensor,
    release,
    unsupported,
    TAG_NONE,
)
from device import ctx_for, ctx_ptr, memset_bytes, memset_typed
from kernels import KernelCall
from op_utils import MAX_RANK


def _padded(shape: IndexList[MAX_RANK]) -> List[Int]:
    var out = List[Int](capacity=MAX_RANK)
    for i in range(MAX_RANK):
        out.append(shape[i])
    return out^


def copy_strided_into(dst: T, src: T) raises:
    """dst[...] = src[...] for equal logical shapes, any strides, same dtype
    (memory_ops CopyStrided: element-size dispatch, rank <= MAX_RANK)."""
    if dst.numel == 0:
        return
    if src.stype != dst.stype:
        raise Error("copy_strided_into: dtype mismatch")
    if src.numel != dst.numel:
        raise Error("copy_strided_into: element count mismatch")
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("memory_ops", "CopyStrided")
    call.int(dst.ptr)
    call.int(src.ptr)
    call.tuple(_padded(dst.shape))
    call.tuple(_padded(dst.strides))
    call.tuple(_padded(src.strides))
    call.int(dst.itemsize)
    call.int(cp)
    call.run()
    _ = ctx


def contiguous(t: T) raises -> T:
    """`t` itself when already contiguous, else a fresh contiguous copy
    (an owned handle: release it or return it)."""
    if t.contig:
        return t.copy()
    var out = new_like(t)
    copy_strided_into(out, t)
    return out^


def fill_value(t: T, value: Float64) raises:
    """Constant fill for any layout: a memset when contiguous, else the
    strided fill kernel."""
    if t.numel == 0:
        return
    if t.contig:
        _fill_contiguous(t, value)
        return
    var ctx = ctx_for(t.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("memory_ops", "StridedFill")
    call.int(t.ptr)
    call.f64(value)
    call.tuple(_padded(t.shape))
    call.tuple(_padded(t.strides))
    call.int(dtype_code(t.dtype))
    call.int(cp)
    call.run()
    _ = ctx


def _fill_contiguous(t: T, value: Float64) raises:
    var ctx = ctx_for(t.device)
    if value == 0.0:
        memset_bytes(ctx, t.ptr, 0, t.numel * t.itemsize)
    elif t.dtype == DType.float32:
        memset_typed[DType.float32](ctx, t.ptr, Float32(value), t.numel)
    elif t.dtype == DType.bfloat16:
        memset_typed[DType.bfloat16](ctx, t.ptr, BFloat16(value), t.numel)
    elif t.dtype == DType.float16:
        memset_typed[DType.float16](ctx, t.ptr, Float16(value), t.numel)
    elif t.dtype == DType.float64:
        memset_typed[DType.float64](ctx, t.ptr, value, t.numel)
    elif t.dtype == DType.int64:
        memset_typed[DType.int64](ctx, t.ptr, Int64(Int(value)), t.numel)
    elif t.dtype == DType.int32:
        memset_typed[DType.int32](ctx, t.ptr, Int32(Int(value)), t.numel)
    elif t.dtype == DType.int16:
        memset_typed[DType.int16](ctx, t.ptr, Int16(Int(value)), t.numel)
    elif t.dtype == DType.int8:
        memset_typed[DType.int8](ctx, t.ptr, Int8(Int(value)), t.numel)
    elif t.dtype == DType.uint8 or t.dtype == DType.bool:
        memset_bytes(ctx, t.ptr, UInt8(Int(value)), t.numel)
    elif t.dtype == DType.uint16:
        memset_typed[DType.uint16](ctx, t.ptr, UInt16(Int(value)), t.numel)
    elif t.dtype == DType.uint32:
        memset_typed[DType.uint32](ctx, t.ptr, UInt32(Int(value)), t.numel)
    elif t.dtype == DType.uint64:
        memset_typed[DType.uint64](ctx, t.ptr, UInt64(Int(value)), t.numel)
    else:
        unsupported("fill of dtype " + String(t.dtype))
    _ = ctx


def cast_into(dst: T, src: T) raises:
    """dst = src.to(dst.dtype) for a contiguous dst (data_movement_ops CastSpec).
    """
    if src.numel == 0:
        return
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("data_movement_ops", "CastSpec")
    call.arg_dtype(0, src.dtype)
    call.out_dtype(dst.dtype)
    call.spec(src.spec(cp))
    call.int(dtype_code(dst.dtype))
    call.spec(dst.spec(cp))
    call.run()
    _ = ctx


def cast_to(t: T, stype: Int32) raises -> T:
    """A contiguous copy of `t` in dtype `stype` (t itself when unchanged)."""
    if t.stype == stype:
        return t.copy()
    var out = new_like_dtype(t, stype)
    var src = contiguous(t)
    cast_into(out, src)
    if src.h != t.h:
        release(src.h)
    return out^


def resize_storage_for(
    t: T, shape: IndexList[MAX_RANK], rank: Int, offset: Int
) raises:
    """Grow `t`'s storage, if needed, to hold `shape` at `offset`, before
    `abi.set_sizes_strides` reshapes it: `tmb_tensor_set_sizes_strides`
    bounds-checks the new (sizes, strides, offset) against the storage's
    CURRENT byte size, so an `out=` op that grows its target -- the common
    case is a fresh `at::empty({0}, ...)` composite hands `arange.start_out`
    -- must grow the storage first. Mirrors `Tensor::resize_`'s
    grow-if-needed contract (never shrinks the actual allocation); generic
    for any group's `out=` op that may need to grow its target."""
    var numel = 1
    for i in range(rank):
        numel *= shape[MAX_RANK - rank + i]
    var nbytes = (offset + numel) * t.itemsize
    if nbytes > t.storage_nbytes():
        check(
            external_call["tmb_storage_resize", Int32](t.h, Int64(nbytes)),
            "tmb_storage_resize",
        )


def philox_reserve(
    generator: Int, device: Int, increment: Int
) raises -> Tuple[UInt64, UInt64]:
    """Atomically reserve `increment` counters of a device's (generator=0)
    or an explicit generator's Philox stream: returns `(seed, base_offset)`
    as they stood *before* the reservation (tmb_philox_reserve,
    docs/native_backend.md). Generic device-runtime plumbing any RNG op of
    any group needs, not specific to one op group."""
    var seed: UInt64 = 0
    var offset: UInt64 = 0
    check(
        external_call["tmb_philox_reserve", Int32](
            generator,
            Int32(device),
            UInt64(increment),
            Pointer(to=seed),
            Pointer(to=offset),
        ),
        "tmb_philox_reserve",
    )
    return (seed, offset)


def call_op(
    name: String, overload: String, var args: List[Value], n_rets: Int
) raises -> List[Value]:
    """Call any aten op through the real torch dispatcher (tmb_call_op):
    composites and CPU/other-device fallbacks reachable from inside a Mojo
    op body (e.g. `normal_`'s host draw, `arange`'s host fallback). `name`
    must be namespace-qualified (`"aten::normal_"`, not `"normal_"`) --
    `c10::Dispatcher::findSchemaOrThrow` looks it up as one `OperatorName`
    together with `overload` (`""` for the default/unnamed overload,
    `"start_out"` for `aten::arange.start_out`). `args` are value records in
    schema order (exact arity: the dispatcher checks it against the op's
    schema); the result records are returned as-is -- any TAG_TENSOR among
    them is a freshly owned handle the caller must release (or return) like
    any other allocation."""
    var op_name = name
    var op_overload = overload
    var rets = List[Value](capacity=max(n_rets, 1))
    for _ in range(n_rets):
        rets.append(Value(TAG_NONE, 0, 0, 0))
    check(
        external_call["tmb_call_op", Int32](
            op_name.as_c_string_slice().unsafe_ptr(),
            op_overload.as_c_string_slice().unsafe_ptr(),
            args.unsafe_ptr(),
            Int32(len(args)),
            rets.unsafe_ptr(),
            Int32(n_rets),
        ),
        "tmb_call_op",
    )
    return rets^
