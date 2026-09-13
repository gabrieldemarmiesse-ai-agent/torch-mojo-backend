"""Helpers every op group shares: materializing a contiguous copy, strided
copies and fills, dtype casts. All of them go through the memory_ops /
data_movement_ops families on the tensor's current stream."""
from std.utils import IndexList

from abi import (
    T,
    contiguous_strides,
    dtype_code,
    dtype_itemsize,
    max_dtype,
    new_like,
    new_like_dtype,
    new_tensor,
    release,
    unsupported,
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


def release_if_new(result: T, original: T):
    """Release `result` only when it is a fresh allocation distinct from
    `original`. `contiguous()`/`cast_to()` alias their input (returning it
    unchanged, via `T.copy()`) instead of allocating whenever the input
    already has the requested layout/dtype -- so a caller that wraps their
    result in `own()` unconditionally would release a handle it never
    allocated (an argument the caller only borrowed, e.g. an op's `self`).
    Call this instead of `own(...)` whenever the input might be a borrowed
    tensor."""
    if result.h != original.h:
        release(result.h)
