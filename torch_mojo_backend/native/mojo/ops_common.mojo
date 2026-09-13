"""Helpers every op group shares: materializing a contiguous copy, strided
copies and fills, dtype casts (through the memory_ops / data_movement_ops
families on the tensor's current stream), scalar embedding and binary type
promotion, `out=` resizing, and two device-level primitives (Philox
reservation, calling any aten op through the real dispatcher) that more than
one group needs."""
from std.ffi import external_call
from std.utils import IndexList

from abi import (
    T,
    Value,
    check,
    UNSUPPORTED_PREFIX,
    Values,
    contiguous_strides,
    dtype_code,
    dtype_itemsize,
    max_dtype,
    new_like,
    new_like_dtype,
    new_tensor,
    release,
    set_sizes_strides,
    torch_dtype,
    shim_error,
    unsupported,
    TAG_NONE,
    v_f64,
    v_scalar_is_integral,
    v_int,
)
from device import ctx_for, ctx_ptr, dev, memset_bytes, memset_typed
from kernels import KernelCall
from op_utils import MAX_RANK


def call_op_raw(
    op: String,
    overload: String,
    args: Values,
    n_args: Int,
    rets: Values,
    n_rets: Int,
) raises:
    """Call any aten op through torch's dispatcher (`tmb_call_op`) over
    caller-owned record arrays; `call_op` below is the List-based form.

    What an op uses to reach a neighbouring op's kernel or ATen's own
    composite: the records are the same ones a kernel gets, tensor arguments
    are borrowed and tensor results come back as owned handles. Dispatch is on
    the arguments, so an op must never call *itself* this way. A declining
    kernel comes back as `unsupported` (rc 2) and keeps its prefix, so a
    caller with another route can tell it apart from a real failure.
    """
    var name = String(op)
    var over = String(overload)
    var rc = external_call["tmb_call_op", Int32](
        name.as_c_string_slice().unsafe_ptr(),
        over.as_c_string_slice().unsafe_ptr(),
        args,
        Int32(n_args),
        rets,
        Int32(n_rets),
    )
    if rc == 2:
        raise Error(UNSUPPORTED_PREFIX, shim_error())
    if rc != 0:
        raise Error(op, ": ", shim_error())


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


def resize_out(mut t: T, shape: IndexList[MAX_RANK], rank: Int) raises:
    """Resize a caller's `out=` tensor in place to a fresh contiguous
    `shape` (torch's generic `resize_output` semantics, for a backend with
    no registered `aten::resize_` kernel of its own: every `out=` op here
    must do this itself for an out tensor of the wrong shape, rather than
    relying on a resize that would otherwise happen before dispatch).

    Grows the storage through the shim's allocator when the new shape needs
    more bytes (`tmb_storage_resize`, which preserves existing bytes up to
    min(old, new) like torch's own `resize_`), then rewrites sizes/strides
    (`tmb_tensor_set_sizes_strides`, which requires the storage already be
    big enough -- hence the order). `t`'s cached view fields are refreshed
    from the tensor afterward since its shape/strides/numel/contig changed.
    """
    var strides = contiguous_strides(shape, rank)
    var numel = 1
    for i in range(rank):
        numel *= shape[MAX_RANK - rank + i]
    var nbytes = numel * t.itemsize
    if nbytes > t.storage_nbytes():
        check(
            external_call["tmb_storage_resize", Int32](t.h, Int64(nbytes)),
            "tmb_storage_resize",
        )
    set_sizes_strides(t, shape, strides, rank, 0)
    t = T(t.h)


def contiguous(t: T) raises -> T:
    """`t` itself when already contiguous, else a fresh contiguous copy
    (an owned handle: release it or return it)."""
    if t.contig:
        return t.copy()
    var out = new_like(t)
    copy_strided_into(out, t)
    return out^


def fill_value(t: T, value: Float64) raises:
    """Constant fill for any layout: a memset when contiguous on an
    accelerator, else the strided fill kernel. On the MAX CPU device a
    memset is not ordered against kernel launches (measured: a kernel
    reading a just-filled buffer saw stale memory 4 times in 50), so that
    device always fills with the kernel."""
    if t.numel == 0:
        return
    if t.contig and not dev(t.device)[].is_cpu:
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


# ---------------------------------------------------------------------------
# Scalar embedding and binary dtype promotion, shared by the compare and
# binary op groups (ported from `_scalar_embed` / `_binary_promotion` /
# `_promoted_pair` in the old eager_kernels/aten_fast.py).
# ---------------------------------------------------------------------------

# int64 scalars round-trip through a Float64 fill argument exactly up to
# this magnitude.
comptime _MAX_EXACT_INT = 9007199254740992  # 2**53


def _is_cast_dtype(dtype: DType) -> Bool:
    """Dtypes `binary_promotion`/`promoted_pair` can materialize a cast into
    (matches the old `_CAST_DTYPES`)."""
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.int64
        or dtype == DType.int32
        or dtype == DType.uint8
        or dtype == DType.bool
    )


def _is_embeddable_dtype(dtype: DType) -> Bool:
    """Dtypes `scalar_embed`'s destination fill can target (matches the old
    `_FILL_DTYPES`; a strict subset of what `fill_value` itself supports,
    kept for fidelity with the old eager path)."""
    return (
        dtype == DType.float32
        or dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.float64
        or dtype == DType.int8
        or dtype == DType.int16
        or dtype == DType.int32
        or dtype == DType.int64
        or dtype == DType.uint8
        or dtype == DType.bool
    )


def scalar_embed(v: Value, dtype: DType) raises -> Float64:
    """`v` (an ATen Scalar record) validated for lossless embedding into
    `dtype`, as a Float64 ready for `fill_value` / a 0-d fill tensor.

    Ported from `_scalar_embed`: an int/bool magnitude above 2**53 would
    lose precision through the Float64 round-trip and is declined, a bool
    destination only accepts 0/1, and a float scalar against a
    non-floating destination is declined (no implicit promotion here --
    callers that want promotion cast the tensor operand first).
    """
    if not _is_embeddable_dtype(dtype):
        unsupported("scalar embedding into dtype " + String(dtype))
    if v_scalar_is_integral(v):
        var i = v_int(v)
        if abs(i) > _MAX_EXACT_INT:
            unsupported("scalar magnitude exceeds the exact float64 range")
        if dtype == DType.bool and i != 0 and i != 1:
            unsupported("a bool tensor's scalar must be 0 or 1")
        return Float64(i)
    if (
        dtype != DType.float16
        and dtype != DType.bfloat16
        and dtype != DType.float32
        and dtype != DType.float64
    ):
        unsupported("a float scalar against a non-floating tensor")
    return v_f64(v)


def binary_promotion(a_dtype: DType, b_dtype: DType) raises -> DType:
    """torch's promotion for a binary pair, restricted to what the
    broadcast-strided spec kernels cover: equal dtypes; bool with any
    castable dtype; int32/int64; float32 with float16/bfloat16; and
    float16<->bfloat16 (widens both sides to float32). Declines
    (`unsupported`) any other pair.

    A caller casts each operand into the returned dtype with
    `cast_to(operand, torch_dtype(result))`, which already no-ops when an
    operand is already that dtype -- so, unlike the old `_binary_promotion`,
    this returns just the common dtype rather than a (cast lhs?, cast rhs?,
    dtype) triple; there is no separate cast-skipping fast path to expose.
    Ported from `_binary_promotion`.
    """
    if a_dtype == b_dtype:
        return a_dtype
    if a_dtype == DType.bool and _is_cast_dtype(b_dtype):
        return b_dtype
    if b_dtype == DType.bool and _is_cast_dtype(a_dtype):
        return a_dtype
    if a_dtype == DType.int32 and b_dtype == DType.int64:
        return DType.int64
    if b_dtype == DType.int32 and a_dtype == DType.int64:
        return DType.int64
    if a_dtype == DType.float32 and (
        b_dtype == DType.float16 or b_dtype == DType.bfloat16
    ):
        return DType.float32
    if b_dtype == DType.float32 and (
        a_dtype == DType.float16 or a_dtype == DType.bfloat16
    ):
        return DType.float32
    if (a_dtype == DType.float16 and b_dtype == DType.bfloat16) or (
        a_dtype == DType.bfloat16 and b_dtype == DType.float16
    ):
        return DType.float32
    unsupported(
        "no dtype promotion for " + String(a_dtype) + " and " + String(b_dtype)
    )
    return a_dtype


def promoted_pair(a: T, b: T) raises -> Tuple[T, T]:
    """Same-dtype tensor pair following torch's promotion, materializing a
    cast side through `cast_to`.

    A deliberate SUBSET of `binary_promotion` (bool+castable, int32/int64
    only): `where`/`masked_fill` must keep declining mixed float widths
    rather than silently widening them here. Ported from `_promoted_pair`.
    The caller releases whichever of the pair has a handle (`.h`) different
    from the corresponding input -- that side was freshly materialized.
    """
    if a.dtype == b.dtype:
        return (a.copy(), b.copy())
    if a.dtype == DType.bool and _is_cast_dtype(b.dtype):
        return (cast_to(a, torch_dtype(b.dtype)), b.copy())
    if b.dtype == DType.bool and _is_cast_dtype(a.dtype):
        return (a.copy(), cast_to(b, torch_dtype(a.dtype)))
    if a.dtype == DType.int32 and b.dtype == DType.int64:
        return (cast_to(a, torch_dtype(DType.int64)), b.copy())
    if b.dtype == DType.int32 and a.dtype == DType.int64:
        return (a.copy(), cast_to(b, torch_dtype(DType.int64)))
    unsupported("mixed dtypes " + String(a.dtype) + " and " + String(b.dtype))
    return (a.copy(), b.copy())


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
