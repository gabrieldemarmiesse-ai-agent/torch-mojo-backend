# ===----------------------------------------------------------------------=== #
# On-device `aten::uniform_` for mojo_device eager mode.
#
# RNG design
# ==========
# Counter-based Philox4x32-10, the same stateless generator (and the same
# `_philox4x32_10` implementation) `native_dropout_kernels.mojo` documents and
# `nn_ops.mojo` already reuses.  The host reserves a Philox counter interval
# per call (`_reserve_philox_state`) and passes `(seed, base_offset)`; nothing
# about the stream depends on launch geometry, vector width or tail handling,
# so the same `(seed, base_offset, size)` reproduces the same tensor on any
# grid and on any of CPU/CUDA/ROCm/Metal.
#
# One Philox evaluation yields four 32-bit words, and how many elements those
# four words cover depends on how many bits an element needs:
#
# - 32-bit arithmetic (float32/float16/bfloat16): one word per element, so
#   element `i` sits in group `g = i // 4` at lane `i % 4`.
# - float64: 53 mantissa bits need two words, so element `i` sits in group
#   `g = i // 2` and consumes lanes `(2 * (i % 2), 2 * (i % 2) + 1)`.
#
# The host reserves `ceildiv(size * words_per_element, 4)` counters, which is
# exactly the set of groups this file reads.  A ragged final group consumes
# its whole counter and emits only the elements that exist, so the next draw
# starts past it and two adjacent draws can never overlap.
#
# Why not `nn.rand_uniform` from the modular repo: it reads its seed from a
# device pointer (a staged allocation plus a copy per call, and no reserved
# counter interval, so repeated calls would redraw the same stream), and it
# documents its float range as the CLOSED `[lower_bound, upper_bound]` -- it
# scales in float32 and casts down, which is precisely the endpoint bug the
# transform below is written to avoid.
#
# The transform
# =============
# `uniform_(from, to)` must return values in the HALF-OPEN `[from, to)`.  Two
# separate roundings can push a value onto the closed endpoint, and both are
# handled here:
#
# 1. The word -> unit-interval step is exact by construction: the low 24 bits
#    (float32 arithmetic) or low 53 bits (float64) scaled by 2**-24 / 2**-53
#    are integers below 2**24 / 2**53, so both the integer and the product are
#    exactly representable and `u` lands in [0, 1 - 2**-24] / [0, 1 - 2**-53].
#    Nothing here can round up to 1.0.
# 2. `u * range + from` in the arithmetic type, then a narrowing cast to the
#    output dtype, can still reach `to`: for bfloat16 (8 mantissa bits) any
#    `u` above 1 - 2**-9 rounds to 1.0, which happens for one draw in ~512.
#    So the value is compared against `to` AFTER the cast and folded back to
#    `from`, the way ATen's own CUDA kernel does
#    (`aten/src/ATen/native/cuda/DistributionTemplates.h`, `uniform_kernel`:
#    `value == to ? from : value`, "BEFORE TOUCHING THIS CODE READ
#    pytorch/pytorch#96947").  The comparison here is `>=` rather than `==`:
#    `range` is itself the rounded difference `fl(to - from)`, so
#    `fl(from + range)` is not guaranteed to be exactly `to`, and `>=` covers
#    that double rounding as well.  Rounding is monotonic, so no draw can
#    exceed the folded endpoint: every surviving value is strictly below `to`.
#
# Bit-exact agreement with ATen is neither claimed nor possible (a different
# generator and a different word->float mapping); the distribution and the
# bounds are what must be right.
# ===----------------------------------------------------------------------=== #

from max.algorithm import elementwise
from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.sys.info import has_accelerator, has_apple_gpu_accelerator, size_of
from std.utils.coord import Coord

from native_dropout_kernels import _philox4x32_10
from op_utils import FILL_THREADS, _enqueue_cached, _fill_blocks, _make_ptr


@always_inline
def _uniform_math_dtype[dtype: DType]() -> DType:
    """The type the transform is evaluated in (ATen's `opmath_type`)."""
    comptime if dtype == DType.float64:
        return DType.float64
    else:
        return DType.float32


@always_inline
def _uniform_group[dtype: DType]() -> Int:
    """Elements one Philox4x32-10 evaluation supplies for `dtype`."""
    comptime if dtype == DType.float64:
        return 2
    else:
        return 4


@always_inline
def _uniform_tag[dtype: DType]() -> StaticString:
    """The dtype token this family's kernels carry in their profiler name.

    Kernel names are what CUPTI, Nsight and torch.profiler print, so the name
    has to say which dtype actually ran.
    """
    comptime if dtype == DType.float64:
        return "f64"
    elif dtype == DType.float32:
        return "f32"
    elif dtype == DType.float16:
        return "f16"
    else:
        return "bf16"


@always_inline
def _uniform_store_tag[WIDE: Bool]() -> StaticString:
    """`v` for one vector store per group, `s` for scalar stores."""
    comptime if WIDE:
        return "v"
    else:
        return "s"


@always_inline
def _uniform_draw[
    dtype: DType, GROUP: Int
](
    rnd: SIMD[DType.uint32, 4],
    from_out: Scalar[dtype],
    to_out: Scalar[dtype],
    from_math: Scalar[_uniform_math_dtype[dtype]()],
    range_math: Scalar[_uniform_math_dtype[dtype]()],
) -> SIMD[dtype, GROUP]:
    """One Philox group's four words as `GROUP` `[from, to)` values.

    `GROUP` is `_uniform_group[dtype]()` -- passed as a parameter, not
    recomputed, so the vector length in the signature is one symbol the type
    checker can match at every call site.

    See the module header for why the unit-interval step cannot reach 1.0 and
    why the endpoint fold uses `>=`.
    """
    comptime assert GROUP == _uniform_group[dtype]()
    comptime if _uniform_math_dtype[dtype]() == DType.float32:
        # 24 = float32's mantissa bits, so both operands and the product are
        # exact; ATen's CPU `uniform_real_distribution` masks the same way.
        comptime SCALE = Float32(1.0) / Float32(1 << 24)
        var mantissa = rnd & SIMD[DType.uint32, 4](0x00FF_FFFF)
        var u = mantissa.cast[DType.float32]() * SIMD[DType.float32, 4](SCALE)
        var value = (
            u * SIMD[DType.float32, 4](range_math)
            + SIMD[DType.float32, 4](from_math)
        ).cast[dtype]()
        var folded = value.ge(SIMD[dtype, 4](to_out)).select(
            SIMD[dtype, 4](from_out), value
        )
        return rebind[SIMD[dtype, GROUP]](folded)
    else:
        # 53 = float64's mantissa bits; two words per element, high word
        # first, matching the group/lane mapping in the module header.
        comptime SCALE = Float64(1.0) / Float64(1 << 53)
        var word = rnd.cast[DType.uint64]()
        var high = SIMD[DType.uint64, 2](word[0], word[2])
        var low = SIMD[DType.uint64, 2](word[1], word[3])
        var bits = (high << 32) | low
        var u = (bits >> 11).cast[DType.float64]() * SIMD[DType.float64, 2](
            SCALE
        )
        var value = (
            u * SIMD[DType.float64, 2](range_math)
            + SIMD[DType.float64, 2](from_math)
        ).cast[dtype]()
        var folded = value.ge(SIMD[dtype, 2](to_out)).select(
            SIMD[dtype, 2](from_out), value
        )
        return rebind[SIMD[dtype, GROUP]](folded)


# Named for the profiler: the generator, the layout regime, the dtype that ran
# and how one thread stores its group (`_v4` = one 16-byte store of four
# float32 values, `_s4` = the same four values stored one at a time).
@__name(
    t"uniform_philox_contig_{_uniform_tag[dtype]()}_{_uniform_store_tag[WIDE]()}{_uniform_group[dtype]()}"
)
def _uniform_kernel[
    dtype: DType, WIDE: Bool
](
    dst_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    from_out: Scalar[dtype],
    to_out: Scalar[dtype],
    from_math: Scalar[_uniform_math_dtype[dtype]()],
    range_math: Scalar[_uniform_math_dtype[dtype]()],
    groups_arg: Int64,
    size_arg: Int64,
    seed: UInt64,
    base_offset: UInt64,
):
    """`dst[i] = uniform(from, to)` over `size` contiguous elements.

    One thread owns one Philox group, grid-stride so any grid covers any
    length. `WIDE` is the launcher's runtime alignment decision made
    compile-time: it is only instantiated when the base address is aligned
    for the whole group AND `size` is a multiple of the group, so the bounds
    test is not needed there.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var groups = Int(groups_arg)
    var size = Int(size_arg)
    comptime GROUP = _uniform_group[dtype]()
    comptime ALIGN = min(16, GROUP * size_of[dtype]())
    var group = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)

    while group < groups:
        var values = _uniform_draw[dtype, GROUP](
            _philox4x32_10(base_offset + UInt64(group), seed),
            from_out,
            to_out,
            from_math,
            range_math,
        )
        var base = group * GROUP
        comptime if WIDE:
            dst_ptr.store[alignment=ALIGN](base, values)
        else:
            comptime for lane in range(GROUP):
                if base + lane < size:
                    dst_ptr[base + lane] = values[lane]
        group += gstride


def enqueue_uniform[
    dtype: DType
](
    dst_addr: Int,
    from_value: Float64,
    to_value: Float64,
    size: Int,
    seed: UInt64,
    base_offset: UInt64,
    ctx: DeviceContext,
) raises:
    """Fill `size` contiguous elements at `dst_addr` from `[from, to)`.

    The endpoints arrive as Float64 (one ABI for every dtype) and are narrowed
    HERE, on the host: nothing float64 may reach a Metal kernel, and the
    device only ever sees the output dtype and the arithmetic dtype.
    """
    comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
        if ctx.api() != "cpu":
            raise Error("float64 is not supported on Apple GPU")
    if size <= 0:
        return

    comptime MATH = _uniform_math_dtype[dtype]()
    comptime GROUP = _uniform_group[dtype]()
    # ATen narrows both endpoints to the output dtype before the transform
    # (`uniform_kernel`'s `static_cast<scalar_t>`), so the fold below compares
    # against the same `to` the values were rounded into.
    var from_out = from_value.cast[dtype]()
    var to_out = to_value.cast[dtype]()
    var from_math = from_out.cast[MATH]()
    var range_math = to_out.cast[MATH]() - from_math
    var groups = ceildiv(size, GROUP)
    var dst_ptr = _make_ptr[dtype](dst_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(
            dst_ptr,
            from_out,
            to_out,
            from_math,
            range_math,
            size,
            seed,
            base_offset,
        )
        def cpu_group[width: Int, alignment: Int = 1](idx: Coord):
            var group = Int(idx[0].value())
            var values = _uniform_draw[dtype, GROUP](
                _philox4x32_10(base_offset + UInt64(group), seed),
                from_out,
                to_out,
                from_math,
                range_math,
            )
            var base = group * GROUP

            comptime for lane in range(GROUP):
                if base + lane < size:
                    dst_ptr[base + lane] = values[lane]

        elementwise[cpu_group, simd_width=1](Coord(groups), ctx)
        return

    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        # A ragged length or an under-aligned base is a correctness question,
        # not a speed one: the wide store has to be declined on the address
        # itself, since a tensor can start at any element offset inside its
        # storage (`x[1:]`).
        comptime ALIGN = min(16, GROUP * size_of[dtype]())
        var wide = size % GROUP == 0 and dst_addr % ALIGN == 0

        # `_fill_blocks`: one thread per group, capped, exactly as the
        # constant fill does -- a generated fill reads nothing and has no
        # reuse to protect either, so the same "cover the slots" rule applies
        # and no constant fitted to one card enters the geometry.
        if wide:
            _enqueue_cached[_uniform_kernel[dtype, True]](
                ctx,
                String(t"uniform_{dtype}_v{GROUP}"),
                _fill_blocks(groups),
                1,
                1,
                FILL_THREADS,
                dst_ptr.as_unsafe_any_origin(),
                from_out,
                to_out,
                from_math,
                range_math,
                Int64(groups),
                Int64(size),
                seed,
                base_offset,
            )
        else:
            _enqueue_cached[_uniform_kernel[dtype, False]](
                ctx,
                String(t"uniform_{dtype}_s{GROUP}"),
                _fill_blocks(groups),
                1,
                1,
                FILL_THREADS,
                dst_ptr.as_unsafe_any_origin(),
                from_out,
                to_out,
                from_math,
                range_math,
                Int64(groups),
                Int64(size),
                seed,
                base_offset,
            )
