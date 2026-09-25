"""Exactness probe for tests/native/test_gpu_elementwise_launcher.py.

Candidate NVIDIA launcher (`tmb.kernels.common.gpu_elementwise`) vs MAX
`elementwise`, bit for bit. For every (dtype, simd_width, shape) the same
body runs through both launchers into separate outputs with 64 sentinel
guard elements on each side; outputs must be bit-identical, every in-range
element written (no sentinel left), and no guard element touched. The body
loads with the `alignment` the launcher claims, so a wrong alignment claim
faults. Covers ranks 1-3, empty, tails, and both closure forms.

Build and run locked (needs a real NVIDIA GPU): see
test_gpu_elementwise_launcher.py, which takes /tmp/gpu_lock_0.lock.
"""
from std.memory import bitcast
from std.sys import size_of
from std.collections import List
from std.utils.coord import Coord
from std.utils.index import IndexList
from max.algorithm import elementwise as max_elementwise
from max.gpu.host import DeviceContext, DeviceBuffer
from tmb.kernels.common.gpu_elementwise import elementwise as cand_elementwise

comptime MutPtr[dtype: DType] = Pointer[Scalar[dtype], MutUntrackedOrigin]
comptime GUARD = 64


def _ptr[dtype: DType](buf: DeviceBuffer[dtype], offset: Int) -> MutPtr[dtype]:
    return MutPtr[dtype](
        unsafe_from_address=Int(buf.unsafe_ptr()) + offset * size_of[dtype]()
    )


def run_one[
    dtype: DType, rank: Int, simd_width: Int, use_cand: Bool, unified: Bool
](
    ctx: DeviceContext,
    shape: IndexList[rank],
    out_ptr: MutPtr[dtype],
    in_ptr: MutPtr[dtype],
) raises:
    var strides = IndexList[rank](1)
    comptime for k in reversed(range(rank - 1)):
        strides[k] = strides[k + 1] * shape[k + 1]

    @always_inline
    @__parameter
    @__copy_capture(out_ptr, in_ptr, strides)
    def body[width: Int, alignment: Int = 1](idx: Coord):
        var flat = 0
        comptime for k in range(rank):
            flat += Int(idx[k].value()) * strides[k]
        comptime ba = alignment * size_of[dtype]()
        var a = in_ptr.unsafe_load[width=width, alignment=ba](flat)
        comptime if dtype.is_floating_point():
            out_ptr.unsafe_store[width=width, alignment=ba](flat, a * 2 + 1)
        else:
            out_ptr.unsafe_store[width=width, alignment=ba](flat, a * 3 + 1)

    comptime if unified:
        def ubody[width: Int, alignment: Int = 1](idx: Coord) {
            var out_ptr, var in_ptr, var strides
        }:
            var flat = 0
            comptime for k in range(rank):
                flat += Int(idx[k].value()) * strides[k]
            comptime ba = alignment * size_of[dtype]()
            var a = in_ptr.unsafe_load[width=width, alignment=ba](flat)
            out_ptr.unsafe_store[width=width, alignment=ba](flat, a * 2 + 1)

        comptime if use_cand:
            cand_elementwise[simd_width, target="gpu"](ubody, Coord(shape), ctx)
        else:
            max_elementwise[simd_width, target="gpu"](ubody, Coord(shape), ctx)
    else:
        comptime if use_cand:
            cand_elementwise[body, simd_width=simd_width, target="gpu"](
                Coord(shape), ctx
            )
        else:
            max_elementwise[body, simd_width=simd_width, target="gpu"](
                Coord(shape), ctx
            )


def check[
    dtype: DType, rank: Int, simd_width: Int, unified: Bool
](ctx: DeviceContext, shape: IndexList[rank]) raises -> Int:
    var n = 1
    comptime for k in range(rank):
        n *= shape[k]
    var total = n + 2 * GUARD
    comptime U = DType.uint8 if size_of[dtype]() == 1 else (
        DType.uint16 if size_of[dtype]() == 2 else DType.uint32
    )
    comptime SENT = 0xAB if size_of[dtype]() == 1 else (
        0xABCD if size_of[dtype]() == 2 else 0xABCDEF12
    )
    var hin = ctx.enqueue_create_host_buffer[dtype](total)
    var hsent = ctx.enqueue_create_host_buffer[dtype](total)
    for i in range(total):
        comptime if dtype.is_floating_point():
            hin.unsafe_ptr()[unsafe_offset=i] = Scalar[dtype](
                Float64((i * 7919) % 1000) / 1000.0 - 0.5
            )
        else:
            hin.unsafe_ptr()[unsafe_offset=i] = Scalar[dtype](
                (i * 7919) % 64  # 2a+1, 3a+1 never hit the 0xAB sentinel
            )
        hsent.unsafe_ptr()[unsafe_offset=i] = bitcast[dtype](Scalar[U](SENT))
    var x = ctx.enqueue_create_buffer[dtype](total)
    var ym = ctx.enqueue_create_buffer[dtype](total)
    var yc = ctx.enqueue_create_buffer[dtype](total)
    ctx.enqueue_copy(x, hin)
    ctx.enqueue_copy(ym, hsent)
    ctx.enqueue_copy(yc, hsent)
    run_one[dtype, rank, simd_width, False, unified](
        ctx, shape, _ptr(ym, GUARD), _ptr(x, GUARD)
    )
    run_one[dtype, rank, simd_width, True, unified](
        ctx, shape, _ptr(yc, GUARD), _ptr(x, GUARD)
    )
    var hm = ctx.enqueue_create_host_buffer[dtype](total)
    var hc = ctx.enqueue_create_host_buffer[dtype](total)
    ctx.enqueue_copy(hm, ym)
    ctx.enqueue_copy(hc, yc)
    ctx.synchronize()
    var bad = 0
    for i in range(total):
        var m = bitcast[U](hm.unsafe_ptr()[unsafe_offset=i])
        var c = bitcast[U](hc.unsafe_ptr()[unsafe_offset=i])
        var inside = i >= GUARD and i < GUARD + n
        if m != c or (inside and c == Scalar[U](SENT)) or (
            not inside and c != Scalar[U](SENT)
        ):
            if bad < 3:
                print(
                    "  MISMATCH", String(dtype), "rank", rank, "w", simd_width,
                    "unified" if unified else "capturing", "shape", shape,
                    "flat", i - GUARD, "max", m, "cand", c,
                )
            bad += 1
    return bad


def sweep[dtype: DType, simd_width: Int](ctx: DeviceContext) raises -> Int:
    var bad = 0
    var sizes: List[Int] = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 127, 128, 129,
        255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
        4095, 4096, 4097, 8191, 8192, 8193, 16383, 16385, 65536, 65537,
        131071, 131073, 357 * 789, 1048576 + 3, 3000009, 16777216 + 5,
        # Around the candidate's regime thresholds (tiny = 1 or 2 waves of
        # 114 * 2048 threads, big = 2**20 vectors) for widths 1..16.
        233472, 233473, 466944, 466945, 933888, 933889, 933891, 1867776,
        1867777, 1867783, 3735552, 3735569, 1048575, 1048576, 2097151,
        2097152, 2097153, 4194303, 4194304, 4194305, 8388607, 8388608,
        8388615,
    ]
    for n in sizes:
        bad += check[dtype, 1, simd_width, False](ctx, IndexList[1](n))
    bad += check[dtype, 1, simd_width, True](ctx, IndexList[1](357 * 789))
    var shapes2: List[IndexList[2]] = [
        IndexList[2](0, 5), IndexList[2](1, 1), IndexList[2](7, 13),
        IndexList[2](3, 1024), IndexList[2](1000, 17), IndexList[2](357, 789),
        IndexList[2](4096, 3),
    ]
    for s in shapes2:
        bad += check[dtype, 2, simd_width, False](ctx, s)
    bad += check[dtype, 2, simd_width, True](ctx, IndexList[2](357, 789))
    bad += check[dtype, 3, simd_width, False](ctx, IndexList[3](2, 3, 4097))
    bad += check[dtype, 3, simd_width, False](ctx, IndexList[3](5, 16, 8))
    return bad


def main() raises:
    var ctx = DeviceContext()
    var bad = 0
    bad += sweep[DType.float16, 1](ctx)
    bad += sweep[DType.float16, 4](ctx)
    bad += sweep[DType.float16, 8](ctx)
    bad += sweep[DType.bfloat16, 8](ctx)
    bad += sweep[DType.float32, 1](ctx)
    bad += sweep[DType.float32, 2](ctx)
    bad += sweep[DType.float32, 4](ctx)
    bad += sweep[DType.float16, 2](ctx)
    bad += sweep[DType.float16, 16](ctx)
    bad += sweep[DType.bfloat16, 1](ctx)
    bad += sweep[DType.bfloat16, 4](ctx)
    bad += sweep[DType.float32, 8](ctx)
    bad += sweep[DType.uint8, 1](ctx)
    bad += sweep[DType.uint8, 16](ctx)
    bad += sweep[DType.int8, 4](ctx)
    # Past 2**31 elements: the candidate's 64-bit index path (1-byte dtype
    # keeps it at ~2 GiB per buffer).
    bad += check[DType.uint8, 1, 16, False](ctx, IndexList[1]((1 << 31) + 37))
    bad += check[DType.uint8, 1, 1, True](ctx, IndexList[1]((1 << 31) + 37))
    if bad:
        raise Error("FAILED: " + String(bad) + " bad elements")
    print("ALL PASS")
