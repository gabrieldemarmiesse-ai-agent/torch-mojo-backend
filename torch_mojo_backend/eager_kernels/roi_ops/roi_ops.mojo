"""Dynamic ROI sampling with shared geometry and atomic backward scatter."""

from max.gpu.host import DeviceBuffer
from std.atomic import Atomic, Ordering
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.intrinsics import mulhi
from std.sys import is_amd_gpu, is_nvidia_gpu, inlined_assembly
from std.sys.info import has_nvidia_gpu_accelerator
from std.utils.fast_div import FastDiv
from std.math import ceil, ceildiv, floor
from std.memory import bitcast
from op_utils import (
    Argv,
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_f64,
    _raw_int,
)
from variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime BLOCK = 256
# Measured on H100: 32 forward blocks/SM, 8 scatter blocks/SM.
comptime FORWARD_BLOCKS_PER_SM = 32
comptime BACKWARD_BLOCKS_PER_SM = 8


@always_inline
def _valid_roi[
    dt: DType, acc: DType
](
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    roi: Int,
    n: Int,
    scale: Scalar[acc],
) -> Bool:
    var batch = rois[unsafe_offset=roi * 5].cast[acc]()
    if not (batch >= 0 and batch < Scalar[acc](n)):
        return False
    for axis in range(1, 5):
        var coord = rois[unsafe_offset=roi * 5 + axis].cast[acc]() * scale
        # Invalid coordinates must never become pointer offsets.
        if not (abs(coord) < Scalar[acc](1 << 60)):
            return False
    return True


@always_inline
def _axis[
    dt: DType
](var pos: Scalar[dt], size: Int) -> Tuple[Int, Int, Scalar[dt]]:
    pos = max(pos, Scalar[dt](0))
    var low = Int(pos)
    if low >= size - 1:
        return (size - 1, size - 1, Scalar[dt](0))
    return (low, low + 1, pos - Scalar[dt](low))


@always_inline
def _sample[
    dt: DType, acc: DType
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    base: Int,
    y: Scalar[acc],
    x: Scalar[acc],
    h: Int,
    w: Int,
) -> Scalar[acc]:
    if y < -1 or y > Scalar[acc](h) or x < -1 or x > Scalar[acc](w):
        return 0
    var (yl, yh, ly) = _axis(y, h)
    var (xl, xh, lx) = _axis(x, w)
    var hy = 1 - ly
    var hx = 1 - lx
    return (
        hy * hx * input[unsafe_offset=base + yl * w + xl].cast[acc]()
        + hy * lx * input[unsafe_offset=base + yl * w + xh].cast[acc]()
        + ly * hx * input[unsafe_offset=base + yh * w + xl].cast[acc]()
        + ly * lx * input[unsafe_offset=base + yh * w + xh].cast[acc]()
    )


@always_inline
def _round_away[dt: DType](value: Scalar[dt]) -> Int:
    return Int(floor(value + 0.5)) if value >= 0 else Int(ceil(value - 0.5))


@always_inline
def _divisor(divisor: Int) -> SIMD[DType.uint32, 4]:
    # FastDiv is not DevicePassable; pack its pinned multiplier/shift fields.
    var d = FastDiv[DType.uint32](divisor)
    return SIMD[DType.uint32, 4](
        UInt32(d._mprime),
        UInt32(d._sh1),
        UInt32(d._log2_shift) if d._is_pow2 else UInt32(d._sh2),
        0,
    )


@always_inline
def _divide(value: UInt32, divisor: SIMD[DType.uint32, 4]) -> UInt32:
    var high = mulhi(divisor[0], value)
    return (high + ((value - high) >> divisor[1])) >> divisor[2]


@always_inline
def _coordinates[
    fast: Bool
](
    index: Int,
    c: Int,
    ph: Int,
    pw: Int,
    div_pw: SIMD[DType.uint32, 4],
    div_ph: SIMD[DType.uint32, 4],
    div_c: SIMD[DType.uint32, 4],
) -> Tuple[Int, Int, Int, Int]:
    comptime if fast:
        var q0 = _divide(UInt32(index), div_pw)
        var q1 = _divide(q0, div_ph)
        var q2 = _divide(q1, div_c)
        return (
            Int(UInt32(index) - q0 * UInt32(pw)),
            Int(q0 - q1 * UInt32(ph)),
            Int(q1 - q2 * UInt32(c)),
            Int(q2),
        )
    else:
        return (
            index % pw,
            index // pw % ph,
            index // (pw * ph) % c,
            index // (pw * ph * c),
        )


@always_inline
def _pool_bounds[
    dt: DType, acc: DType
](
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    roi: Int,
    n: Int,
    h: Int,
    w: Int,
    ph: Int,
    pw: Int,
    by: Int,
    bx: Int,
    scale: Scalar[acc],
) -> Tuple[Int, Int, Int, Int, Int]:
    if not _valid_roi(rois, roi, n, scale):
        return (-1, 0, 0, 0, 0)
    var batch = Int(rois[unsafe_offset=roi * 5])
    var x0 = _round_away(rois[unsafe_offset=roi * 5 + 1].cast[acc]() * scale)
    var y0 = _round_away(rois[unsafe_offset=roi * 5 + 2].cast[acc]() * scale)
    var x1 = _round_away(rois[unsafe_offset=roi * 5 + 3].cast[acc]() * scale)
    var y1 = _round_away(rois[unsafe_offset=roi * 5 + 4].cast[acc]() * scale)
    var bh = Scalar[acc](max(y1 - y0 + 1, 1)) / Scalar[acc](ph)
    var bw = Scalar[acc](max(x1 - x0 + 1, 1)) / Scalar[acc](pw)
    var ys = min(max(y0 + Int(floor(Scalar[acc](by) * bh)), 0), h)
    var ye = min(max(y0 + Int(ceil(Scalar[acc](by + 1) * bh)), 0), h)
    var xs = min(max(x0 + Int(floor(Scalar[acc](bx) * bw)), 0), w)
    var xe = min(max(x0 + Int(ceil(Scalar[acc](bx + 1) * bw)), 0), w)
    return (batch, ys, ye, xs, xe)


@__name("roi_pool_bin_geometry_" + String(dt))
def _pool_geometry[
    dt: DType, acc: DType
](
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    geometry: Pointer[Int64, MutAnyOrigin],
    n: Int64,
    h: Int64,
    w: Int64,
    k: Int64,
    ph: Int64,
    pw: Int64,
    scale: Scalar[acc],
):
    var index = Int32(block_idx.x) * BLOCK + Int32(thread_idx.x)
    var count = Int32(k * ph * pw)
    while index < count:
        var roi = Int(index // Int32(ph * pw))
        var by = Int(index // Int32(pw) % Int32(ph))
        var bx = Int(index % Int32(pw))
        var bounds = _pool_bounds(
            rois, roi, Int(n), Int(h), Int(w), Int(ph), Int(pw), by, bx, scale
        )
        comptime for field in range(5):
            geometry[unsafe_offset=Int(index) * 5 + field] = Int64(
                bounds[field]
            )
        index += Int32(grid_dim.x) * BLOCK


@__name("roi_align_fwd_" + String(dt) + ("_fastdiv" if fast else "_i64"))
def _align_forward[
    dt: DType, acc: DType, fast: Bool
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
    sampling64: Int64,
    aligned64: Int64,
    div_pw: SIMD[DType.uint32, 4],
    div_ph: SIMD[DType.uint32, 4],
    div_c: SIMD[DType.uint32, 4],
):
    var aligned = aligned64 != 0
    var n = Int(n64)
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    comptime idt = DType.int32 if fast else DType.int64
    var count = Scalar[idt](Int(k64) * c * ph * pw)
    var index = Scalar[idt](block_idx.x) * BLOCK + Scalar[idt](thread_idx.x)
    while index < count:
        var (bx, by, channel, roi) = _coordinates[fast](
            Int(index), c, ph, pw, div_pw, div_ph, div_c
        )
        if not _valid_roi(rois, roi, n, scale):
            output[unsafe_offset=Int(index)] = 0
            index += Scalar[idt](grid_dim.x) * BLOCK
            continue
        var batch = Int(rois[unsafe_offset=roi * 5])
        var offset = Scalar[acc](0.5) if aligned else Scalar[acc](0)
        var x0 = rois[unsafe_offset=roi * 5 + 1].cast[acc]() * scale - offset
        var y0 = rois[unsafe_offset=roi * 5 + 2].cast[acc]() * scale - offset
        var rw = (
            rois[unsafe_offset=roi * 5 + 3].cast[acc]() * scale - offset - x0
        )
        var rh = (
            rois[unsafe_offset=roi * 5 + 4].cast[acc]() * scale - offset - y0
        )
        if not aligned:
            rw = max(rw, Scalar[acc](1))
            rh = max(rh, Scalar[acc](1))
        var bh = rh / Scalar[acc](ph)
        var bw = rw / Scalar[acc](pw)
        var gh = Int(sampling64) if sampling64 > 0 else Int(ceil(bh))
        var gw = Int(sampling64) if sampling64 > 0 else Int(ceil(bw))
        var value = Scalar[acc](0)
        if batch >= 0 and batch < n:
            for iy in range(gh):
                var y = (
                    y0
                    + Scalar[acc](by) * bh
                    + (Scalar[acc](iy) + 0.5) * bh / Scalar[acc](gh)
                )
                for ix in range(gw):
                    var x = (
                        x0
                        + Scalar[acc](bx) * bw
                        + (Scalar[acc](ix) + 0.5) * bw / Scalar[acc](gw)
                    )
                    value += _sample(
                        input, (batch * c + channel) * h * w, y, x, h, w
                    )
        output[unsafe_offset=Int(index)] = (
            value / Scalar[acc](max(gh * gw, 1))
        ).cast[dt]()
        index += Scalar[idt](grid_dim.x) * BLOCK


@__name(
    "roi_pool_fwd_"
    + String(dt)
    + ("_geometry" if precomputed else "_fastdiv" if fast else "_i64")
)
def _pool_forward[
    dt: DType, acc: DType, fast: Bool, precomputed: Bool
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    argmax: Pointer[Int32, MutAnyOrigin],
    geometry: Pointer[Int64, MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
    div_pw: SIMD[DType.uint32, 4],
    div_ph: SIMD[DType.uint32, 4],
    div_c: SIMD[DType.uint32, 4],
):
    var n = Int(n64)
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    comptime idt = DType.int32 if fast else DType.int64
    var count = Scalar[idt](Int(k64) * c * ph * pw)
    var index = Scalar[idt](block_idx.x) * BLOCK + Scalar[idt](thread_idx.x)
    while index < count:
        var (bx, by, channel, roi) = _coordinates[fast](
            Int(index), c, ph, pw, div_pw, div_ph, div_c
        )
        var bounds = (Int(0), Int(0), Int(0), Int(0), Int(0))
        comptime if precomputed:
            var bin = (roi * ph + by) * pw + bx
            bounds = (
                Int(geometry[unsafe_offset=bin * 5]),
                Int(geometry[unsafe_offset=bin * 5 + 1]),
                Int(geometry[unsafe_offset=bin * 5 + 2]),
                Int(geometry[unsafe_offset=bin * 5 + 3]),
                Int(geometry[unsafe_offset=bin * 5 + 4]),
            )
        else:
            bounds = _pool_bounds(rois, roi, n, h, w, ph, pw, by, bx, scale)
        var (batch, ys, ye, xs, xe) = bounds
        var value = Scalar[dt](-3.4028234663852886e38)
        var winner = -1
        if ye <= ys or xe <= xs or batch < 0 or batch >= n:
            value = 0
        else:
            for y in range(ys, ye):
                for x in range(xs, xe):
                    var v = input[
                        unsafe_offset=(batch * c + channel) * h * w + y * w + x
                    ]
                    if v > value:
                        value = v
                        winner = y * w + x
        output[unsafe_offset=Int(index)] = value
        argmax[unsafe_offset=Int(index)] = Int32(winner)
        index += Scalar[idt](grid_dim.x) * BLOCK


@always_inline
def _scope() -> StaticString:
    comptime if is_nvidia_gpu():
        return "device"
    elif is_amd_gpu():
        return "agent"
    else:
        return ""


@always_inline
def _add[dt: DType](ptr: Pointer[Scalar[dt], MutAnyOrigin], value: Scalar[dt]):
    _ = Atomic[dt, scope=_scope()].fetch_add[ordering=Ordering.RELAXED](
        ptr, value
    )


@__name("roi_align_bwd_scatter_" + String(dt))
def _align_scatter[
    dt: DType, acc: DType
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[acc], MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
    sampling64: Int64,
    aligned64: Int64,
):
    var aligned = aligned64 != 0
    var n = Int(n64)
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    var count = Int(k64) * c * ph * pw
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while index < count:
        var bx = index % pw
        var by = index // pw % ph
        var channel = index // (pw * ph) % c
        var roi = index // (pw * ph * c)
        if not _valid_roi(rois, roi, n, scale):
            index += Int(grid_dim.x) * BLOCK
            continue
        var batch = Int(rois[unsafe_offset=roi * 5])
        var offset = Scalar[acc](0.5) if aligned else Scalar[acc](0)
        var x0 = rois[unsafe_offset=roi * 5 + 1].cast[acc]() * scale - offset
        var y0 = rois[unsafe_offset=roi * 5 + 2].cast[acc]() * scale - offset
        var rw = (
            rois[unsafe_offset=roi * 5 + 3].cast[acc]() * scale - offset - x0
        )
        var rh = (
            rois[unsafe_offset=roi * 5 + 4].cast[acc]() * scale - offset - y0
        )
        if not aligned:
            rw = max(rw, Scalar[acc](1))
            rh = max(rh, Scalar[acc](1))
        var bh = rh / Scalar[acc](ph)
        var bw = rw / Scalar[acc](pw)
        var gh = Int(sampling64) if sampling64 > 0 else Int(ceil(bh))
        var gw = Int(sampling64) if sampling64 > 0 else Int(ceil(bw))
        var grad = input[unsafe_offset=index].cast[acc]()
        grad /= Scalar[acc](max(gh * gw, 1))
        if batch >= 0 and batch < n:
            for iy in range(gh):
                var y = (
                    y0
                    + Scalar[acc](by) * bh
                    + (Scalar[acc](iy) + 0.5) * bh / Scalar[acc](gh)
                )
                for ix in range(gw):
                    var x = (
                        x0
                        + Scalar[acc](bx) * bw
                        + (Scalar[acc](ix) + 0.5) * bw / Scalar[acc](gw)
                    )
                    if (
                        y < -1
                        or y > Scalar[acc](h)
                        or x < -1
                        or x > Scalar[acc](w)
                    ):
                        continue
                    var (yl, yh, ly) = _axis(y, h)
                    var (xl, xh, lx) = _axis(x, w)
                    var base = (batch * c + channel) * h * w
                    _add(
                        output.unsafe_offset(base + yl * w + xl),
                        (grad * (1 - ly) * (1 - lx)).cast[acc](),
                    )
                    _add(
                        output.unsafe_offset(base + yl * w + xh),
                        (grad * (1 - ly) * lx).cast[acc](),
                    )
                    _add(
                        output.unsafe_offset(base + yh * w + xl),
                        (grad * ly * (1 - lx)).cast[acc](),
                    )
                    _add(
                        output.unsafe_offset(base + yh * w + xh),
                        (grad * ly * lx).cast[acc](),
                    )
        index += Int(grid_dim.x) * BLOCK


@always_inline
def _pool_add_half[
    dt: DType
](
    ptr: Pointer[Scalar[dt], MutAnyOrigin],
    value: Scalar[dt],
    offset: Int,
    count: Int,
):
    comptime if is_nvidia_gpu():
        var address = UInt64(ptr)
        var lane = Int((address >> 1) & 1)
        # The zero neighbor must also be inside this allocation.
        if (lane == 0 and offset + 1 < count) or (lane == 1 and offset > 0):
            var bits = UInt32(bitcast[DType.uint16](value)) << UInt32(lane * 16)
            _ = inlined_assembly[
                "atom.relaxed.gpu.global.add.noftz.f16x2 $0, [$1], $2;",
                UInt32,
                constraints="=r,l,r,~{memory}",
            ](address & ~UInt64(3), bits)
        else:
            _add(ptr, value)
    else:
        _add(ptr, value)


@__name("roi_pool_bwd_scatter_" + String(dt))
def _pool_scatter[
    dt: DType, acc: DType
](
    grad: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[acc], MutAnyOrigin],
    argmax: Pointer[Int32, MutAnyOrigin],
    n: Int64,
    c: Int64,
    hw: Int64,
    bins: Int64,
    count: Int64,
):
    comptime coord = DType.float32 if acc == DType.float16 else acc
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    while i < Int(count):
        var plane = i // Int(bins)
        var roi = plane // Int(c)
        var channel = plane % Int(c)
        var batch_value = rois[unsafe_offset=roi * 5].cast[coord]()
        var pixel = Int(argmax[unsafe_offset=i])
        if (
            batch_value >= 0
            and batch_value < Scalar[coord](n)
            and pixel >= 0
            and pixel < Int(hw)
        ):
            var offset = (Int(batch_value) * Int(c) + channel) * Int(hw) + pixel
            comptime if acc == DType.float16 and is_nvidia_gpu():
                _pool_add_half(
                    output.unsafe_offset(offset),
                    grad[unsafe_offset=i].cast[acc](),
                    offset,
                    Int(n * c * hw),
                )
            else:
                _add(
                    output.unsafe_offset(offset),
                    grad[unsafe_offset=i].cast[acc](),
                )
        i += Int(grid_dim.x) * 256


def _launch_backward[dt: DType, pool: Bool](argv: Argv, argc: Int) raises:
    if argc != 15:
        raise Error("ROI kernel expects 15 argument slots")
    comptime half_pool = pool and dt == DType.float16 and has_nvidia_gpu_accelerator()
    comptime acc = DType.float16 if half_pool else (
        DType.float64 if dt == DType.float64 else DType.float32
    )
    var input = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var rois = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var output = _make_ptr[acc](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var n = Int64(_raw_int(argv[unsafe_offset=4]))
    var c = Int64(_raw_int(argv[unsafe_offset=5]))
    var h = Int64(_raw_int(argv[unsafe_offset=6]))
    var w = Int64(_raw_int(argv[unsafe_offset=7]))
    var k = Int64(_raw_int(argv[unsafe_offset=8]))
    var ph = Int64(_raw_int(argv[unsafe_offset=9]))
    var pw = Int64(_raw_int(argv[unsafe_offset=10]))
    var ctx = _raw_ctx(argv[unsafe_offset=14])
    var nin = Int(n * c * h * w)
    if nin == 0:
        return
    var buffer = DeviceBuffer[acc](
        ctx, output.unsafe_origin_cast[MutUntrackedOrigin](), nin, owning=False
    )
    ctx.enqueue_memset(buffer, Scalar[acc](0))
    var nout = Int(k * c * ph * pw)
    if nout == 0:
        return
    var blocks = min(
        ceildiv(nout, BLOCK), _device_sm_count(ctx) * BACKWARD_BLOCKS_PER_SM
    )
    comptime if pool:
        var indices = Pointer[Int32, MutAnyOrigin](
            unsafe_from_address=_raw_int(argv[unsafe_offset=3])
        )
        _enqueue_cached[_pool_scatter[dt, acc]](
            ctx,
            "_pool_scatter_" + String(dt),
            blocks,
            1,
            1,
            BLOCK,
            input,
            rois,
            output,
            indices,
            n,
            c,
            h * w,
            ph * pw,
            Int64(nout),
        )
    else:
        var scale = Scalar[acc](_raw_f64(argv[unsafe_offset=11]))
        var sampling = Int64(_raw_int(argv[unsafe_offset=12]))
        var aligned = Int64(_raw_int(argv[unsafe_offset=13]))
        _enqueue_cached[_align_scatter[dt, acc]](
            ctx,
            "_align_scatter_" + String(dt),
            blocks,
            1,
            1,
            BLOCK,
            input,
            rois,
            output,
            n,
            c,
            h,
            w,
            k,
            ph,
            pw,
            scale,
            sampling,
            aligned,
        )
    _ = buffer
    _ = ctx


def _enqueue_forward[
    dt: DType, pool: Bool, fast: Bool
](argv: Argv, blocks: Int, sm: Int) raises:
    comptime acc = DType.float64 if dt == DType.float64 else DType.float32
    var input = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var rois = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var output = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var n = Int64(_raw_int(argv[unsafe_offset=4]))
    var c = Int64(_raw_int(argv[unsafe_offset=5]))
    var h = Int64(_raw_int(argv[unsafe_offset=6]))
    var w = Int64(_raw_int(argv[unsafe_offset=7]))
    var k = Int64(_raw_int(argv[unsafe_offset=8]))
    var ph = Int64(_raw_int(argv[unsafe_offset=9]))
    var pw = Int64(_raw_int(argv[unsafe_offset=10]))
    var scale = Scalar[acc](_raw_f64(argv[unsafe_offset=11]))
    var ctx = _raw_ctx(argv[unsafe_offset=14])
    var div_pw = SIMD[DType.uint32, 4](0)
    var div_ph = SIMD[DType.uint32, 4](0)
    var div_c = SIMD[DType.uint32, 4](0)
    comptime if fast:
        div_pw = _divisor(Int(pw))
        div_ph = _divisor(Int(ph))
        div_c = _divisor(Int(c))
    comptime if pool:
        var indices = Pointer[Int32, MutAnyOrigin](
            unsafe_from_address=_raw_int(argv[unsafe_offset=3])
        )
        # H100 measurements favor shared bin geometry beyond 32 channels and one 32-block/SM wave.
        comptime if fast:
            if c >= 32 and k * c * ph * pw >= Int64(
                sm * BLOCK * FORWARD_BLOCKS_PER_SM
            ):
                var geometry = ctx.enqueue_create_buffer[DType.int64](
                    Int(k * ph * pw * 5)
                )
                var geom = geometry.unsafe_ptr().as_unsafe_any_origin()
                _enqueue_cached[_pool_geometry[dt, acc]](
                    ctx,
                    "roi_pool_geometry_" + String(dt),
                    min(ceildiv(Int(k * ph * pw), BLOCK), sm * 8),
                    1,
                    1,
                    BLOCK,
                    rois,
                    geom,
                    n,
                    h,
                    w,
                    k,
                    ph,
                    pw,
                    scale,
                )
                _enqueue_cached[_pool_forward[dt, acc, fast, True]](
                    ctx,
                    "roi_pool_fwd_geometry_" + String(dt),
                    blocks,
                    1,
                    1,
                    BLOCK,
                    input,
                    rois,
                    output,
                    indices,
                    geom,
                    n,
                    c,
                    h,
                    w,
                    k,
                    ph,
                    pw,
                    scale,
                    div_pw,
                    div_ph,
                    div_c,
                )
                _ = geometry
                _ = ctx
                return
        _enqueue_cached[_pool_forward[dt, acc, fast, False]](
            ctx,
            "roi_pool_fwd_" + String(dt) + "_" + String(fast),
            blocks,
            1,
            1,
            BLOCK,
            input,
            rois,
            output,
            indices,
            indices.unsafe_bitcast[Int64](),
            n,
            c,
            h,
            w,
            k,
            ph,
            pw,
            scale,
            div_pw,
            div_ph,
            div_c,
        )
    else:
        var sampling = Int64(_raw_int(argv[unsafe_offset=12]))
        var aligned = Int64(_raw_int(argv[unsafe_offset=13]))
        _enqueue_cached[_align_forward[dt, acc, fast]](
            ctx,
            "roi_align_fwd_" + String(dt) + "_" + String(fast),
            blocks,
            1,
            1,
            BLOCK,
            input,
            rois,
            output,
            n,
            c,
            h,
            w,
            k,
            ph,
            pw,
            scale,
            sampling,
            aligned,
            div_pw,
            div_ph,
            div_c,
        )
    _ = ctx


def _launch[
    dt: DType, pool: Bool, backward: Bool
](argv: Argv, argc: Int) raises:
    if argc != 15:
        raise Error("ROI kernel expects 15 argument slots")
    comptime if backward:
        _launch_backward[dt, pool](argv, argc)
    else:
        var count = (
            _raw_int(argv[unsafe_offset=8])
            * _raw_int(argv[unsafe_offset=5])
            * _raw_int(argv[unsafe_offset=9])
            * _raw_int(argv[unsafe_offset=10])
        )
        if count == 0:
            return
        var ctx = _raw_ctx(argv[unsafe_offset=14])
        var sm = _device_sm_count(ctx)
        var blocks = min(ceildiv(count, BLOCK), sm * FORWARD_BLOCKS_PER_SM)
        # Reserve the final grid-stride increment as well as every valid index.
        if count <= 2147483647 - blocks * BLOCK:
            _enqueue_forward[dt, pool, True](argv, blocks, sm)
        else:
            _enqueue_forward[dt, pool, False](argv, blocks, sm)
        _ = ctx


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    try:
        comptime for dt in [DType.float16, DType.float32, DType.float64]:
            comptime if _dtype_arg_on[0, dt]():
                comptime if _op_on["RoiAlignForward"]():
                    _launch[dt, False, False](argv, argc)
                    return 0
                elif _op_on["RoiAlignBackward"]():
                    _launch[dt, False, True](argv, argc)
                    return 0
                elif _op_on["RoiPoolForward"]():
                    _launch[dt, True, False](argv, argc)
                    return 0
                elif _op_on["RoiPoolBackward"]():
                    _launch[dt, True, True](argv, argc)
                    return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
