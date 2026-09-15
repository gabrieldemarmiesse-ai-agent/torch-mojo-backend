"""Dynamic ROI sampling; backward gives each input pixel a unique writer."""

from std.gpu import block_idx, grid_dim, thread_idx
from std.math import ceil, ceildiv, floor
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
# A portable occupancy cap, not an architecture-fitted tuning constant.
comptime BLOCKS_PER_SM = 8


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
def _weight[dt: DType](pos: Scalar[dt], pixel: Int, size: Int) -> Scalar[dt]:
    if pos < -1 or pos > Scalar[dt](size):
        return 0
    var (lo, hi, frac) = _axis(pos, size)
    var result = Scalar[dt](0)
    if pixel == lo:
        result += 1 - frac
    if pixel == hi:
        result += frac
    return result


@always_inline
def _round_away[dt: DType](value: Scalar[dt]) -> Int:
    return Int(floor(value + 0.5)) if value >= 0 else Int(ceil(value - 0.5))


@__name("roi_align_fwd_" + String(dt))
def _align_forward[
    dt: DType, acc: DType
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
            output[unsafe_offset=index] = 0
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
        output[unsafe_offset=index] = (
            value / Scalar[acc](max(gh * gw, 1))
        ).cast[dt]()
        index += Int(grid_dim.x) * BLOCK


@__name("roi_pool_fwd_" + String(dt))
def _pool_forward[
    dt: DType, acc: DType
](
    input: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    argmax: Pointer[Int32, MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
):
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
            output[unsafe_offset=index] = 0
            argmax[unsafe_offset=index] = -1
            index += Int(grid_dim.x) * BLOCK
            continue
        var batch = Int(rois[unsafe_offset=roi * 5])
        var x0 = _round_away(
            rois[unsafe_offset=roi * 5 + 1].cast[acc]() * scale
        )
        var y0 = _round_away(
            rois[unsafe_offset=roi * 5 + 2].cast[acc]() * scale
        )
        var x1 = _round_away(
            rois[unsafe_offset=roi * 5 + 3].cast[acc]() * scale
        )
        var y1 = _round_away(
            rois[unsafe_offset=roi * 5 + 4].cast[acc]() * scale
        )
        var bh = Scalar[acc](max(y1 - y0 + 1, 1)) / Scalar[acc](ph)
        var bw = Scalar[acc](max(x1 - x0 + 1, 1)) / Scalar[acc](pw)
        var ys = min(max(y0 + Int(floor(Scalar[acc](by) * bh)), 0), h)
        var ye = min(max(y0 + Int(ceil(Scalar[acc](by + 1) * bh)), 0), h)
        var xs = min(max(x0 + Int(floor(Scalar[acc](bx) * bw)), 0), w)
        var xe = min(max(x0 + Int(ceil(Scalar[acc](bx + 1) * bw)), 0), w)
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
        output[unsafe_offset=index] = value
        argmax[unsafe_offset=index] = Int32(winner)
        index += Int(grid_dim.x) * BLOCK


@__name("roi_align_bwd_gather_" + String(dt))
def _align_backward[
    dt: DType, acc: DType
](
    grad: Pointer[Scalar[dt], MutAnyOrigin],
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
):
    var aligned = aligned64 != 0
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    var count = Int(n64) * c * h * w
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while index < count:
        var px = index % w
        var py = index // w % h
        var channel = index // (w * h) % c
        var batch = index // (w * h * c)
        var total = Scalar[acc](0)
        for roi in range(Int(k64)):
            if not _valid_roi(rois, roi, Int(n64), scale):
                continue
            if Int(rois[unsafe_offset=roi * 5]) != batch:
                continue
            var offset = Scalar[acc](0.5) if aligned else Scalar[acc](0)
            var x0 = (
                rois[unsafe_offset=roi * 5 + 1].cast[acc]() * scale - offset
            )
            var y0 = (
                rois[unsafe_offset=roi * 5 + 2].cast[acc]() * scale - offset
            )
            var rw = (
                rois[unsafe_offset=roi * 5 + 3].cast[acc]() * scale
                - offset
                - x0
            )
            var rh = (
                rois[unsafe_offset=roi * 5 + 4].cast[acc]() * scale
                - offset
                - y0
            )
            if not aligned:
                rw = max(rw, Scalar[acc](1))
                rh = max(rh, Scalar[acc](1))
            var bh = rh / Scalar[acc](ph)
            var bw = rw / Scalar[acc](pw)
            var gh = Int(sampling64) if sampling64 > 0 else Int(ceil(bh))
            var gw = Int(sampling64) if sampling64 > 0 else Int(ceil(bw))
            if gh <= 0 or gw <= 0:
                continue
            var by0 = 0
            var by1 = ph
            var bx0 = 0
            var bx1 = pw
            if bh > 0 and bw > 0:
                if (
                    Scalar[acc](py + 1) < y0
                    or Scalar[acc](py - 1) > y0 + rh
                    or Scalar[acc](px + 1) < x0
                    or Scalar[acc](px - 1) > x0 + rw
                ):
                    continue
                by0 = min(
                    max(Int(floor((Scalar[acc](py - 1) - y0) / bh)), 0), ph
                )
                by1 = min(
                    max(Int(ceil((Scalar[acc](py + 1) - y0) / bh)), 0), ph
                )
                bx0 = min(
                    max(Int(floor((Scalar[acc](px - 1) - x0) / bw)), 0), pw
                )
                bx1 = min(
                    max(Int(ceil((Scalar[acc](px + 1) - x0) / bw)), 0), pw
                )
            for by in range(by0, by1):
                for bx in range(bx0, bx1):
                    var g = grad[
                        unsafe_offset=((roi * c + channel) * ph + by) * pw + bx
                    ].cast[acc]() / Scalar[acc](gh * gw)
                    for iy in range(gh):
                        var y = (
                            y0
                            + Scalar[acc](by) * bh
                            + (Scalar[acc](iy) + 0.5) * bh / Scalar[acc](gh)
                        )
                        var wy = _weight(y, py, h)
                        if wy == 0:
                            continue
                        for ix in range(gw):
                            var x = (
                                x0
                                + Scalar[acc](bx) * bw
                                + (Scalar[acc](ix) + 0.5) * bw / Scalar[acc](gw)
                            )
                            total += g * wy * _weight(x, px, w)
        output[unsafe_offset=index] = total.cast[dt]()
        index += Int(grid_dim.x) * BLOCK


@__name("roi_pool_bwd_gather_" + String(dt))
def _pool_backward[
    dt: DType, acc: DType
](
    grad: Pointer[Scalar[dt], MutAnyOrigin],
    rois: Pointer[Scalar[dt], MutAnyOrigin],
    output: Pointer[Scalar[dt], MutAnyOrigin],
    argmax: Pointer[Int32, MutAnyOrigin],
    n64: Int64,
    c64: Int64,
    h64: Int64,
    w64: Int64,
    k64: Int64,
    ph64: Int64,
    pw64: Int64,
    scale: Scalar[acc],
):
    var c = Int(c64)
    var h = Int(h64)
    var w = Int(w64)
    var ph = Int(ph64)
    var pw = Int(pw64)
    var count = Int(n64) * c * h * w
    var index = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while index < count:
        var px = index % w
        var py = index // w % h
        var channel = index // (w * h) % c
        var batch = index // (w * h * c)
        var total = Scalar[acc](0)
        for roi in range(Int(k64)):
            if not _valid_roi(rois, roi, Int(n64), scale):
                continue
            if Int(rois[unsafe_offset=roi * 5]) != batch:
                continue
            var x0 = _round_away(
                rois[unsafe_offset=roi * 5 + 1].cast[acc]() * scale
            )
            var y0 = _round_away(
                rois[unsafe_offset=roi * 5 + 2].cast[acc]() * scale
            )
            var x1 = _round_away(
                rois[unsafe_offset=roi * 5 + 3].cast[acc]() * scale
            )
            var y1 = _round_away(
                rois[unsafe_offset=roi * 5 + 4].cast[acc]() * scale
            )
            var rw = max(x1 - x0 + 1, 1)
            var rh = max(y1 - y0 + 1, 1)
            if px < x0 or px >= x0 + rw or py < y0 or py >= y0 + rh:
                continue
            var bh = Scalar[acc](rh) / Scalar[acc](ph)
            var bw = Scalar[acc](rw) / Scalar[acc](pw)
            var by0 = min(max(Int(floor(Scalar[acc](py - y0) / bh)), 0), ph)
            var by1 = min(max(Int(ceil(Scalar[acc](py - y0 + 1) / bh)), 0), ph)
            var bx0 = min(max(Int(floor(Scalar[acc](px - x0) / bw)), 0), pw)
            var bx1 = min(max(Int(ceil(Scalar[acc](px - x0 + 1) / bw)), 0), pw)
            for by in range(by0, by1):
                for bx in range(bx0, bx1):
                    var gi = ((roi * c + channel) * ph + by) * pw + bx
                    if Int(argmax[unsafe_offset=gi]) == py * w + px:
                        total += grad[unsafe_offset=gi].cast[acc]()
        output[unsafe_offset=index] = total.cast[dt]()
        index += Int(grid_dim.x) * BLOCK


def _launch[
    dt: DType, pool: Bool, backward: Bool
](argv: Argv, argc: Int) raises:
    if argc != 15:
        raise Error("ROI kernel expects 15 argument slots")
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
    var count = Int(n * c * h * w) if backward else Int(k * c * ph * pw)
    var blocks = min(
        ceildiv(count, BLOCK), _device_sm_count(ctx) * BLOCKS_PER_SM
    )
    if count == 0:
        return
    comptime if pool:
        var indices = Pointer[Int32, MutAnyOrigin](
            unsafe_from_address=_raw_int(argv[unsafe_offset=3])
        )
        comptime if backward:
            _enqueue_cached[_pool_backward[dt, acc]](
                ctx,
                "_pool_backward_" + String(dt),
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
                h,
                w,
                k,
                ph,
                pw,
                scale,
            )
        else:
            _enqueue_cached[_pool_forward[dt, acc]](
                ctx,
                "_pool_forward_" + String(dt),
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
                h,
                w,
                k,
                ph,
                pw,
                scale,
            )
    else:
        var sampling = Int64(_raw_int(argv[unsafe_offset=12]))
        var aligned = Int64(_raw_int(argv[unsafe_offset=13]))
        comptime if backward:
            _enqueue_cached[_align_backward[dt, acc]](
                ctx,
                "_align_backward_" + String(dt),
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
        else:
            _enqueue_cached[_align_forward[dt, acc]](
                ctx,
                "_align_forward_" + String(dt),
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
