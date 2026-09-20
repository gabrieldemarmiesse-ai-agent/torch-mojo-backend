"""curand's Philox4x32-10 as curand_kernel.h / curand_uniform.h /
curand_normal.h (CUDA 12.8) compute it, so a `(seed, subsequence, offset)`
triple yields the same words, floats and normals as a CUDA kernel calling
`curand_init` + `curand4` / `curand_uniform4` / `curand_normal4` with them.

After `curand_init(seed, subsequence, offset)` with `offset % 4 == 0` (torch's
generator keeps that invariant): `key = (seed lo, seed hi)`, `ctr =
(lo(offset/4), hi(offset/4), lo(subsequence), hi(subsequence))`, and the k-th
`curand4` call returns `Philox(ctr + k)` with the carry chain x -> y -> z -> w.

Every `a * b + c` below is an `fma`: nvcc contracts them in the CUDA sources
(`--fmad=true`), and the bit-exactness tests against stock CUDA confirm it.
"""
from std.math import fma

from tmb.kernels.common.libdevice_port import (
    nv_fast_sincosf,
    nv_log,
    nv_logf,
    nv_sincospi,
)
from tmb.kernels.common.op_utils import ieee_sqrt

comptime U32x2 = SIMD[DType.uint32, 2]
comptime U32x4 = SIMD[DType.uint32, 4]

comptime _M0 = UInt32(0xD2511F53)
comptime _M1 = UInt32(0xCD9E8D57)
comptime _W0 = UInt32(0x9E3779B9)
comptime _W1 = UInt32(0xBB67AE85)

comptime CURAND_2POW32_INV = Float32(2.3283064e-10)
comptime CURAND_2POW53_INV_DOUBLE = Float64(1.1102230246251565e-16)
# curand_globals.h: `(2.3283064e-10f * 6.2831855f)`, a float32 product.
comptime CURAND_2POW32_INV_2PI = Float32(2.3283064e-10) * Float32(6.2831855)


@always_inline
def philox4x32_10(ctr: U32x4, key: U32x2) -> U32x4:
    var c0 = ctr[0]
    var c1 = ctr[1]
    var c2 = ctr[2]
    var c3 = ctr[3]
    var k0 = key[0]
    var k1 = key[1]

    comptime for _round in range(10):
        var prod0 = _M0.cast[DType.uint64]() * c0.cast[DType.uint64]()
        var prod1 = _M1.cast[DType.uint64]() * c2.cast[DType.uint64]()
        var hi0 = (prod0 >> 32).cast[DType.uint32]()
        var lo0 = prod0.cast[DType.uint32]()
        var hi1 = (prod1 >> 32).cast[DType.uint32]()
        var lo1 = prod1.cast[DType.uint32]()
        var n0 = hi1 ^ c1 ^ k0
        var n2 = hi0 ^ c3 ^ k1
        c0 = n0
        c1 = lo1
        c2 = n2
        c3 = lo0
        k0 += _W0
        k1 += _W1
    return U32x4(c0, c1, c2, c3)


@always_inline
def curand_key(seed: UInt64) -> U32x2:
    return U32x2(seed.cast[DType.uint32](), (seed >> 32).cast[DType.uint32]())


@always_inline
def curand_ctr(offset: UInt64, subsequence: UInt64) -> U32x4:
    var n = offset >> 2
    return U32x4(
        n.cast[DType.uint32](),
        (n >> 32).cast[DType.uint32](),
        subsequence.cast[DType.uint32](),
        (subsequence >> 32).cast[DType.uint32](),
    )


@always_inline
def ctr_add(ctr: U32x4, n: UInt64) -> U32x4:
    """`Philox_State_Incr(state, n)`."""
    var nlo = n.cast[DType.uint32]()
    var nhi = (n >> 32).cast[DType.uint32]()
    var x = ctr[0] + nlo
    if x < nlo:
        nhi += 1
    var y = ctr[1] + nhi
    var z = ctr[2]
    var w = ctr[3]
    if not (nhi <= y):
        z += 1
        if z == 0:
            w += 1
    return U32x4(x, y, z, w)


@always_inline
def curand4(ctr: U32x4, key: U32x2, k: UInt64) -> U32x4:
    """The k-th `curand4` result of a state initialised to `(ctr, key)`."""
    return philox4x32_10(ctr_add(ctr, k), key)


@always_inline
def curand_uniform4(w: U32x4) -> SIMD[DType.float32, 4]:
    """(0, 1] floats: `x * 2^-32 + 2^-33`."""
    return fma(
        w.cast[DType.float32](),
        SIMD[DType.float32, 4](CURAND_2POW32_INV),
        SIMD[DType.float32, 4](CURAND_2POW32_INV / 2),
    )


@always_inline
def _hq_bits(w: U32x4) -> SIMD[DType.uint64, 2]:
    """`_curand_uniform_double_hq`'s 53-bit integers from word pairs."""
    var w64 = w.cast[DType.uint64]()
    return SIMD[DType.uint64, 2](
        w64[0] ^ (w64[1] << 21), w64[2] ^ (w64[3] << 21)
    )


@always_inline
def curand_uniform2_double(w: U32x4) -> SIMD[DType.float64, 2]:
    return fma(
        _hq_bits(w).cast[DType.float64](),
        SIMD[DType.float64, 2](CURAND_2POW53_INV_DOUBLE),
        SIMD[DType.float64, 2](CURAND_2POW53_INV_DOUBLE / 2),
    )


@always_inline
def curand_box_muller(x: UInt32, y: UInt32) -> Tuple[Float32, Float32]:
    var u = fma(
        x.cast[DType.float32](), CURAND_2POW32_INV, CURAND_2POW32_INV / 2
    )
    var v = fma(
        y.cast[DType.float32](),
        CURAND_2POW32_INV_2PI,
        CURAND_2POW32_INV_2PI / 2,
    )
    var s = ieee_sqrt(Float32(-2.0) * nv_logf(u))
    var sc = nv_fast_sincosf(v)
    return (sc[0] * s, sc[1] * s)


@always_inline
def curand_normal4(w: U32x4) -> SIMD[DType.float32, 4]:
    var a = curand_box_muller(w[0], w[1])
    var b = curand_box_muller(w[2], w[3])
    return SIMD[DType.float32, 4](a[0], a[1], b[0], b[1])


@always_inline
def curand_normal2_double(w: U32x4) -> SIMD[DType.float64, 2]:
    var bits = _hq_bits(w)
    var u = fma(
        bits[0].cast[DType.float64](),
        CURAND_2POW53_INV_DOUBLE,
        CURAND_2POW53_INV_DOUBLE / 2,
    )
    var v = fma(
        bits[1].cast[DType.float64](),
        CURAND_2POW53_INV_DOUBLE * 2,
        CURAND_2POW53_INV_DOUBLE,
    )
    var s = ieee_sqrt(Float64(-2.0) * nv_log(u))
    var sc = nv_sincospi(v)
    return SIMD[DType.float64, 2](sc[0] * s, sc[1] * s)
