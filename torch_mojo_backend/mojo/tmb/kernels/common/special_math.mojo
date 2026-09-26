# ===----------------------------------------------------------------------=== #
# Scalar ports of ATen's special functions, as stock torch runs them on CUDA.
#
# Every function is the float32 instantiation of the jiterator string of the
# same name in aten/src/ATen/native/cuda/Math.cuh (torch v2.14; Cephes-derived
# polynomials, Faddeeva's erfcx), or of the CUDA kernel lambda where the op
# has no jiterator string (UnarySpecialOpsKernel.cu, UnaryGammaKernels.cu).
# The C++ templates run with T = float for float, half and bfloat16 inputs
# (the jiterator computes reduced floats in float), with each `static const T
# A[] = {double literals}` rounded to float -- which `Float32(c)` on the
# comptime lists below reproduces -- and with the CUDA float libm the device
# code calls (`cuda_math.mojo` / `libdevice_port.mojo`). A few steps compute
# in double in the C++ (a double literal in the expression, `modf` on a
# double); those do here too, and are marked.
#
# nvrtc contracts `a * b + c` into an fma (default -fmad=true), so the Horner
# and Clenshaw recurrences below spell the fma out.
# ===----------------------------------------------------------------------=== #

from std.collections import Array
from std.math import fma

from tmb.kernels.common.cuda_math import (
    _abs,
    _copysign,
    _floor,
    _trunc,
    ieee_sqrtf,
    nv_cosf,
    nv_erfcf,
    nv_lgammaf,
    nv_sinf,
)
from tmb.kernels.common.libdevice_port import (
    nv_exp,
    nv_expf,
    nv_log,
    nv_log1pf,
    nv_logf,
    nv_tan,
)

comptime _INF = Float32(from_bits=UInt32(0x7F800000))
comptime _NAN = Float32(from_bits=UInt32(0x7FC00000))


@always_inline
def _horner[n: Int, //, c: Array[Float64, n]](x: Float32) -> Float32:
    """`result = result * x + A[i]` from 0 over the coefficient list, highest
    degree first (the jiterator `polevl` and the open-coded bessel loops)."""
    var r = Float32(0.0)
    comptime for i in range(n):
        comptime ci = c[i]
        r = fma(r, x, Float32(ci))
    return r


@always_inline
def _chbevl[n: Int, //, c: Array[Float64, n]](x: Float32) -> Float32:
    """Cephes `chbevl`: the Clenshaw sum 0.5 (b0 - b2) of a Chebyshev series."""
    comptime c0 = c[0]
    var b0 = Float32(c0)
    var b1 = Float32(0.0)
    var b2 = Float32(0.0)
    comptime for i in range(1, n):
        b2 = b1
        b1 = b0
        comptime ci = c[i]
        b0 = fma(x, b1, -b2) + Float32(ci)
    return Float32(0.5) * (b0 - b2)


# --------------------------------------------------------------------------- #
# Elementary compositions
# --------------------------------------------------------------------------- #


@always_inline
def sinc_f(a: Float32) -> Float32:
    """`sinc_string`: sin(pi a) / (pi a), 1 at 0."""
    if a == Float32(0.0):
        return Float32(1.0)
    var product = Float32(3.14159265358979323846) * a
    return nv_sinf(product) / product


@always_inline
def entr_f(a: Float32) -> Float32:
    """`entr_string`: -a log a; 0 at 0, -inf below."""
    if a != a:
        return a
    if a > Float32(0.0):
        return -a * nv_logf(a)
    if a == Float32(0.0):
        return Float32(0.0)
    return -_INF


@always_inline
def logit_f(x: Float32, eps: Float32) -> Float32:
    """`logit_kernel_cuda`: log(z / (1 - z)), z = x clamped to [eps, 1 - eps]
    when eps >= 0 (a negative eps is the "None" of the schema)."""
    var z = x
    if not (eps < Float32(0.0)):
        var hi = Float32(1.0) - eps
        z = eps if x < eps else (hi if x > hi else x)
    return nv_logf(z / (Float32(1.0) - z))


@always_inline
def _powf(a: Float32, b: Float32) -> Float32:
    """`pow(float, float)` for zeta's `pow(q, -x)`: exp(b log|a|) in double,
    which rounds to the correctly rounded float almost always (powf is
    within 1 ulp of it), with the sign of an odd integral power."""
    var mag = nv_exp(Float64(b) * nv_log(Float64(_abs(a))))
    var r = Float32(mag)
    if a < Float32(0.0):
        if b != _trunc(b):
            return _NAN
        if _trunc(b * Float32(0.5)) * Float32(2.0) != b:
            r = -r
    return r


# --------------------------------------------------------------------------- #
# Gamma family
# --------------------------------------------------------------------------- #


@always_inline
def digamma_f(x_in: Float32) -> Float32:
    """`digamma_string`."""
    var x = x_in
    if x == Float32(0.0):
        return _copysign(_INF, -x)
    var result = Float32(0.0)
    if x < Float32(0.0):
        if x == _trunc(x):
            return _NAN
        # double: r = modf(double(x)), -pi / tan(pi r).
        comptime PI_F64 = 3.14159265358979323846
        var xd = Float64(x)
        var r = xd - Float64(_trunc(x))
        result = Float32(-PI_F64 / nv_tan(PI_F64 * r))
        x = Float32(1.0) - x
    while x < Float32(10.0):
        result -= Float32(1.0) / x
        x += Float32(1.0)
    if x == Float32(10.0):
        return result + Float32(2.25175258906672110764)
    var y = Float32(0.0)
    if x < Float32(1.0e17):
        comptime A = [
            8.33333333333333333333e-2,
            -2.10927960927960927961e-2,
            7.57575757575757575758e-3,
            -4.16666666666666666667e-3,
            3.96825396825396825397e-3,
            -8.33333333333333333333e-3,
            8.33333333333333333333e-2,
        ]
        var z = Float32(1.0) / (x * x)
        y = z * _horner[A](z)
    return ((nv_logf(x) - (Float32(0.5) / x)) - y) + result


@always_inline
def trigamma_f(x_in: Float32) -> Float32:
    """`trigamma_string`."""
    comptime PI = Float32(3.14159265358979323846)
    var x = x_in
    var sign = Float32(1.0)
    var result = Float32(0.0)
    if x < Float32(0.5):
        sign = Float32(-1.0)
        var sin_pi_x = nv_sinf(PI * x)
        result -= (PI * PI) / (sin_pi_x * sin_pi_x)
        x = Float32(1.0) - x
    for _ in range(6):
        result += Float32(1.0) / (x * x)
        x += Float32(1.0)
    var ixx = Float32(1.0) / (x * x)
    comptime one = Float32(1.0)
    result += (
        one
        + one / (Float32(2.0) * x)
        + ixx
        * (
            one / Float32(6.0)
            - ixx * (one / Float32(30.0) - ixx * (one / Float32(42.0)))
        )
    ) / x
    return sign * result


@always_inline
def zeta_f(x: Float32, q: Float32) -> Float32:
    """`zeta_string`: the Hurwitz zeta function (Cephes, Euler-Maclaurin)."""
    comptime MACHEP = Float32(1.11022302462515654042e-16)
    comptime A = [
        12.0,
        -720.0,
        30240.0,
        -1209600.0,
        47900160.0,
        -1.8924375803183791606e9,
        7.47242496e10,
        -2.950130727918164224e12,
        1.1646782814350067249e14,
        -4.5979787224074726105e15,
        1.8152105401943546773e17,
        -7.1661652561756670113e18,
    ]
    if x == Float32(1.0):
        return _INF
    if x < Float32(1.0):
        return _NAN
    if q <= Float32(0.0):
        if q == _floor(q):
            return _INF
        if x != _floor(x):
            return _NAN
    var s = _powf(q, -x)
    var a = q
    var i = 0
    var b = Float32(0.0)
    while (i < 9) or (a <= Float32(9.0)):
        i += 1
        a += Float32(1.0)
        b = _powf(a, -x)
        s += b
        if (-MACHEP * s < b) and (b < MACHEP * s):
            return s
    var w = a
    s += b * w / (x - Float32(1.0))
    s -= Float32(0.5) * b
    var aa = Float32(1.0)
    var k = Float32(0.0)
    comptime for j in range(12):
        aa *= x + k
        b /= w
        comptime aj = A[j]
        var t = aa * b / Float32(aj)
        s = s + t
        t = _abs(t / s)
        if t < MACHEP:
            return s
        k += Float32(1.0)
        aa *= x + k
        b /= w
        k += Float32(1.0)
    return s


@always_inline
def polygamma_f(x: Float32, n: Int) -> Float32:
    """`polygamma_kernel_cuda`: digamma, trigamma, then
    (-1)^(n+1) n! zeta(n + 1, x) from n = 2 on (`polygamma_string`)."""
    if n == 0:
        return digamma_f(x)
    if n == 1:
        return trigamma_f(x)
    var one = Float32(1.0)
    var sign = one if n % 2 == 1 else -one
    var nf = Float32(n)
    return sign * nv_expf(nv_lgammaf(nf + one)) * zeta_f(Float32(n + 1), x)


# --------------------------------------------------------------------------- #
# Error-function family
# --------------------------------------------------------------------------- #


@always_inline
def _erfcx_y100(y100: Float32) -> Float32:
    """`erfcx_y100`: erfcx on y = 4 / (4 + x), one of 100 Chebyshev fits
    picked by int(100 y); the double literals make it a double polynomial."""
    var k = Int(y100)
    if k < 0 or k > 99:
        # y = 1, i.e. |x| < 4 eps: erfcx is 1 to within 1e-15.
        return Float32(1.0)
    var t = Float64(Float32(2.0) * y100 - Float32(2 * k + 1))
    var base = 7 * k
    var table = materialize[_ERFCX_Y100]()
    var p = table[base + 6]
    comptime for j in range(5, -1, -1):
        p = fma(p, t, table[base + j])
    return Float32(p)


@always_inline
def erfcx_f(x: Float32) -> Float32:
    """`erfcx_string`: exp(x^2) erfc(x)."""
    if x != x:
        return x
    if x >= Float32(0.0):
        if x > Float32(50.0):
            comptime ispi = Float32(0.56418958354775628694807945156)
            if x > Float32(5e7):
                return ispi / x
            var x2 = x * x
            return (
                ispi
                * (x2 * (x2 + Float32(4.5)) + Float32(2.0))
                / (x * (x2 * (x2 + Float32(5.0)) + Float32(3.75)))
            )
        return _erfcx_y100(Float32(400.0) / (Float32(4.0) + x))
    if x < Float32(-26.7):
        return _INF
    if x < Float32(-6.1):
        return Float32(2.0) * nv_expf(x * x)
    return Float32(2.0) * nv_expf(x * x) - _erfcx_y100(
        Float32(400.0) / (Float32(4.0) - x)
    )


@always_inline
def log_ndtr_f(x: Float32) -> Float32:
    """`log_ndtr_string`: log of the standard normal CDF."""
    comptime SQRT1_2 = Float32(0.707106781186547524400844362104849039)
    var t = x * SQRT1_2
    if x < Float32(-1.0):
        return nv_logf(erfcx_f(-t) / Float32(2.0)) - t * t
    return nv_log1pf(-nv_erfcf(t) / Float32(2.0))


@always_inline
def ndtri_f(y0: Float32) -> Float32:
    """`ndtri_string`: the inverse of the standard normal CDF (Cephes)."""
    comptime zero = Float32(0.0)
    comptime one = Float32(1.0)
    if y0 == zero:
        return -_INF
    if y0 == one:
        return _INF
    if y0 < zero or y0 > one:
        return _NAN
    var code = True
    var y = y0
    comptime EXPM2 = Float32(0.13533528323661269189)
    if y > one - EXPM2:
        y = one - y
        code = False
    if y > EXPM2:
        comptime P0 = [
            -5.99633501014107895267e1,
            9.80010754185999661536e1,
            -5.66762857469070293439e1,
            1.39312609387279679503e1,
            -1.23916583867381258016e0,
        ]
        comptime Q0 = [
            1.00000000000000000000e0,
            1.95448858338141759834e0,
            4.67627912898881538453e0,
            8.63602421390890590575e1,
            -2.25462687854119370527e2,
            2.00260212380060660359e2,
            -8.20372256168333339912e1,
            1.59056225126211695515e1,
            -1.18331621121330003142e0,
        ]
        comptime s2pi = Float32(2.50662827463100050242e0)
        y = y - Float32(0.5)
        var y2 = y * y
        var x = y + y * (y2 * _horner[P0](y2) / _horner[Q0](y2))
        return x * s2pi
    var x = ieee_sqrtf(Float32(-2.0) * nv_logf(y))
    var x0 = x - (nv_logf(x) / x)
    var z = one / x
    var x1: Float32
    if x < Float32(8.0):
        comptime P1 = [
            4.05544892305962419923e0,
            3.15251094599893866154e1,
            5.71628192246421288162e1,
            4.40805073893200834700e1,
            1.46849561928858024014e1,
            2.18663306850790267539e0,
            -1.40256079171354495875e-1,
            -3.50424626827848203418e-2,
            -8.57456785154685413611e-4,
        ]
        comptime Q1 = [
            1.00000000000000000000e0,
            1.57799883256466749731e1,
            4.53907635128879210584e1,
            4.13172038254672030440e1,
            1.50425385692907503408e1,
            2.50464946208309415979e0,
            -1.42182922854787788574e-1,
            -3.80806407691578277194e-2,
            -9.33259480895457427372e-4,
        ]
        x1 = z * _horner[P1](z) / _horner[Q1](z)
    else:
        comptime P2 = [
            3.23774891776946035970e0,
            6.91522889068984211695e0,
            3.93881025292474443415e0,
            1.33303460815807542389e0,
            2.01485389549179081538e-1,
            1.23716634817820021358e-2,
            3.01581553508235416007e-4,
            2.65806974686737550832e-6,
            6.23974539184983293730e-9,
        ]
        comptime Q2 = [
            1.00000000000000000000e0,
            6.02427039364742014255e0,
            3.67983563856160859403e0,
            1.37702099489081330271e0,
            2.16236993594496635890e-1,
            1.34204006088543189037e-2,
            3.28014464682127739104e-4,
            2.89247864745380683936e-6,
            6.79019408009981274425e-9,
        ]
        x1 = z * _horner[P2](z) / _horner[Q2](z)
    x = x0 - x1
    return x if not code else -x


# --------------------------------------------------------------------------- #
# Modified Bessel functions of the first kind (i0, i0e, i1, i1e)
# --------------------------------------------------------------------------- #

comptime _I0_A = [
    -4.41534164647933937950e-18,
    3.33079451882223809783e-17,
    -2.43127984654795469359e-16,
    1.71539128555513303061e-15,
    -1.16853328779934516808e-14,
    7.67618549860493561688e-14,
    -4.85644678311192946090e-13,
    2.95505266312963983461e-12,
    -1.72682629144155570723e-11,
    9.67580903537323691224e-11,
    -5.18979560163526290666e-10,
    2.65982372468238665035e-9,
    -1.30002500998624804212e-8,
    6.04699502254191894932e-8,
    -2.67079385394061173391e-7,
    1.11738753912010371815e-6,
    -4.41673835845875056359e-6,
    1.64484480707288970893e-5,
    -5.75419501008210370398e-5,
    1.88502885095841655729e-4,
    -5.76375574538582365885e-4,
    1.63947561694133579842e-3,
    -4.32430999505057594430e-3,
    1.05464603945949983183e-2,
    -2.37374148058994688156e-2,
    4.93052842396707084878e-2,
    -9.49010970480476444210e-2,
    1.71620901522208775349e-1,
    -3.04682672343198398683e-1,
    6.76795274409476084995e-1,
]

comptime _I0_B = [
    -7.23318048787475395456e-18,
    -4.83050448594418207126e-18,
    4.46562142029675999901e-17,
    3.46122286769746109310e-17,
    -2.82762398051658348494e-16,
    -3.42548561967721913462e-16,
    1.77256013305652638360e-15,
    3.81168066935262242075e-15,
    -9.55484669882830764870e-15,
    -4.15056934728722208663e-14,
    1.54008621752140982691e-14,
    3.85277838274214270114e-13,
    7.18012445138366623367e-13,
    -1.79417853150680611778e-12,
    -1.32158118404477131188e-11,
    -3.14991652796324136454e-11,
    1.18891471078464383424e-11,
    4.94060238822496958910e-10,
    3.39623202570838634515e-9,
    2.26666899049817806459e-8,
    2.04891858946906374183e-7,
    2.89137052083475648297e-6,
    6.88975834691682398426e-5,
    3.36911647825569408990e-3,
    8.04490411014108831608e-1,
]

comptime _I1_A = [
    2.77791411276104639959e-18,
    -2.11142121435816608115e-17,
    1.55363195773620046921e-16,
    -1.10559694773538630805e-15,
    7.60068429473540693410e-15,
    -5.04218550472791168711e-14,
    3.22379336594557470981e-13,
    -1.98397439776494371520e-12,
    1.17361862988909016308e-11,
    -6.66348972350202774223e-11,
    3.62559028155211703701e-10,
    -1.88724975172282928790e-9,
    9.38153738649577178388e-9,
    -4.44505912879632808065e-8,
    2.00329475355213526229e-7,
    -8.56872026469545474066e-7,
    3.47025130813767847674e-6,
    -1.32731636560394358279e-5,
    4.78156510755005422638e-5,
    -1.61760815825896745588e-4,
    5.12285956168575772895e-4,
    -1.51357245063125314899e-3,
    4.15642294431288815669e-3,
    -1.05640848946261981558e-2,
    2.47264490306265168283e-2,
    -5.29459812080949914269e-2,
    1.02643658689847095384e-1,
    -1.76416518357834055153e-1,
    2.52587186443633654823e-1,
]

comptime _I1_B = [
    7.51729631084210481353e-18,
    4.41434832307170791151e-18,
    -4.65030536848935832153e-17,
    -3.20952592199342395980e-17,
    2.96262899764595013876e-16,
    3.30820231092092828324e-16,
    -1.88035477551078244854e-15,
    -3.81440307243700780478e-15,
    1.04202769841288027642e-14,
    4.27244001671195135429e-14,
    -2.10154184277266431302e-14,
    -4.08355111109219731823e-13,
    -7.19855177624590851209e-13,
    2.03562854414708950722e-12,
    1.41258074366137813316e-11,
    3.25260358301548823856e-11,
    -1.89749581235054123450e-11,
    -5.58974346219658380687e-10,
    -3.83538038596423702205e-9,
    -2.63146884688951950684e-8,
    -2.51223623787020892529e-7,
    -3.88256480887769039346e-6,
    -1.10588938762623716291e-4,
    -9.76109749136146840777e-3,
    7.78576235018280120474e-1,
]

# The float specialization of `i1e_string` keeps only the tail of each series.
comptime _I1E_A_F32 = [
    9.38153738649577178388e-9,
    -4.44505912879632808065e-8,
    2.00329475355213526229e-7,
    -8.56872026469545474066e-7,
    3.47025130813767847674e-6,
    -1.32731636560394358279e-5,
    4.78156510755005422638e-5,
    -1.61760815825896745588e-4,
    5.12285956168575772895e-4,
    -1.51357245063125314899e-3,
    4.15642294431288815669e-3,
    -1.05640848946261981558e-2,
    2.47264490306265168283e-2,
    -5.29459812080949914269e-2,
    1.02643658689847095384e-1,
    -1.76416518357834055153e-1,
    2.52587186443633654823e-1,
]

comptime _I1E_B_F32 = [
    -3.83538038596423702205e-9,
    -2.63146884688951950684e-8,
    -2.51223623787020892529e-7,
    -3.88256480887769039346e-6,
    -1.10588938762623716291e-4,
    -9.76109749136146840777e-3,
    7.78576235018280120474e-1,
]


@always_inline
def i0_f(a: Float32) -> Float32:
    """`i0_string` (also `modified_bessel_i0_string`, the same series)."""
    var x = _abs(a)
    if x <= Float32(8.0):
        var y = (x / Float32(2.0)) - Float32(2.0)
        return nv_expf(x) * _chbevl[_I0_A](y)
    return (
        nv_expf(x) * _chbevl[_I0_B](Float32(32.0) / x - Float32(2.0))
    ) / ieee_sqrtf(x)


@always_inline
def i0e_f(a: Float32) -> Float32:
    """`i0e_string` (Math.h `calc_i0e`)."""
    var x = _abs(a)
    if x <= Float32(8.0):
        return _chbevl[_I0_A]((x / Float32(2.0)) - Float32(2.0))
    return _chbevl[_I0_B](Float32(32.0) / x - Float32(2.0)) / ieee_sqrtf(x)


@always_inline
def i1_f(a: Float32) -> Float32:
    """`i1_string` (also `modified_bessel_i1_string`)."""
    var x = _abs(a)
    var res: Float32
    if x <= Float32(8.0):
        var y = x / Float32(2.0) - Float32(2.0)
        res = nv_expf(x) * x * _chbevl[_I1_A](y)
    else:
        res = (
            nv_expf(x) * _chbevl[_I1_B](Float32(32.0) / x - Float32(2.0))
        ) / ieee_sqrtf(x)
    return -res if a < Float32(0.0) else res


@always_inline
def i1e_f(a: Float32) -> Float32:
    """`i1e_string`, float specialization."""
    var x = _abs(a)
    var res: Float32
    if x <= Float32(8.0):
        var y = x / Float32(2.0) - Float32(2.0)
        res = _chbevl[_I1E_A_F32](y) * x
    else:
        res = _chbevl[_I1E_B_F32](
            Float32(32.0) / x - Float32(2.0)
        ) / ieee_sqrtf(x)
    return -res if a < Float32(0.0) else res


@always_inline
def modified_bessel_i0_f(x: Float32) -> Float32:
    """`modified_bessel_i0_string`: i0's series, with the scaling written
    exp(|x|) * (0.5 (a - p)) [/ sqrt(|x|)]."""
    var ax = _abs(x)
    if ax <= Float32(8.0):
        return nv_expf(ax) * _chbevl[_I0_A]((ax / Float32(2.0)) - Float32(2.0))
    return (
        nv_expf(ax) * _chbevl[_I0_B](Float32(32.0) / ax - Float32(2.0))
    ) / ieee_sqrtf(ax)


@always_inline
def modified_bessel_i1_f(x: Float32) -> Float32:
    """`modified_bessel_i1_string`."""
    var ax = _abs(x)
    var res: Float32
    if ax <= Float32(8.0):
        var s = _chbevl[_I1_A]((ax / Float32(2.0)) - Float32(2.0))
        res = s * ax * nv_expf(ax)
    else:
        res = (
            nv_expf(ax) * _chbevl[_I1_B](Float32(32.0) / ax - Float32(2.0))
        ) / ieee_sqrtf(ax)
    return -res if x < Float32(0.0) else res


# --------------------------------------------------------------------------- #
# Modified Bessel functions of the second kind (k0, k1, and scaled)
# --------------------------------------------------------------------------- #

comptime _K0_A = [
    1.37446543561352307156e-16,
    4.25981614279661018399e-14,
    1.03496952576338420167e-11,
    1.90451637722020886025e-09,
    2.53479107902614945675e-07,
    2.28621210311945178607e-05,
    1.26461541144692592338e-03,
    3.59799365153615016266e-02,
    3.44289899924628486886e-01,
    -5.35327393233902768720e-01,
]

comptime _K0_B = [
    5.30043377268626276149e-18,
    -1.64758043015242134646e-17,
    5.21039150503902756861e-17,
    -1.67823109680541210385e-16,
    5.51205597852431940784e-16,
    -1.84859337734377901440e-15,
    6.34007647740507060557e-15,
    -2.22751332699166985548e-14,
    8.03289077536357521100e-14,
    -2.98009692317273043925e-13,
    1.14034058820847496303e-12,
    -4.51459788337394416547e-12,
    1.85594911495471785253e-11,
    -7.95748924447710747776e-11,
    3.57739728140030116597e-10,
    -1.69753450938905987466e-09,
    8.57403401741422608519e-09,
    -4.66048989768794782956e-08,
    2.76681363944501510342e-07,
    -1.83175552271911948767e-06,
    1.39498137188764993662e-05,
    -1.28495495816278026384e-04,
    1.56988388573005337491e-03,
    -3.14481013119645005427e-02,
    2.44030308206595545468e00,
]

comptime _K1_A = [
    -7.02386347938628759343e-18,
    -2.42744985051936593393e-15,
    -6.66690169419932900609e-13,
    -1.41148839263352776110e-10,
    -2.21338763073472585583e-08,
    -2.43340614156596823496e-06,
    -1.73028895751305206302e-04,
    -6.97572385963986435018e-03,
    -1.22611180822657148235e-01,
    -3.53155960776544875667e-01,
    1.52530022733894777053e00,
]

comptime _K1_B = [
    -5.75674448366501715755e-18,
    1.79405087314755922667e-17,
    -5.68946255844285935196e-17,
    1.83809354436663880070e-16,
    -6.05704724837331885336e-16,
    2.03870316562433424052e-15,
    -7.01983709041831346144e-15,
    2.47715442448130437068e-14,
    -8.97670518232499435011e-14,
    3.34841966607842919884e-13,
    -1.28917396095102890680e-12,
    5.13963967348173025100e-12,
    -2.12996783842756842877e-11,
    9.21831518760500529508e-11,
    -4.19035475934189648750e-10,
    2.01504975519703286596e-09,
    -1.03457624656780970260e-08,
    5.74108412545004946722e-08,
    -3.50196060308781257119e-07,
    2.40648494783721712015e-06,
    -1.93619797416608296024e-05,
    1.95215518471351631108e-04,
    -2.85781685962277938680e-03,
    1.03923736576817238437e-01,
    2.72062619048444266945e00,
]


@always_inline
def modified_bessel_k0_f(x: Float32, scaled: Bool) -> Float32:
    """`modified_bessel_k0_string` / `scaled_modified_bessel_k0_string`."""
    if x == Float32(0.0):
        return _INF
    if x < Float32(0.0):
        return _NAN
    if x <= Float32(2.0):
        var s = _chbevl[_K0_A](x * x - Float32(2.0))
        var i0 = modified_bessel_i0_f(x)
        if scaled:
            return (s - nv_logf(Float32(0.5) * x) * i0) * nv_expf(x)
        # double: `log(0.5 * x)` has a double literal, so the whole
        # expression is evaluated in double and rounded once.
        var lg = nv_log(0.5 * Float64(x))
        return Float32(Float64(s) - lg * Float64(i0))
    var s = _chbevl[_K0_B](Float32(8.0) / x - Float32(2.0))
    if scaled:
        return s / ieee_sqrtf(x)
    return nv_expf(-x) * s / ieee_sqrtf(x)


@always_inline
def modified_bessel_k1_f(x: Float32, scaled: Bool) -> Float32:
    """`modified_bessel_k1_string` / `scaled_modified_bessel_k1_string`."""
    if x == Float32(0.0):
        return _INF
    if x < Float32(0.0):
        return _NAN
    if x <= Float32(2.0):
        var s = _chbevl[_K1_A](x * x - Float32(2.0))
        var r = nv_logf(Float32(0.5) * x) * modified_bessel_i1_f(x) + s / x
        if scaled:
            return r * nv_expf(x)
        return r
    var s = _chbevl[_K1_B](Float32(8.0) / x - Float32(2.0))
    if scaled:
        return s / ieee_sqrtf(x)
    return nv_expf(-x) * s / ieee_sqrtf(x)


# --------------------------------------------------------------------------- #
# Bessel functions of the first and second kind (j0, j1, y0, y1)
# --------------------------------------------------------------------------- #

comptime _J0_PP = [
    7.96936729297347051624e-04,
    8.28352392107440799803e-02,
    1.23953371646414299388e00,
    5.44725003058768775090e00,
    8.74716500199817011941e00,
    5.30324038235394892183e00,
    9.99999999999999997821e-01,
]
comptime _J0_PQ = [
    9.24408810558863637013e-04,
    8.56288474354474431428e-02,
    1.25352743901058953537e00,
    5.47097740330417105182e00,
    8.76190883237069594232e00,
    5.30605288235394617618e00,
    1.00000000000000000218e00,
]
comptime _J0_QP = [
    -1.13663838898469149931e-02,
    -1.28252718670509318512e00,
    -1.95539544257735972385e01,
    -9.32060152123768231369e01,
    -1.77681167980488050595e02,
    -1.47077505154951170175e02,
    -5.14105326766599330220e01,
    -6.05014350600728481186e00,
]
comptime _J0_QQ = [
    6.43178256118178023184e01,
    8.56430025976980587198e02,
    3.88240183605401609683e03,
    7.24046774195652478189e03,
    5.93072701187316984827e03,
    2.06209331660327847417e03,
    2.42005740240291393179e02,
]
comptime _J0_RP = [
    -4.79443220978201773821e09,
    1.95617491946556577543e12,
    -2.49248344360967716204e14,
    9.70862251047306323952e15,
]
comptime _J0_RQ = [
    4.99563147152651017219e02,
    1.73785401676374683123e05,
    4.84409658339962045305e07,
    1.11855537045356834862e10,
    2.11277520115489217587e12,
    3.10518229857422583814e14,
    3.18121955943204943306e16,
    1.71086294081043136091e18,
]
comptime _Y0_YP = [
    1.55924367855235737965e04,
    -1.46639295903971606143e07,
    5.43526477051876500413e09,
    -9.82136065717911466409e11,
    8.75906394395366999549e13,
    -3.46628303384729719441e15,
    4.42733268572569800351e16,
    -1.84950800436986690637e16,
]
comptime _Y0_YQ = [
    1.04128353664259848412e03,
    6.26107330137134956842e05,
    2.68919633393814121987e08,
    8.64002487103935000337e10,
    2.02979612750105546709e13,
    3.17157752842975028269e15,
    2.50596256172653059228e17,
]
comptime _J1_PP = [
    7.62125616208173112003e-04,
    7.31397056940917570436e-02,
    1.12719608129684925192e00,
    5.11207951146807644818e00,
    8.42404590141772420927e00,
    5.21451598682361504063e00,
    1.00000000000000000254e00,
]
comptime _J1_PQ = [
    5.71323128072548699714e-04,
    6.88455908754495404082e-02,
    1.10514232634061696926e00,
    5.07386386128601488557e00,
    8.39985554327604159757e00,
    5.20982848682361821619e00,
    9.99999999999999997461e-01,
]
comptime _J1_QP = [
    5.10862594750176621635e-02,
    4.98213872951233449420e00,
    7.58238284132545283818e01,
    3.66779609360150777800e02,
    7.10856304998926107277e02,
    5.97489612400613639965e02,
    2.11688757100572135698e02,
    2.52070205858023719784e01,
]
comptime _J1_QQ = [
    7.42373277035675149943e01,
    1.05644886038262816351e03,
    4.98641058337653607651e03,
    9.56231892404756170795e03,
    7.99704160447350683650e03,
    2.82619278517639096600e03,
    3.36093607810698293419e02,
]
comptime _J1_RP = [
    -8.99971225705559398224e08,
    4.52228297998194034323e11,
    -7.27494245221818276015e13,
    3.68295732863852883286e15,
]
comptime _J1_RQ = [
    6.20836478118054335476e02,
    2.56987256757748830383e05,
    8.35146791431949253037e07,
    2.21511595479792499675e10,
    4.74914122079991414898e12,
    7.84369607876235854894e14,
    8.95222336184627338078e16,
    5.32278620332680085395e18,
]
comptime _Y1_YP = [
    1.26320474790178026440e09,
    -6.47355876379160291031e11,
    1.14509511541823727583e14,
    -8.12770255501325109621e15,
    2.02439475713594898196e17,
    -7.78877196265950026825e17,
]
comptime _Y1_YQ = [
    5.94301592346128195359e02,
    2.35564092943068577943e05,
    7.34811944459721705660e07,
    1.87601316108706159478e10,
    3.88231277496238566008e12,
    6.20557727146953693363e14,
    6.87141087355300489866e16,
    3.97270608116560655612e18,
]

comptime _PIO4 = Float32(0.785398163397448309615660845819875721)
comptime _3PIO4 = Float32(2.356194490192344928846982537459627163)
comptime _SQ2OPI = Float32(0.797884560802865355879892119868763737)
comptime _TWOOPI = Float32(0.636619772367581343075535053490057448)


@always_inline
def bessel_j0_f(x_in: Float32) -> Float32:
    """`bessel_j0_string`."""
    var x = _abs(x_in) if x_in < Float32(0.0) else x_in
    if x <= Float32(5.0):
        if x < Float32(0.00001):
            return Float32(1.0) - x * x / Float32(4.0)
        var xx = x * x
        var rp = _horner[_J0_RP](xx)
        var rq = _horner[_J0_RQ](xx)
        return (
            (x * x - Float32(5.78318596294678452118e00))
            * (x * x - Float32(3.04712623436620863991e01))
            * rp
            / rq
        )
    var z = Float32(25.0) / (x * x)
    var pp = _horner[_J0_PP](z)
    var pq = _horner[_J0_PQ](z)
    var qp = _horner[_J0_QP](z)
    var qq = _horner[_J0_QQ](z)
    return (
        (
            pp / pq * nv_cosf(x - _PIO4)
            - Float32(5.0) / x * (qp / qq) * nv_sinf(x - _PIO4)
        )
        * _SQ2OPI
        / ieee_sqrtf(x)
    )


@always_inline
def bessel_y0_f(x: Float32) -> Float32:
    """`bessel_y0_string`."""
    if x <= Float32(5.0):
        if x == Float32(0.0):
            return -_INF
        var xx = x * x
        var yp = _horner[_Y0_YP](xx)
        var yq = _horner[_Y0_YQ](xx)
        return yp / yq + (_TWOOPI * nv_logf(x) * bessel_j0_f(x))
    var z = Float32(25.0) / (x * x)
    var pp = _horner[_J0_PP](z)
    var pq = _horner[_J0_PQ](z)
    var qp = _horner[_J0_QP](z)
    var qq = _horner[_J0_QQ](z)
    return (
        (
            pp / pq * nv_sinf(x - _PIO4)
            + Float32(5.0) / x * (qp / qq) * nv_cosf(x - _PIO4)
        )
        * _SQ2OPI
        / ieee_sqrtf(x)
    )


@always_inline
def _bessel_j1_pos(x: Float32) -> Float32:
    if x <= Float32(5.0):
        var xx = x * x
        var rp = _horner[_J1_RP](xx)
        var rq = _horner[_J1_RQ](xx)
        return (
            rp
            / rq
            * x
            * (x * x - Float32(1.46819706421238932572e01))
            * (x * x - Float32(4.92184563216946036703e01))
        )
    var z = Float32(5.0) / x * (Float32(5.0) / x)
    var pp = _horner[_J1_PP](z)
    var pq = _horner[_J1_PQ](z)
    var qp = _horner[_J1_QP](z)
    var qq = _horner[_J1_QQ](z)
    return (
        (
            pp / pq * nv_cosf(x - _3PIO4)
            - Float32(5.0) / x * (qp / qq) * nv_sinf(x - _3PIO4)
        )
        * _SQ2OPI
        / ieee_sqrtf(x)
    )


@always_inline
def bessel_j1_f(x: Float32) -> Float32:
    """`bessel_j1_string` (odd: j1(-x) = -j1(x))."""
    if x < Float32(0.0):
        return -_bessel_j1_pos(-x)
    return _bessel_j1_pos(x)


@always_inline
def bessel_y1_f(x: Float32) -> Float32:
    """`bessel_y1_string`."""
    if x <= Float32(5.0):
        if x == Float32(0.0):
            return -_INF
        if x <= Float32(0.0):
            return _NAN
        var xx = x * x
        var yp = _horner[_Y1_YP](xx)
        var yq = _horner[_Y1_YQ](xx)
        return x * (yp / yq) + (
            _TWOOPI * (bessel_j1_f(x) * nv_logf(x) - Float32(1.0) / x)
        )
    var z = Float32(5.0) / x * (Float32(5.0) / x)
    var pp = _horner[_J1_PP](z)
    var pq = _horner[_J1_PQ](z)
    var qp = _horner[_J1_QP](z)
    var qq = _horner[_J1_QQ](z)
    return (
        (
            pp / pq * nv_sinf(x - _3PIO4)
            + Float32(5.0) / x * (qp / qq) * nv_cosf(x - _3PIO4)
        )
        * _SQ2OPI
        / ieee_sqrtf(x)
    )


@always_inline
def spherical_bessel_j0_f(x: Float32) -> Float32:
    """`spherical_bessel_j0_string`: sin(x) / x."""
    if _abs(x) == _INF:
        return Float32(0.0)
    if _abs(x) < Float32(0.5):
        var x2 = x * x
        return Float32(1.0) + x2 * (
            Float32(-1.0) / Float32(6.0)
            + x2
            * (
                Float32(1.0) / Float32(120.0)
                + x2
                * (
                    Float32(-1.0) / Float32(5040.0)
                    + x2
                    * (
                        Float32(1.0) / Float32(362880.0)
                        + x2
                        * (
                            Float32(-1.0) / Float32(39916800.0)
                            + x2 * (Float32(1.0) / Float32(6227020800.0))
                        )
                    )
                )
            )
        )
    return nv_sinf(x) / x


# --------------------------------------------------------------------------- #
# Airy Ai
# --------------------------------------------------------------------------- #

comptime _AI_AN = [
    3.46538101525629032477e-01,
    1.20075952739645805542e01,
    7.62796053615234516538e01,
    1.68089224934630576269e02,
    1.59756391350164413639e02,
    7.05360906840444183113e01,
    1.40264691163389668864e01,
    9.99999999999999995305e-01,
]
comptime _AI_AD = [
    5.67594532638770212846e-01,
    1.47562562584847203173e01,
    8.45138970141474626562e01,
    1.77318088145400459522e02,
    1.64234692871529701831e02,
    7.14778400825575695274e01,
    1.40959135607834029598e01,
    1.00000000000000000470e00,
]
comptime _AI_AFN = [
    -1.31696323418331795333e-01,
    -6.26456544431912369773e-01,
    -6.93158036036933542233e-01,
    -2.79779981545119124951e-01,
    -4.91900132609500318020e-02,
    -4.06265923594885404393e-03,
    -1.59276496239262096340e-04,
    -2.77649108155232920844e-06,
    -1.67787698489114633780e-08,
]
comptime _AI_AFD = [
    1.33560420706553243746e01,
    3.26825032795224613948e01,
    2.67367040941499554804e01,
    9.18707402907259625840e00,
    1.47529146771666414581e00,
    1.15687173795188044134e-01,
    4.40291641615211203805e-03,
    7.54720348287414296618e-05,
    4.51850092970580378464e-07,
]
comptime _AI_AGN = [
    1.97339932091685679179e-02,
    3.91103029615688277255e-01,
    1.06579897599595591108e00,
    9.39169229816650230044e-01,
    3.51465656105547619242e-01,
    6.33888919628925490927e-02,
    5.85804113048388458567e-03,
    2.82851600836737019778e-04,
    6.98793669997260967291e-06,
    8.11789239554389293311e-08,
    3.41551784765923618484e-10,
]
comptime _AI_AGD = [
    9.30892908077441974853e00,
    1.98352928718312140417e01,
    1.55646628932864612953e01,
    5.47686069422975497931e00,
    9.54293611618961883998e-01,
    8.64580826352392193095e-02,
    4.12656523824222607191e-03,
    1.01259085116509135510e-04,
    1.17166733214413521882e-06,
    4.91834570062930015649e-09,
]


@always_inline
def airy_ai_f(x: Float32) -> Float32:
    """`airy_ai_string` (Cephes airy, Ai only)."""
    if _abs(x) == _INF:
        return _NAN
    if x > Float32(103.892):
        return Float32(0.0)
    comptime isqpi = Float32(5.64189583547756286948e-01)
    if x < Float32(-2.09):
        var zeta = Float32(-2.0) * x * ieee_sqrtf(-x) / Float32(3.0)
        var z = Float32(1.0) / zeta
        var zz = z * z
        var afn = _horner[_AI_AFN](zz)
        var afd = _horner[_AI_AFD](zz)
        var agn = _horner[_AI_AGN](zz)
        var agd = _horner[_AI_AGD](zz)
        var t = zeta + Float32(0.25) * Float32(3.14159265358979323846)
        return (
            isqpi
            / ieee_sqrtf(ieee_sqrtf(-x))
            * (
                nv_sinf(t) * (Float32(1.0) + z * z * afn / afd)
                - nv_cosf(t) * (z * agn / agd)
            )
        )
    var odd_domain = False
    var ai = Float32(0.0)
    if x >= Float32(2.09):
        odd_domain = True
        var zeta = Float32(2.0) * x * ieee_sqrtf(x) / Float32(3.0)
        var rz = Float32(1.0) / zeta
        var an = _horner[_AI_AN](rz)
        var ad = _horner[_AI_AD](rz)
        ai = (
            isqpi
            * (an / ad)
            / (Float32(2.0) * ieee_sqrtf(ieee_sqrtf(x)) * nv_expf(zeta))
        )
        if x > Float32(8.3203353):
            return ai
    # Maclaurin series of Ai = c1 f - c2 g.
    var f = Float32(1.0)
    var g = x
    var k = Float32(1.0)
    var m = Float32(1.0)
    var n = x
    var t = Float32(1.0)
    var z = x * x * x
    while t > Float32(1.11022302462515654042e-16):
        m *= z
        k += Float32(1.0)
        m /= k
        n *= z
        k += Float32(1.0)
        n /= k
        m /= k
        f += m
        k += Float32(1.0)
        n /= k
        g += n
        t = _abs(m / f)
    if not odd_domain:
        return (
            Float32(0.355028053887817239260) * f
            - Float32(0.258819403792806798405) * g
        )
    return ai


# --------------------------------------------------------------------------- #
# The erfcx lookup table
# --------------------------------------------------------------------------- #

# 100 Chebyshev fits of erfcx on y100 intervals [k, k+1), 7 coefficients
# each, lowest degree first (cuda/Math.cuh `erfcx_y100`, from the MIT
# Faddeeva package). Evaluated in float64, as the C++ literals are.
comptime _ERFCX_Y100: Array[Float64, 700] = [
    0.70878032454106438663e-3,
    0.71234091047026302958e-3,
    0.35779077297597742384e-5,
    0.17403143962587937815e-7,
    0.81710660047307788845e-10,
    0.36885022360434957634e-12,
    0.15917038551111111111e-14,
    0.21479143208285144230e-2,
    0.72686402367379996033e-3,
    0.36843175430938995552e-5,
    0.18071841272149201685e-7,
    0.85496449296040325555e-10,
    0.38852037518534291510e-12,
    0.16868473576888888889e-14,
    0.36165255935630175090e-2,
    0.74182092323555510862e-3,
    0.37948319957528242260e-5,
    0.18771627021793087350e-7,
    0.89484715122415089123e-10,
    0.40935858517772440862e-12,
    0.17872061464888888889e-14,
    0.51154983860031979264e-2,
    0.75722840734791660540e-3,
    0.39096425726735703941e-5,
    0.19504168704300468210e-7,
    0.93687503063178993915e-10,
    0.43143925959079664747e-12,
    0.18939926435555555556e-14,
    0.66457513172673049824e-2,
    0.77310406054447454920e-3,
    0.40289510589399439385e-5,
    0.20271233238288381092e-7,
    0.98117631321709100264e-10,
    0.45484207406017752971e-12,
    0.20076352213333333333e-14,
    0.82082389970241207883e-2,
    0.78946629611881710721e-3,
    0.41529701552622656574e-5,
    0.21074693344544655714e-7,
    0.10278874108587317989e-9,
    0.47965201390613339638e-12,
    0.21285907413333333333e-14,
    0.98039537275352193165e-2,
    0.80633440108342840956e-3,
    0.42819241329736982942e-5,
    0.21916534346907168612e-7,
    0.10771535136565470914e-9,
    0.50595972623692822410e-12,
    0.22573462684444444444e-14,
    0.11433927298290302370e-1,
    0.82372858383196561209e-3,
    0.44160495311765438816e-5,
    0.22798861426211986056e-7,
    0.11291291745879239736e-9,
    0.53386189365816880454e-12,
    0.23944209546666666667e-14,
    0.13099232878814653979e-1,
    0.84167002467906968214e-3,
    0.45555958988457506002e-5,
    0.23723907357214175198e-7,
    0.11839789326602695603e-9,
    0.56346163067550237877e-12,
    0.25403679644444444444e-14,
    0.14800987015587535621e-1,
    0.86018092946345943214e-3,
    0.47008265848816866105e-5,
    0.24694040760197315333e-7,
    0.12418779768752299093e-9,
    0.59486890370320261949e-12,
    0.26957764568888888889e-14,
    0.16540351739394069380e-1,
    0.87928458641241463952e-3,
    0.48520195793001753903e-5,
    0.25711774900881709176e-7,
    0.13030128534230822419e-9,
    0.62820097586874779402e-12,
    0.28612737351111111111e-14,
    0.18318536789842392647e-1,
    0.89900542647891721692e-3,
    0.50094684089553365810e-5,
    0.26779777074218070482e-7,
    0.13675822186304615566e-9,
    0.66358287745352705725e-12,
    0.30375273884444444444e-14,
    0.20136801964214276775e-1,
    0.91936908737673676012e-3,
    0.51734830914104276820e-5,
    0.27900878609710432673e-7,
    0.14357976402809042257e-9,
    0.70114790311043728387e-12,
    0.32252476000000000000e-14,
    0.21996459598282740954e-1,
    0.94040248155366777784e-3,
    0.53443911508041164739e-5,
    0.29078085538049374673e-7,
    0.15078844500329731137e-9,
    0.74103813647499204269e-12,
    0.34251892320000000000e-14,
    0.23898877187226319502e-1,
    0.96213386835900177540e-3,
    0.55225386998049012752e-5,
    0.30314589961047687059e-7,
    0.15840826497296335264e-9,
    0.78340500472414454395e-12,
    0.36381553564444444445e-14,
    0.25845480155298518485e-1,
    0.98459293067820123389e-3,
    0.57082915920051843672e-5,
    0.31613782169164830118e-7,
    0.16646478745529630813e-9,
    0.82840985928785407942e-12,
    0.38649975768888888890e-14,
    0.27837754783474696598e-1,
    0.10078108563256892757e-2,
    0.59020366493792212221e-5,
    0.32979263553246520417e-7,
    0.17498524159268458073e-9,
    0.87622459124842525110e-12,
    0.41066206488888888890e-14,
    0.29877251304899307550e-1,
    0.10318204245057349310e-2,
    0.61041829697162055093e-5,
    0.34414860359542720579e-7,
    0.18399863072934089607e-9,
    0.92703227366365046533e-12,
    0.43639844053333333334e-14,
    0.31965587178596443475e-1,
    0.10566560976716574401e-2,
    0.63151633192414586770e-5,
    0.35924638339521924242e-7,
    0.19353584758781174038e-9,
    0.98102783859889264382e-12,
    0.46381060817777777779e-14,
    0.34104450552588334840e-1,
    0.10823541191350532574e-2,
    0.65354356159553934436e-5,
    0.37512918348533521149e-7,
    0.20362979635817883229e-9,
    0.10384187833037282363e-11,
    0.49300625262222222221e-14,
    0.36295603928292425716e-1,
    0.11089526167995268200e-2,
    0.67654845095518363577e-5,
    0.39184292949913591646e-7,
    0.21431552202133775150e-9,
    0.10994259106646731797e-11,
    0.52409949102222222221e-14,
    0.38540888038840509795e-1,
    0.11364917134175420009e-2,
    0.70058230641246312003e-5,
    0.40943644083718586939e-7,
    0.22563034723692881631e-9,
    0.11642841011361992885e-11,
    0.55721092871111111110e-14,
    0.40842225954785960651e-1,
    0.11650136437945673891e-2,
    0.72569945502343006619e-5,
    0.42796161861855042273e-7,
    0.23761401711005024162e-9,
    0.12332431172381557035e-11,
    0.59246802364444444445e-14,
    0.43201627431540222422e-1,
    0.11945628793917272199e-2,
    0.75195743532849206263e-5,
    0.44747364553960993492e-7,
    0.25030885216472953674e-9,
    0.13065684400300476484e-11,
    0.63000532853333333334e-14,
    0.45621193513810471438e-1,
    0.12251862608067529503e-2,
    0.77941720055551920319e-5,
    0.46803119830954460212e-7,
    0.26375990983978426273e-9,
    0.13845421370977119765e-11,
    0.66996477404444444445e-14,
    0.48103121413299865517e-1,
    0.12569331386432195113e-2,
    0.80814333496367673980e-5,
    0.48969667335682018324e-7,
    0.27801515481905748484e-9,
    0.14674637611609884208e-11,
    0.71249589351111111110e-14,
    0.50649709676983338501e-1,
    0.12898555233099055810e-2,
    0.83820428414568799654e-5,
    0.51253642652551838659e-7,
    0.29312563849675507232e-9,
    0.15556512782814827846e-11,
    0.75775607822222222221e-14,
    0.53263363664388864181e-1,
    0.13240082443256975769e-2,
    0.86967260015007658418e-5,
    0.53662102750396795566e-7,
    0.30914568786634796807e-9,
    0.16494420240828493176e-11,
    0.80591079644444444445e-14,
    0.55946601353500013794e-1,
    0.13594491197408190706e-2,
    0.90262520233016380987e-5,
    0.56202552975056695376e-7,
    0.32613310410503135996e-9,
    0.17491936862246367398e-11,
    0.85713381688888888890e-14,
    0.58702059496154081813e-1,
    0.13962391363223647892e-2,
    0.93714365487312784270e-5,
    0.58882975670265286526e-7,
    0.34414937110591753387e-9,
    0.18552853109751857859e-11,
    0.91160736711111111110e-14,
    0.61532500145144778048e-1,
    0.14344426411912015247e-2,
    0.97331446201016809696e-5,
    0.61711860507347175097e-7,
    0.36325987418295300221e-9,
    0.19681183310134518232e-11,
    0.96952238400000000000e-14,
    0.64440817576653297993e-1,
    0.14741275456383131151e-2,
    0.10112293819576437838e-4,
    0.64698236605933246196e-7,
    0.38353412915303665586e-9,
    0.20881176114385120186e-11,
    0.10310784480000000000e-13,
    0.67430045633130393282e-1,
    0.15153655418916540370e-2,
    0.10509857606888328667e-4,
    0.67851706529363332855e-7,
    0.40504602194811140006e-9,
    0.22157325110542534469e-11,
    0.10964842115555555556e-13,
    0.70503365513338850709e-1,
    0.15582323336495709827e-2,
    0.10926868866865231089e-4,
    0.71182482239613507542e-7,
    0.42787405890153386710e-9,
    0.23514379522274416437e-11,
    0.11659571751111111111e-13,
    0.73664114037944596353e-1,
    0.16028078812438820413e-2,
    0.11364423678778207991e-4,
    0.74701423097423182009e-7,
    0.45210162777476488324e-9,
    0.24957355004088569134e-11,
    0.12397238257777777778e-13,
    0.76915792420819562379e-1,
    0.16491766623447889354e-2,
    0.11823685320041302169e-4,
    0.78420075993781544386e-7,
    0.47781726956916478925e-9,
    0.26491544403815724749e-11,
    0.13180196462222222222e-13,
    0.80262075578094612819e-1,
    0.16974279491709504117e-2,
    0.12305888517309891674e-4,
    0.82350717698979042290e-7,
    0.50511496109857113929e-9,
    0.28122528497626897696e-11,
    0.14010889635555555556e-13,
    0.83706822008980357446e-1,
    0.17476561032212656962e-2,
    0.12812343958540763368e-4,
    0.86506399515036435592e-7,
    0.53409440823869467453e-9,
    0.29856186620887555043e-11,
    0.14891851591111111111e-13,
    0.87254084284461718231e-1,
    0.17999608886001962327e-2,
    0.13344443080089492218e-4,
    0.90900994316429008631e-7,
    0.56486134972616465316e-9,
    0.31698707080033956934e-11,
    0.15825697795555555556e-13,
    0.90908120182172748487e-1,
    0.18544478050657699758e-2,
    0.13903663143426120077e-4,
    0.95549246062549906177e-7,
    0.59752787125242054315e-9,
    0.33656597366099099413e-11,
    0.16815130613333333333e-13,
    0.94673404508075481121e-1,
    0.19112284419887303347e-2,
    0.14491572616545004930e-4,
    0.10046682186333613697e-6,
    0.63221272959791000515e-9,
    0.35736693975589130818e-11,
    0.17862931591111111111e-13,
    0.98554641648004456555e-1,
    0.19704208544725622126e-2,
    0.15109836875625443935e-4,
    0.10567036667675984067e-6,
    0.66904168640019354565e-9,
    0.37946171850824333014e-11,
    0.18971959040000000000e-13,
    0.10255677889470089531e0,
    0.20321499629472857418e-2,
    0.15760224242962179564e-4,
    0.11117756071353507391e-6,
    0.70814785110097658502e-9,
    0.40292553276632563925e-11,
    0.20145143075555555556e-13,
    0.10668502059865093318e0,
    0.20965479776148731610e-2,
    0.16444612377624983565e-4,
    0.11700717962026152749e-6,
    0.74967203250938418991e-9,
    0.42783716186085922176e-11,
    0.21385479360000000000e-13,
    0.11094484319386444474e0,
    0.21637548491908170841e-2,
    0.17164995035719657111e-4,
    0.12317915750735938089e-6,
    0.79376309831499633734e-9,
    0.45427901763106353914e-11,
    0.22696025653333333333e-13,
    0.11534201115268804714e0,
    0.22339187474546420375e-2,
    0.17923489217504226813e-4,
    0.12971465288245997681e-6,
    0.84057834180389073587e-9,
    0.48233721206418027227e-11,
    0.24079890062222222222e-13,
    0.11988259392684094740e0,
    0.23071965691918689601e-2,
    0.18722342718958935446e-4,
    0.13663611754337957520e-6,
    0.89028385488493287005e-9,
    0.51210161569225846701e-11,
    0.25540227111111111111e-13,
    0.12457298393509812907e0,
    0.23837544771809575380e-2,
    0.19563942105711612475e-4,
    0.14396736847739470782e-6,
    0.94305490646459247016e-9,
    0.54366590583134218096e-11,
    0.27080225920000000000e-13,
    0.12941991566142438816e0,
    0.24637684719508859484e-2,
    0.20450821127475879816e-4,
    0.15173366280523906622e-6,
    0.99907632506389027739e-9,
    0.57712760311351625221e-11,
    0.28703099555555555556e-13,
    0.13443048593088696613e0,
    0.25474249981080823877e-2,
    0.21385669591362915223e-4,
    0.15996177579900443030e-6,
    0.10585428844575134013e-8,
    0.61258809536787882989e-11,
    0.30412080142222222222e-13,
    0.13961217543434561353e0,
    0.26349215871051761416e-2,
    0.22371342712572567744e-4,
    0.16868008199296822247e-6,
    0.11216596910444996246e-8,
    0.65015264753090890662e-11,
    0.32210394506666666666e-13,
    0.14497287157673800690e0,
    0.27264675383982439814e-2,
    0.23410870961050950197e-4,
    0.17791863939526376477e-6,
    0.11886425714330958106e-8,
    0.68993039665054288034e-11,
    0.34101266222222222221e-13,
    0.15052089272774618151e0,
    0.28222846410136238008e-2,
    0.24507470422713397006e-4,
    0.18770927679626136909e-6,
    0.12597184587583370712e-8,
    0.73203433049229821618e-11,
    0.36087889048888888890e-13,
    0.15626501395774612325e0,
    0.29226079376196624949e-2,
    0.25664553693768450545e-4,
    0.19808568415654461964e-6,
    0.13351257759815557897e-8,
    0.77658124891046760667e-11,
    0.38173420035555555555e-13,
    0.16221449434620737567e0,
    0.30276865332726475672e-2,
    0.26885741326534564336e-4,
    0.20908350604346384143e-6,
    0.14151148144240728728e-8,
    0.82369170665974313027e-11,
    0.40360957457777777779e-13,
    0.16837910595412130659e0,
    0.31377844510793082301e-2,
    0.28174873844911175026e-4,
    0.22074043807045782387e-6,
    0.14999481055996090039e-8,
    0.87348993661930809254e-11,
    0.42653528977777777779e-13,
    0.17476916455659369953e0,
    0.32531815370903068316e-2,
    0.29536024347344364074e-4,
    0.23309632627767074202e-6,
    0.15899007843582444846e-8,
    0.92610375235427359475e-11,
    0.45054073102222222221e-13,
    0.18139556223643701364e0,
    0.33741744168096996041e-2,
    0.30973511714709500836e-4,
    0.24619326937592290996e-6,
    0.16852609412267750744e-8,
    0.98166442942854895573e-11,
    0.47565418097777777779e-13,
    0.18826980194443664549e0,
    0.35010775057740317997e-2,
    0.32491914440014267480e-4,
    0.26007572375886319028e-6,
    0.17863299617388376116e-8,
    0.10403065638343878679e-10,
    0.50190265831111111110e-13,
    0.19540403413693967350e0,
    0.36342240767211326315e-2,
    0.34096085096200907289e-4,
    0.27479061117017637474e-6,
    0.18934228504790032826e-8,
    0.11021679075323598664e-10,
    0.52931171733333333334e-13,
    0.20281109560651886959e0,
    0.37739673859323597060e-2,
    0.35791165457592409054e-4,
    0.29038742889416172404e-6,
    0.20068685374849001770e-8,
    0.11673891799578381999e-10,
    0.55790523093333333334e-13,
    0.21050455062669334978e0,
    0.39206818613925652425e-2,
    0.37582602289680101704e-4,
    0.30691836231886877385e-6,
    0.21270101645763677824e-8,
    0.12361138551062899455e-10,
    0.58770520160000000000e-13,
    0.21849873453703332479e0,
    0.40747643554689586041e-2,
    0.39476163820986711501e-4,
    0.32443839970139918836e-6,
    0.22542053491518680200e-8,
    0.13084879235290858490e-10,
    0.61873153262222222221e-13,
    0.22680879990043229327e0,
    0.42366354648628516935e-2,
    0.41477956909656896779e-4,
    0.34300544894502810002e-6,
    0.23888264229264067658e-8,
    0.13846596292818514601e-10,
    0.65100183751111111110e-13,
    0.23545076536988703937e0,
    0.44067409206365170888e-2,
    0.43594444916224700881e-4,
    0.36268045617760415178e-6,
    0.25312606430853202748e-8,
    0.14647791812837903061e-10,
    0.68453122631111111110e-13,
    0.24444156740777432838e0,
    0.45855530511605787178e-2,
    0.45832466292683085475e-4,
    0.38352752590033030472e-6,
    0.26819103733055603460e-8,
    0.15489984390884756993e-10,
    0.71933206364444444445e-13,
    0.25379911500634264643e0,
    0.47735723208650032167e-2,
    0.48199253896534185372e-4,
    0.40561404245564732314e-6,
    0.28411932320871165585e-8,
    0.16374705736458320149e-10,
    0.75541379822222222221e-13,
    0.26354234756393613032e0,
    0.49713289477083781266e-2,
    0.50702455036930367504e-4,
    0.42901079254268185722e-6,
    0.30095422058900481753e-8,
    0.17303497025347342498e-10,
    0.79278273368888888890e-13,
    0.27369129607732343398e0,
    0.51793846023052643767e-2,
    0.53350152258326602629e-4,
    0.45379208848865015485e-6,
    0.31874057245814381257e-8,
    0.18277905010245111046e-10,
    0.83144182364444444445e-13,
    0.28426714781640316172e0,
    0.53983341916695141966e-2,
    0.56150884865255810638e-4,
    0.48003589196494734238e-6,
    0.33752476967570796349e-8,
    0.19299477888083469086e-10,
    0.87139049137777777779e-13,
    0.29529231465348519920e0,
    0.56288077305420795663e-2,
    0.59113671189913307427e-4,
    0.50782393781744840482e-6,
    0.35735475025851713168e-8,
    0.20369760937017070382e-10,
    0.91262442613333333334e-13,
    0.30679050522528838613e0,
    0.58714723032745403331e-2,
    0.62248031602197686791e-4,
    0.53724185766200945789e-6,
    0.37827999418960232678e-8,
    0.21490291930444538307e-10,
    0.95513539182222222221e-13,
    0.31878680111173319425e0,
    0.61270341192339103514e-2,
    0.65564012259707640976e-4,
    0.56837930287837738996e-6,
    0.40035151353392378882e-8,
    0.22662596341239294792e-10,
    0.99891109760000000000e-13,
    0.33130773722152622027e0,
    0.63962406646798080903e-2,
    0.69072209592942396666e-4,
    0.60133006661885941812e-6,
    0.42362183765883466691e-8,
    0.23888182347073698382e-10,
    0.10439349811555555556e-12,
    0.34438138658041336523e0,
    0.66798829540414007258e-2,
    0.72783795518603561144e-4,
    0.63619220443228800680e-6,
    0.44814499336514453364e-8,
    0.25168535651285475274e-10,
    0.10901861383111111111e-12,
    0.35803744972380175583e0,
    0.69787978834882685031e-2,
    0.76710543371454822497e-4,
    0.67306815308917386747e-6,
    0.47397647975845228205e-8,
    0.26505114141143050509e-10,
    0.11376390933333333333e-12,
    0.37230734890119724188e0,
    0.72938706896461381003e-2,
    0.80864854542670714092e-4,
    0.71206484718062688779e-6,
    0.50117323769745883805e-8,
    0.27899342394100074165e-10,
    0.11862637614222222222e-12,
    0.38722432730555448223e0,
    0.76260375162549802745e-2,
    0.85259785810004603848e-4,
    0.75329383305171327677e-6,
    0.52979361368388119355e-8,
    0.29352606054164086709e-10,
    0.12360253370666666667e-12,
    0.40282355354616940667e0,
    0.79762880915029728079e-2,
    0.89909077342438246452e-4,
    0.79687137961956194579e-6,
    0.55989731807360403195e-8,
    0.30866246101464869050e-10,
    0.12868841946666666667e-12,
    0.41914223158913787649e0,
    0.83456685186950463538e-2,
    0.94827181359250161335e-4,
    0.84291858561783141014e-6,
    0.59154537751083485684e-8,
    0.32441553034347469291e-10,
    0.13387957943111111111e-12,
    0.43621971639463786896e0,
    0.87352841828289495773e-2,
    0.10002929142066799966e-3,
    0.89156148280219880024e-6,
    0.62480008150788597147e-8,
    0.34079760983458878910e-10,
    0.13917107176888888889e-12,
    0.45409763548534330981e0,
    0.91463027755548240654e-2,
    0.10553137232446167258e-3,
    0.94293113464638623798e-6,
    0.65972492312219959885e-8,
    0.35782041795476563662e-10,
    0.14455745872000000000e-12,
    0.47282001668512331468e0,
    0.95799574408860463394e-2,
    0.11135019058000067469e-3,
    0.99716373005509038080e-6,
    0.69638453369956970347e-8,
    0.37549499088161345850e-10,
    0.15003280712888888889e-12,
    0.49243342227179841649e0,
    0.10037550043909497071e-1,
    0.11750334542845234952e-3,
    0.10544006716188967172e-5,
    0.73484461168242224872e-8,
    0.39383162326435752965e-10,
    0.15559069118222222222e-12,
    0.51298708979209258326e0,
    0.10520454564612427224e-1,
    0.12400930037494996655e-3,
    0.11147886579371265246e-5,
    0.77517184550568711454e-8,
    0.41283980931872622611e-10,
    0.16122419680000000000e-12,
    0.53453307979101369843e0,
    0.11030120618800726938e-1,
    0.13088741519572269581e-3,
    0.11784797595374515432e-5,
    0.81743383063044825400e-8,
    0.43252818449517081051e-10,
    0.16692592640000000000e-12,
    0.55712643071169299478e0,
    0.11568077107929735233e-1,
    0.13815797838036651289e-3,
    0.12456314879260904558e-5,
    0.86169898078969313597e-8,
    0.45290446811539652525e-10,
    0.17268801084444444444e-12,
    0.58082532122519320968e0,
    0.12135935999503877077e-1,
    0.14584223996665838559e-3,
    0.13164068573095710742e-5,
    0.90803643355106020163e-8,
    0.47397540713124619155e-10,
    0.17850211608888888889e-12,
    0.60569124025293375554e0,
    0.12735396239525550361e-1,
    0.15396244472258863344e-3,
    0.13909744385382818253e-5,
    0.95651595032306228245e-8,
    0.49574672127669041550e-10,
    0.18435945564444444444e-12,
    0.63178916494715716894e0,
    0.13368247798287030927e-1,
    0.16254186562762076141e-3,
    0.14695084048334056083e-5,
    0.10072078109604152350e-7,
    0.51822304995680707483e-10,
    0.19025081422222222222e-12,
    0.65918774689725319200e0,
    0.14036375850601992063e-1,
    0.17160483760259706354e-3,
    0.15521885688723188371e-5,
    0.10601827031535280590e-7,
    0.54140790105837520499e-10,
    0.19616655146666666667e-12,
    0.68795950683174433822e0,
    0.14741765091365869084e-1,
    0.18117679143520433835e-3,
    0.16392004108230585213e-5,
    0.11155116068018043001e-7,
    0.56530360194925690374e-10,
    0.20209663662222222222e-12,
    0.71818103808729967036e0,
    0.15486504187117112279e-1,
    0.19128428784550923217e-3,
    0.17307350969359975848e-5,
    0.11732656736113607751e-7,
    0.58991125287563833603e-10,
    0.20803065333333333333e-12,
    0.74993321911726254661e0,
    0.16272790364044783382e-1,
    0.20195505163377912645e-3,
    0.18269894883203346953e-5,
    0.12335161021630225535e-7,
    0.61523068312169087227e-10,
    0.21395783431111111111e-12,
    0.78330143531283492729e0,
    0.17102934132652429240e-1,
    0.21321800585063327041e-3,
    0.19281661395543913713e-5,
    0.12963340087354341574e-7,
    0.64126040998066348872e-10,
    0.21986708942222222222e-12,
    0.81837581041023811832e0,
    0.17979364149044223802e-1,
    0.22510330592753129006e-3,
    0.20344732868018175389e-5,
    0.13617902941839949718e-7,
    0.66799760083972474642e-10,
    0.22574701262222222222e-12,
    0.85525144775685126237e0,
    0.18904632212547561026e-1,
    0.23764237370371255638e-3,
    0.21461248251306387979e-5,
    0.14299555071870523786e-7,
    0.69543803864694171934e-10,
    0.23158593688888888889e-12,
    0.89402868170849933734e0,
    0.19881418399127202569e-1,
    0.25086793128395995798e-3,
    0.22633402747585233180e-5,
    0.15008997042116532283e-7,
    0.72357609075043941261e-10,
    0.23737194737777777778e-12,
    0.93481333942870796363e0,
    0.20912536329780368893e-1,
    0.26481403465998477969e-3,
    0.23863447359754921676e-5,
    0.15746923065472184451e-7,
    0.75240468141720143653e-10,
    0.24309291271111111111e-12,
    0.97771701335885035464e0,
    0.22000938572830479551e-1,
    0.27951610702682383001e-3,
    0.25153688325245314530e-5,
    0.16514019547822821453e-7,
    0.78191526829368231251e-10,
    0.24873652355555555556e-12,
]
