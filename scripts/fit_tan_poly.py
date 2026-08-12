"""Derive the least-squares polynomial used by the float32 GPU `tan` kernel.

`torch_mojo_backend/eager_kernels/elementwise_ops/elementwise_ops.mojo`
(`_float_unary`, `UOP_TAN`, float32 branch) does not compute tan(x) as
sin(x)/cos(x): on GPU those each use a fixed ~1e-6-absolute-error hardware
approx instruction, and dividing two such values right where cos(x) is small
(near a pole, x ~ pi/2 + k*pi) turns that fixed absolute error into a large
*relative* error in tan.

Instead the kernel reduces x to the nearest multiple of pi/2 (Cody-Waite,
exact in float32 for any realistic |k|) so the residual r is always in
[-pi/4, pi/4] -- bounded away from every pole -- then evaluates tan(r) with
the polynomial this script fits, using only the identity
tan(r) = r + r^3 * f(r^2) with f fit directly (no reliance on hardware
sin/cos at all, so the near-zero-r case near a pole stays well-conditioned).
Near a pole the reduction lands r close to 0, and -1/tan(r) reproduces the
pole's blow-up without ever dividing two independently-rounded quantities
against each other.

Run with `uv run python scripts/fit_tan_poly.py` to reproduce the constants
embedded in the kernel (PIO2_HI, PIO2_LO, and the 6 polynomial coefficients).
"""

import math

import numpy as np


def derive_pio2_split() -> tuple[np.float32, np.float32]:
    """Cody-Waite hi/lo split of pi/2, both exactly representable in float32.

    PIO2_HI keeps only the top ~10 explicit mantissa bits (the rest zeroed),
    so `k * PIO2_HI` is an EXACT float32 multiply for any |k| < 2**13 -- far
    beyond any k this op will see (the OpInfo domain for tan is |x| < ~9, so
    k = round(x / (pi/2)) stays in single digits). PIO2_LO is the leftover
    correction, giving the *sum* effectively ~48 bits of precision for pi/2.
    """
    pi_half = np.float64(math.pi) / 2.0
    hi_bits = np.float32(pi_half).view(np.uint32) & np.uint32(0xFFFFE000)
    pio2_hi = hi_bits.view(np.float32)
    pio2_lo = np.float32(pi_half - np.float64(pio2_hi))
    return pio2_hi, pio2_lo


def fit_tan_poly(degree: int = 5) -> list[float]:
    """Least-squares fit of f(z) = (tan(sqrt(z)) - sqrt(z)) / sqrt(z)**3,
    z = r**2, over the reduced domain r in [-pi/4, pi/4] -- so
    tan(r) = r + r**3 * f(r**2). Returns coefficients low-degree-first,
    matching `polynomial_evaluate`'s Horner-form argument order.
    """
    z_max = (math.pi / 4) ** 2
    z = np.linspace(0, z_max, 400_000)
    r = np.sqrt(z)
    tan_true = np.tan(r)
    f_true = np.empty_like(z)
    nonzero = z > 0
    f_true[nonzero] = (tan_true[nonzero] - r[nonzero]) / r[nonzero] ** 3
    f_true[~nonzero] = 1.0 / 3.0  # lim r->0 of (tan(r)-r)/r**3 = 1/3

    coeffs_hi_first = np.polyfit(z, f_true, degree)
    return list(reversed(coeffs_hi_first.tolist()))


def validate_polynomial(coeffs_lo_first: list[float]) -> None:
    """Accuracy of tan(r) = r + r^3*poly(z) alone, r swept densely over its
    whole domain [-pi/4, pi/4] -- the specific "<2e-7 relative error"
    claim the kernel comment makes for the polynomial, isolated from the
    range reduction and from the near-pole amplification `validate` below
    hits for x close to (2k+1)*pi/2 (inherent to any finite-precision tan,
    not a property of this polynomial). Every operator is float32-native,
    same rationale as `validate`."""
    r_max = np.float32(math.pi / 4)
    rs = np.linspace(-r_max, r_max, 400_000, dtype=np.float32)
    max_rel = 0.0
    for r in rs:
        z = r * r
        poly = np.float32(0.0)
        for c in reversed(coeffs_lo_first):
            poly = poly * z + np.float32(c)
        got = float(r + r * z * poly)
        ref = math.tan(float(r))
        if abs(ref) > 1e-6:
            max_rel = max(max_rel, abs(got - ref) / abs(ref))
    print(f"polynomial-only max relative error over [-pi/4, pi/4]: {max_rel:.3e}")


def validate(
    pio2_hi: np.float32, pio2_lo: np.float32, coeffs_lo_first: list[float]
) -> None:
    """Float32 end-to-end check against math.tan, over the actual OpInfo
    domain for `tan` (float32 sample: (20, 20) tensor, |x| < ~9) plus a wider
    sweep for margin. Mirrors torch.testing's combined atol+rtol check
    (default float32 precision: atol=1e-5, rtol=1.3e-6).

    Every operator below is applied to `np.float32` scalars directly (no
    float64 intermediate, even inside one compound expression): with NEP 50
    (NumPy >= 2.0) same-dtype scalar arithmetic stays in that dtype end to
    end, so each `-`/`*`/`+` rounds once, in the same order, as the
    corresponding Mojo `Scalar[Float32]` operator in the actual kernel
    (`_float_unary`, `UOP_TAN`). Routing any of these through float64 first
    would round fewer times than the real kernel does and report a rosier
    number than what actually ships.
    """
    inv_pio2 = np.float32(2.0 / math.pi)

    def tan_impl(x: np.float32) -> np.float32:
        x = np.float32(x)
        # The kernel uses a fused multiply-add here (`af.fma(INV_PIO2,
        # 0.5)`, one rounding for the multiply+add together), not two
        # separate float32 roundings. float64 has enough spare mantissa
        # bits to hold the exact product of two float32s plus a float32
        # exactly (24+24 significant bits fits well inside 53), so summing
        # in Python's native float and rounding once reproduces a true
        # float32 FMA bit-for-bit, unlike routing the whole reduction
        # through float64 (which is what this function must NOT do).
        t = np.float32(float(x) * float(inv_pio2) + 0.5)
        k = np.float32(math.floor(float(t)))
        r = (x - k * pio2_hi) - k * pio2_lo
        z = r * r
        poly = np.float32(0.0)
        for c in reversed(coeffs_lo_first):
            poly = poly * z + np.float32(c)
        tan_r = r + r * z * poly
        return np.float32(-1.0 / float(tan_r)) if int(k) % 2 else tan_r

    rng = np.random.default_rng(0)
    xs = np.concatenate(
        [
            rng.uniform(-9, 9, 200_000).astype(np.float32),
            np.linspace(-9, 9, 100_000).astype(np.float32),
        ]
    )
    atol, rtol = 1e-5, 1.3e-6
    # x within this margin of an odd multiple of pi/2 has |tan(x)| large
    # enough (empirically, into the tens of thousands here) that rtol alone
    # demands sub-ULP agreement on x itself -- no finite-precision tan
    # clears that bar near a true pole, so these are reported separately
    # rather than folded into "fail" as if they were an algorithm defect.
    near_pole_margin = 0.05
    n = fail = near_pole = 0
    max_rel = 0.0
    for x in xs:
        ref = math.tan(float(x))
        if abs(ref) > 1e5 or not math.isfinite(ref):
            continue
        n += 1
        got = float(tan_impl(x))
        diff = abs(got - ref)
        ok = diff <= atol + rtol * abs(ref)
        # Poles sit at odd multiples of pi/2; distance from x to the
        # nearest one, in radians.
        q = float(x) / (math.pi / 2)
        nearest_odd_multiple = 2 * round((q - 1) / 2) + 1
        dist_to_pole = abs(q - nearest_odd_multiple) * (math.pi / 2)
        if not ok:
            if dist_to_pole < near_pole_margin:
                near_pole += 1
            else:
                fail += 1
        if abs(ref) > 1e-3:  # skip zero-crossings: relative error is unbounded there
            max_rel = max(max_rel, diff / abs(ref))
    print(f"OpInfo-domain [-9, 9] validation: {n} samples")
    print(f"  failing tolerance, away from a pole: {fail}")
    print(f"  failing tolerance, within {near_pole_margin} rad of a pole: {near_pole}")
    print(f"  max relative error (|tan(x)| > 1e-3): {max_rel:.3e}")


def main() -> None:
    pio2_hi, pio2_lo = derive_pio2_split()
    print(f"PIO2_HI = Float32({float(pio2_hi)!r})")
    print(f"PIO2_LO = Float32({float(pio2_lo)!r})")
    print(f"INV_PIO2 = Float32({2.0 / math.pi!r})")

    coeffs = fit_tan_poly(degree=5)
    print("polynomial_evaluate coefficients (low-degree first):")
    for c in coeffs:
        print(f"    {c!r},")

    validate_polynomial(coeffs)
    validate(pio2_hi, pio2_lo, coeffs)


if __name__ == "__main__":
    main()
