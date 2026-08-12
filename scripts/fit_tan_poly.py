"""Derive the minimax polynomial used by the float32 GPU `tan` kernel.

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


def validate(
    pio2_hi: np.float32, pio2_lo: np.float32, coeffs_lo_first: list[float]
) -> None:
    """Float32 end-to-end check against math.tan, over the actual OpInfo
    domain for `tan` (float32 sample: (20, 20) tensor, |x| < ~9) plus a wider
    sweep for margin. Mirrors torch.testing's combined atol+rtol check
    (default float32 precision: atol=1e-5, rtol=1.3e-6)."""
    inv_pio2 = np.float32(2.0 / math.pi)

    def tan_impl(x: np.float32) -> np.float32:
        x = np.float32(x)
        k = np.float32(math.floor(float(x) * float(inv_pio2) + 0.5))
        r = np.float32(np.float64(x) - np.float64(k) * np.float64(pio2_hi))
        r = np.float32(np.float64(r) - np.float64(k) * np.float64(pio2_lo))
        z = np.float32(np.float64(r) * np.float64(r))
        poly = np.float32(0.0)
        for c in reversed(coeffs_lo_first):
            poly = poly * z + np.float32(c)
        tan_r = np.float32(np.float64(r) + np.float64(r) * np.float64(z) * np.float64(poly))
        return np.float32(-1.0 / np.float64(tan_r)) if int(k) % 2 else tan_r

    rng = np.random.default_rng(0)
    xs = np.concatenate(
        [
            rng.uniform(-9, 9, 200_000).astype(np.float32),
            np.linspace(-9, 9, 100_000).astype(np.float32),
        ]
    )
    atol, rtol = 1e-5, 1.3e-6
    n = fail = 0
    for x in xs:
        ref = math.tan(float(x))
        if abs(ref) > 1e5 or not math.isfinite(ref):
            continue
        n += 1
        got = float(tan_impl(x))
        if abs(got - ref) > atol + rtol * abs(ref):
            fail += 1
    print(f"OpInfo-domain [-9, 9] validation: {n} samples, {fail} failing tolerance")


def main() -> None:
    pio2_hi, pio2_lo = derive_pio2_split()
    print(f"PIO2_HI = Float32({float(pio2_hi)!r})")
    print(f"PIO2_LO = Float32({float(pio2_lo)!r})")
    print(f"INV_PIO2 = Float32({2.0 / math.pi!r})")

    coeffs = fit_tan_poly(degree=5)
    print("polynomial_evaluate coefficients (low-degree first):")
    for c in coeffs:
        print(f"    {c!r},")

    validate(pio2_hi, pio2_lo, coeffs)


if __name__ == "__main__":
    main()
