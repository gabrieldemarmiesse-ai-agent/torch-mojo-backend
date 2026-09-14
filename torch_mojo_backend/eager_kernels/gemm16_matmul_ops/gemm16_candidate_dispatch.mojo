"""Runtime regime gates of the three measured bf16 candidates.

Reached from the top of `enqueue_gemm16_gemm` (the single-matrix GEMM
entry); every helper here returns False WITHOUT launching anything for a
shape it does not serve, and the whole pre-existing ladder is what runs
then. The TN selection launches unchanged upstream device bodies -- what it
adds is coverage and a wave-cost comparison; NN and fused-NT launch the
kernels in gemm16_rolling_kernels.mojo and gemm16_nt_bias_kernels.mojo.

Every crossover constant below was fitted on an H100 PCIe (114 SMs) at
1410 MHz. Matrix dimensions and the SM count stay runtime values: the gates
are shape REGIMES (residue-64 widths, bounded aspect ratios, deep K, grids
that fill more than one wave), never particular sizes.
"""
from max.gpu.host import DeviceContext, DeviceAttribute
from std.sys.info import _has_sm_9x
from gemm16_dtype import _GEMM16_DT
from gemm16_nn_v4_kernels import _v4_enqueue_nn_persistent
from gemm16_tn_v4_kernels import _v4_enqueue_direct_m128n192
from gemm16_rolling_kernels import enqueue_rolling_persistent
from gemm16_nt_bias_kernels import maybe_enqueue_gemm16_nt_bias_v4

comptime PTR = Pointer[Scalar[_GEMM16_DT], MutAnyOrigin]


def try_enqueue_candidate_nt_bias(
    output: PTR,
    a: PTR,
    b: PTR,
    bias: PTR,
    m: Int,
    n: Int,
    k: Int,
    has_bias: Bool,
    ctx: DeviceContext,
) raises -> Bool:
    """Conservative large NT bias regime; original routes handle the rest.

    TUNE_NT_ROLLING=True and TUNE_NT_RASTER=8 reproduce H100 measurements.
    The existing fused helper checks dtype, architecture, pointers, TMA
    extents, products and launch resources after this performance gate.
    """
    if (
        not has_bias
        or m < 4096
        or n < 1024
        or k < 1024
        or m % 128 != 0
        or n % 64 != 0
        or k % 64 != 0
    ):
        return False
    if n % 128 != 64 and k % 128 != 64:
        return False
    # Overflow-safe max(N,K)<=8*min(N,K); reject very wide vocabulary work.
    if 1 + (max(n, k) - 1) // 8 > min(n, k):
        return False
    return maybe_enqueue_gemm16_nt_bias_v4(output, a, b, bias, m, n, k, ctx)


def try_enqueue_candidate_nn(
    output: PTR,
    a: PTR,
    b: PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Tall, clipped-N bf16 regime; other products retain their old route.

    Compile the measured epilogue with PAIR_CAST=True. The tile and aspect
    crossovers are fitted on H100 PCIe, using runtime waves. No matrix
    dimension is specialized at compilation.
    """
    comptime if not _has_sm_9x() or _GEMM16_DT != DType.bfloat16:
        return False
    if ctx.api() != "cuda":
        return False
    if (
        ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR) != 9
        or ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR) != 0
    ):
        return False
    if (
        m < 256
        or n < 512
        or k < 1024
        or m % 128 != 0
        or n % 256 != 64
        or k % 64 != 0
        or Int(output) % 16 != 0
        or Int(a) % 16 != 0
        or Int(b) % 16 != 0
        or m > 2_147_483_647
        or n > 2_147_483_647
        or k > 2_147_483_647
        or k > 9_223_372_036_854_775_807 // m
        or k > 9_223_372_036_854_775_807 // n
        or n > 9_223_372_036_854_775_807 // m
    ):
        return False
    if m < 4 * n or m > 32 * n or k < n or k > 8 * n:
        return False
    var sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    if sms < 2:
        return False
    var clusters = sms // 2
    var macro192 = (m + 383) // 384
    var work256 = ((m + 255) // 256) * ((n + 255) // 256)
    var wave256 = (work256 + clusters - 1) // clusters
    var width = 192
    var work192 = macro192 * ((n + width - 1) // width)
    if 2 * work192 > ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X):
        return False
    var wave192 = (work192 + clusters - 1) // clusters
    # Admit only when the new geometry reduces or preserves rounded-wave
    # arithmetic. Small/short and aligned regimes keep their old kernels.
    if wave192 * 192 * width > wave256 * 128 * 256:
        return False
    enqueue_rolling_persistent[3, 2, 192, 192, 3, True, False, False, True](
        output, a, b, m, n, k, sms, ctx
    )
    return True


def try_enqueue_candidate_tn(
    output: PTR,
    a: PTR,
    b: PTR,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    comptime if not _has_sm_9x() or _GEMM16_DT != DType.bfloat16:
        return False
    if ctx.api() != "cuda":
        return False
    if (
        ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR) != 9
        or ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MINOR) != 0
    ):
        return False
    # Deep-K and wave crossovers are fitted on H100 PCIe. The gate protects
    # underfilled grids, short reductions and very tall vocabulary products.
    if (
        m < 256
        or n < 256
        or k < 4096
        or m % 64 != 0
        or n % 64 != 0
        or k % 64 != 0
        or Int(output) % 16 != 0
        or Int(a) % 16 != 0
        or Int(b) % 16 != 0
        or m > 2_147_483_647
        or n > 2_147_483_647
        or k > 2_147_483_647
        or k > 9_223_372_036_854_775_807 // m
        or k > 9_223_372_036_854_775_807 // n
        or n > 9_223_372_036_854_775_807 // m
    ):
        return False
    var sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var max_grid = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)
    if sms < 2:
        return False
    var tiles192 = ((m + 127) // 128) * ((n + 191) // 192)
    if tiles192 <= sms or tiles192 > max_grid:
        return False
    var clusters192 = ((m + 255) // 256) * ((n + 191) // 192)
    var clusters256 = ((m + 255) // 256) * ((n + 255) // 256)
    var waves192 = (clusters192 + sms // 2 - 1) // (sms // 2)
    var waves256 = (clusters256 + sms // 2 - 1) // (sms // 2)
    var direct_waves192 = (tiles192 + sms - 1) // sms
    if m % 128 == 64:
        # A wide output exposes the wasted peer row of an M-tail cluster;
        # independent direct CTAs avoid that work. Ceil launch is essential.
        if n >= 2 * m and direct_waves192 * 192 < waves256 * 256:
            _v4_enqueue_direct_m128n192(
                output, a, b, m, n, k, tiles192, True, ctx
            )
            return True
        if waves192 * 192 < waves256 * 256:
            _v4_enqueue_nn_persistent[
                4, 2, 128, 192, 2, True, True, False, True
            ](output, a, b, m, n, k, sms, ctx)
        else:
            _v4_enqueue_nn_persistent[
                3, 2, 128, 256, 2, True, True, False, True
            ](output, a, b, m, n, k, sms, ctx)
        return True
    # Preserve aligned routes except this clipped-column, moderate-aspect
    # regime where the four-stage 192 tile removes excess per-wave work.
    if (
        n % 256 == 64
        and m >= 2 * n
        and m <= 8 * n
        and waves192 * 192 < waves256 * 256
    ):
        _v4_enqueue_nn_persistent[4, 2, 128, 192, 2, True, True, False, True](
            output, a, b, m, n, k, sms, ctx
        )
        return True
    return False
