"""CPU-only-Torch-compatible bridge to the vendored dense FA4 kernels.

All launches use the backend-owned MAX DeviceContext default stream. They are
asynchronous; synchronization belongs at explicit consumer/benchmark
boundaries, never between forward or backward component kernels.
"""

from std.math import ceildiv
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp

from fa4_fwd_launch import launch_fwd_fa4
from fa4_fwd_selfload_launch import launch_fwd_fa4_selfload
from fa4_fwd_selfload_common import kFa4BlockM as kFa4SelfloadBlockM
from fa4_fwd_selfload_common import kFa4CtasPerSm as kFa4SelfloadCtasPerSm
from fa4_bwd_launch import (
    launch_bwd_preprocess,
    launch_bwd_main,
    launch_bwd_convert,
)

# PHASE 2c wave-gate threshold (NOTES.md, /scratch/fa4-fwd-harness-2c,
# "Phase 2c" section 5): the self-loading (3 CTAs/SM) bhsd route beats
# the phase-2b (2 CTAs/SM) geometry once there is enough parallel work
# to fill a third CTA everywhere; below it, the shallower 4-stage ring
# and the missing dedicated producer warpgroup are pure cost with
# nothing to hide behind. Measured waves at this divisor: P0/P1/P2/P3 at
# 4.5/17/3.0/23 all clear the bar; P4/P5 at 0.8/0.2 both regress (5-9%)
# but stay under 1.10x cuDNN either way, so the gate is not delicate.
# FITTED ON H100 PCIe (114 SMs) -- re-derive on other cards; nothing
# here is architecture-specific in principle, but the threshold was
# never measured off Hopper.
comptime _FA4_SELFLOAD_MIN_WAVES: Int = 2


def _fa4_bhsd_selfload_waves(
    batch: Int, seqlen: Int, nheads: Int, ctx_handle_addr: Int
) raises -> Int:
    """Runtime wave count for the self-loading (3 CTAs/SM) bhsd route.

    Deliberately its OWN computation, not shared with
    ``launch_fwd_fa4``'s internal L2-swizzle wave count (which keeps
    dividing by ITS OWN 2-CTAs/SM occupancy): conflating the two would
    move the phase-2b ">= 12 waves" swizzle-group threshold for shapes
    that stay on the phase-2b geometry, an unmeasured combination this
    gate must not create (NOTES.md "Phase 2c" handoff item 4).
    """
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var num_m = ceildiv(seqlen, kFa4SelfloadBlockM(64))
    return (num_m * nheads * batch) // (kFa4SelfloadCtasPerSm(64) * sm_count)


def flash_attention_fwd_bf16_d64_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_fwd_fa4[
        DType.bfloat16,
        64,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_fwd_f16_d64_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same dense d64 causal forward as the bf16 entry point above, only the
    comptime ``dtype`` differs. The f16 RS (register-A) wgmma emitters live
    in ``fa4_wgmma_f16.mojo`` and are selected inside the shared kernel by
    ``comptime if dtype == DType.float16`` -- bf16 keeps the stdlib path
    byte-identical."""
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_fwd_fa4[
        DType.float16,
        64,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_bwd_bf16_d64_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var dout_addr = Int(py=args[4])
    var lse_addr = Int(py=args[5])
    var dq_addr = Int(py=args[6])
    var dk_addr = Int(py=args[7])
    var dv_addr = Int(py=args[8])
    var dpsum_addr = Int(py=args[9])
    var lse_log2_addr = Int(py=args[10])
    var dq_accum_addr = Int(py=args[11])
    var batch = Int(py=args[12])
    var seqlen = Int(py=args[13])
    var nheads = Int(py=args[14])
    var softmax_scale = Float32(py=args[15])
    var ctx_addr = Int(py=args[16])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_bwd_preprocess[
        DType.bfloat16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.bfloat16, 64, False, True, 1, False, False, 0
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
    )
    launch_bwd_convert[
        DType.bfloat16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_bwd_f16_d64_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same dense d64 causal backward as the bf16 entry point above, only
    the comptime ``dtype`` differs (see ``flash_attention_fwd_f16_d64_causal``
    for the f16 RS wgmma note)."""
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var dout_addr = Int(py=args[4])
    var lse_addr = Int(py=args[5])
    var dq_addr = Int(py=args[6])
    var dk_addr = Int(py=args[7])
    var dv_addr = Int(py=args[8])
    var dpsum_addr = Int(py=args[9])
    var lse_log2_addr = Int(py=args[10])
    var dq_accum_addr = Int(py=args[11])
    var batch = Int(py=args[12])
    var seqlen = Int(py=args[13])
    var nheads = Int(py=args[14])
    var softmax_scale = Float32(py=args[15])
    var ctx_addr = Int(py=args[16])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_bwd_preprocess[
        DType.float16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.float16, 64, False, True, 1, False, False, 0
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
    )
    launch_bwd_convert[
        DType.float16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def _check_strided_qkv_layout(
    name: StaticString,
    addr: Int,
    b_stride: Int,
    s_stride: Int,
    h_stride: Int,
    d_stride: Int,
    seqlen: Int,
    nheads: Int,
    head_dim: Int,
) raises:
    """Reject any Q/K/V layout outside the strict zero-copy regime.

    Runs BEFORE any descriptor is created or kernel enqueued so a
    violation never partially launches. Strides are in Q/K/V dtype
    ELEMENTS (bf16 or f16 -- shared by both entry-point families since
    both are 2-byte types with identical 16-byte-multiple math).
    ``head_dim`` is a runtime value here even though every caller
    passes a literal (64 or 128, matching its own comptime instance)
    -- this check has no comptime context of its own.
    """
    if b_stride <= 0 or s_stride <= 0 or h_stride <= 0 or d_stride <= 0:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " strides must all be positive, got (",
            b_stride,
            ", ",
            s_stride,
            ", ",
            h_stride,
            ", ",
            d_stride,
            ")",
        )
    if d_stride != 1:
        raise Error(
            "fa4 strided qkv: ", name, " d_stride must be 1, got ", d_stride
        )
    if h_stride != head_dim:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " h_stride must be ",
            head_dim,
            ", got ",
            h_stride,
        )
    if s_stride < nheads * head_dim:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " s_stride ",
            s_stride,
            " must be >= nheads * head_dim = ",
            nheads * head_dim,
        )
    if b_stride != seqlen * s_stride:
        raise Error(
            "fa4 strided qkv: ",
            name,
            " b_stride ",
            b_stride,
            " must equal seqlen * s_stride = ",
            seqlen * s_stride,
        )
    if addr % 16 != 0:
        raise Error(
            "fa4 strided qkv: ", name, " base address must be 16-byte aligned"
        )
    # TMA global strides are byte strides and every non-innermost one
    # must be a 16-byte multiple (bf16: 2 bytes per element).
    if (
        (b_stride * 2) % 16 != 0
        or (s_stride * 2) % 16 != 0
        or (h_stride * 2) % 16 != 0
    ):
        raise Error(
            "fa4 strided qkv: ",
            name,
            " non-innermost strides must be multiples of 16 bytes, got (",
            b_stride,
            ", ",
            s_stride,
            ", ",
            h_stride,
            ") elements",
        )


def _check_strided_qkv_args(
    batch: Int,
    seqlen: Int,
    nheads: Int,
) raises:
    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        raise Error(
            "fa4 strided qkv: batch, seqlen and nheads must be positive,",
            " got (",
            batch,
            ", ",
            seqlen,
            ", ",
            nheads,
            ")",
        )
    if seqlen % 128 != 0:
        raise Error(
            "fa4 strided qkv: seqlen must be a multiple of 128, got ", seqlen
        )


def flash_attention_fwd_bf16_d64_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Zero-copy fwd: Q/K/V are strided (B, S, H, 64) views described
    by per-tensor runtime element strides (b, s, h, d); out/lse keep
    the contiguous layouts of the dense entry point."""
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var lse_addr = Int(py=args[16])
    var batch = Int(py=args[17])
    var seqlen = Int(py=args[18])
    var nheads = Int(py=args[19])
    var softmax_scale = Float32(py=args[20])
    var ctx_addr = Int(py=args[21])

    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
        64,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
        64,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
        64,
    )

    launch_fwd_fa4[
        DType.bfloat16,
        64,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
        q_b_stride=q_b_stride,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    return PythonObject(None)


def flash_attention_fwd_f16_d64_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same zero-copy strided fwd as the bf16 entry point above (see its
    docstring for the layout contract), only the comptime ``dtype`` differs.
    """
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var lse_addr = Int(py=args[16])
    var batch = Int(py=args[17])
    var seqlen = Int(py=args[18])
    var nheads = Int(py=args[19])
    var softmax_scale = Float32(py=args[20])
    var ctx_addr = Int(py=args[21])

    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
        64,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
        64,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
        64,
    )

    launch_fwd_fa4[
        DType.float16,
        64,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
        q_b_stride=q_b_stride,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    return PythonObject(None)


def _check_bhsd_args(
    batch: Int,
    seqlen: Int,
    nheads: Int,
    q_addr: Int,
    k_addr: Int,
    v_addr: Int,
    out_addr: Int,
) raises:
    """Defensive validation for the BHSD-native forward entry points.

    The Python bridge (``_fa4_bhsd_layout`` in ``aten_fast.py``) has
    already gated on public (B, H, S, D) contiguity and 16-byte
    base-pointer alignment before selecting this path -- TMA descriptor
    creation over the plane-viewed (B*H, S, D) layout requires exactly
    that. Re-check here (same spirit as ``_check_strided_qkv_layout``
    for the strided ABI) so a caller that bypasses the Python gate
    cannot slip an unaligned or degenerate view past descriptor
    creation.
    """
    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        raise Error(
            "fa4 bhsd: batch, seqlen and nheads must be positive, got (",
            batch,
            ", ",
            seqlen,
            ", ",
            nheads,
            ")",
        )
    if (
        q_addr % 16 != 0
        or k_addr % 16 != 0
        or v_addr % 16 != 0
        or out_addr % 16 != 0
    ):
        raise Error(
            "fa4 bhsd: q/k/v/out base addresses must be 16-byte aligned"
        )


def flash_attention_fwd_bf16_d64_causal_bhsd(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Dense causal d64 fwd, BHSD-native: Q/K/V/O TMA descriptors address
    the PUBLIC contiguous (B, H, S, D) layout directly (viewed as
    (B*H, S, D) planes), skipping the BTHD materialization the dense
    entry point above requires. See fa4_fwd_kernel.mojo/fa4_fwd_launch.mojo
    ``bhsd``/``bhsd_qkv`` comptime params.

    Runtime-gated (phase 2c, ``_fa4_bhsd_selfload_waves``) between two
    d64 geometries: the self-loading single-warpgroup, 3-CTAs/SM kernel
    (``fa4_fwd_selfload_launch.mojo``) once there is enough parallel work
    to fill a third CTA everywhere, else the phase-2b 2-CTAs/SM producer/
    consumer kernel this bridge always used before. See
    ``_FA4_SELFLOAD_MIN_WAVES`` above for the threshold and its source."""
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    _check_bhsd_args(batch, seqlen, nheads, q_addr, k_addr, v_addr, out_addr)

    if (
        _fa4_bhsd_selfload_waves(batch, seqlen, nheads, ctx_addr)
        >= _FA4_SELFLOAD_MIN_WAVES
    ):
        launch_fwd_fa4_selfload[
            DType.bfloat16,
            64,
            False,
            True,
            1,
            0,
        ](
            batch,
            seqlen,
            nheads,
            softmax_scale,
            q_addr,
            k_addr,
            v_addr,
            out_addr,
            lse_addr,
            0,
            ctx_addr,
        )
    else:
        launch_fwd_fa4[
            DType.bfloat16,
            64,
            False,
            True,
            1,
            False,
            False,
            False,
            0,
            bhsd_qkv=True,
        ](
            batch,
            seqlen,
            nheads,
            softmax_scale,
            q_addr,
            k_addr,
            v_addr,
            out_addr,
            lse_addr,
            0,
            ctx_addr,
        )
    return PythonObject(None)


def flash_attention_fwd_f16_d64_causal_bhsd(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same BHSD-native dense causal d64 fwd as the bf16 entry point
    above, only the comptime ``dtype`` differs (including the phase-2c
    self-load/phase-2b wave gate)."""
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    _check_bhsd_args(batch, seqlen, nheads, q_addr, k_addr, v_addr, out_addr)

    if (
        _fa4_bhsd_selfload_waves(batch, seqlen, nheads, ctx_addr)
        >= _FA4_SELFLOAD_MIN_WAVES
    ):
        launch_fwd_fa4_selfload[
            DType.float16,
            64,
            False,
            True,
            1,
            0,
        ](
            batch,
            seqlen,
            nheads,
            softmax_scale,
            q_addr,
            k_addr,
            v_addr,
            out_addr,
            lse_addr,
            0,
            ctx_addr,
        )
    else:
        launch_fwd_fa4[
            DType.float16,
            64,
            False,
            True,
            1,
            False,
            False,
            False,
            0,
            bhsd_qkv=True,
        ](
            batch,
            seqlen,
            nheads,
            softmax_scale,
            q_addr,
            k_addr,
            v_addr,
            out_addr,
            lse_addr,
            0,
            ctx_addr,
        )
    return PythonObject(None)


def flash_attention_bwd_bf16_d64_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Zero-copy bwd: Q/K/V are strided (B, S, H, 64) views described
    by per-tensor runtime element strides (b, s, h, d); out/dout/lse,
    the dq/dk/dv outputs and all scratch keep the contiguous layouts
    of the dense entry point."""
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var dout_addr = Int(py=args[16])
    var lse_addr = Int(py=args[17])
    var dq_addr = Int(py=args[18])
    var dk_addr = Int(py=args[19])
    var dv_addr = Int(py=args[20])
    var dpsum_addr = Int(py=args[21])
    var lse_log2_addr = Int(py=args[22])
    var dq_accum_addr = Int(py=args[23])
    var batch = Int(py=args[24])
    var seqlen = Int(py=args[25])
    var nheads = Int(py=args[26])
    var softmax_scale = Float32(py=args[27])
    var ctx_addr = Int(py=args[28])

    # The whole layout contract is validated up front so preprocess
    # never launches for an unsupported layout.
    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
        64,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
        64,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
        64,
    )

    launch_bwd_preprocess[
        DType.bfloat16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.bfloat16,
        64,
        False,
        True,
        1,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    launch_bwd_convert[
        DType.bfloat16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_bwd_f16_d64_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same zero-copy strided bwd as the bf16 entry point above (see its
    docstring for the layout contract), only the comptime ``dtype`` differs.
    """
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var dout_addr = Int(py=args[16])
    var lse_addr = Int(py=args[17])
    var dq_addr = Int(py=args[18])
    var dk_addr = Int(py=args[19])
    var dv_addr = Int(py=args[20])
    var dpsum_addr = Int(py=args[21])
    var lse_log2_addr = Int(py=args[22])
    var dq_accum_addr = Int(py=args[23])
    var batch = Int(py=args[24])
    var seqlen = Int(py=args[25])
    var nheads = Int(py=args[26])
    var softmax_scale = Float32(py=args[27])
    var ctx_addr = Int(py=args[28])

    # The whole layout contract is validated up front so preprocess
    # never launches for an unsupported layout.
    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
        64,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
        64,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
        64,
    )

    launch_bwd_preprocess[
        DType.float16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.float16,
        64,
        False,
        True,
        1,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    launch_bwd_convert[
        DType.float16, 64, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


# ---------------------------------------------------------------------------
# head_dim=128 forward/backward entry points. Structurally identical to the
# d64 family above (dense, strided_qkv zero-copy BTHD, bhsd-native) --
# fa4_fwd_common.mojo / fa4_bwd_common.mojo already carry BM, warpgroup
# count, register budgets and the bwd causal tile_m as head_dim-parametric
# constants mirroring FA4's own sm90 configs (BM=128 not 192, 2 MMA
# warpgroups not 3, bwd causal tile_m=64 not 128), so only the comptime
# head_dim argument to launch_fwd_fa4/launch_bwd_* differs from the d64
# twins below.
# ---------------------------------------------------------------------------


def flash_attention_fwd_bf16_d128_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_fwd_fa4[
        DType.bfloat16,
        128,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_fwd_f16_d128_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same dense d128 causal forward as the bf16 entry point above, only the
    comptime ``dtype`` differs. The f16 RS (register-A) wgmma emitters live
    in ``fa4_wgmma_f16.mojo`` and are selected inside the shared kernel by
    ``comptime if dtype == DType.float16`` -- bf16 keeps the stdlib path
    byte-identical."""
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_fwd_fa4[
        DType.float16,
        128,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_bwd_bf16_d128_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var dout_addr = Int(py=args[4])
    var lse_addr = Int(py=args[5])
    var dq_addr = Int(py=args[6])
    var dk_addr = Int(py=args[7])
    var dv_addr = Int(py=args[8])
    var dpsum_addr = Int(py=args[9])
    var lse_log2_addr = Int(py=args[10])
    var dq_accum_addr = Int(py=args[11])
    var batch = Int(py=args[12])
    var seqlen = Int(py=args[13])
    var nheads = Int(py=args[14])
    var softmax_scale = Float32(py=args[15])
    var ctx_addr = Int(py=args[16])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_bwd_preprocess[
        DType.bfloat16, 128, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.bfloat16, 128, False, True, 1, False, False, 0
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
    )
    launch_bwd_convert[
        DType.bfloat16, 128, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_bwd_f16_d128_causal(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same dense d128 causal backward as the bf16 entry point above, only
    the comptime ``dtype`` differs (see ``flash_attention_fwd_f16_d128_causal``
    for the f16 RS wgmma note)."""
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var dout_addr = Int(py=args[4])
    var lse_addr = Int(py=args[5])
    var dq_addr = Int(py=args[6])
    var dk_addr = Int(py=args[7])
    var dv_addr = Int(py=args[8])
    var dpsum_addr = Int(py=args[9])
    var lse_log2_addr = Int(py=args[10])
    var dq_accum_addr = Int(py=args[11])
    var batch = Int(py=args[12])
    var seqlen = Int(py=args[13])
    var nheads = Int(py=args[14])
    var softmax_scale = Float32(py=args[15])
    var ctx_addr = Int(py=args[16])

    if batch <= 0 or seqlen <= 0 or nheads <= 0:
        return PythonObject(None)

    launch_bwd_preprocess[
        DType.float16, 128, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.float16, 128, False, True, 1, False, False, 0
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
    )
    launch_bwd_convert[
        DType.float16, 128, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_fwd_bf16_d128_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Zero-copy fwd: Q/K/V are strided (B, S, H, 128) views described
    by per-tensor runtime element strides (b, s, h, d); out/lse keep
    the contiguous layouts of the dense entry point."""
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var lse_addr = Int(py=args[16])
    var batch = Int(py=args[17])
    var seqlen = Int(py=args[18])
    var nheads = Int(py=args[19])
    var softmax_scale = Float32(py=args[20])
    var ctx_addr = Int(py=args[21])

    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
        128,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
        128,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
        128,
    )

    launch_fwd_fa4[
        DType.bfloat16,
        128,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
        q_b_stride=q_b_stride,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    return PythonObject(None)


def flash_attention_fwd_f16_d128_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same zero-copy strided fwd as the bf16 entry point above (see its
    docstring for the layout contract), only the comptime ``dtype`` differs.
    """
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var lse_addr = Int(py=args[16])
    var batch = Int(py=args[17])
    var seqlen = Int(py=args[18])
    var nheads = Int(py=args[19])
    var softmax_scale = Float32(py=args[20])
    var ctx_addr = Int(py=args[21])

    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
        128,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
        128,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
        128,
    )

    launch_fwd_fa4[
        DType.float16,
        128,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
        q_b_stride=q_b_stride,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    return PythonObject(None)


def flash_attention_fwd_bf16_d128_causal_bhsd(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Dense causal d128 fwd, BHSD-native: Q/K/V/O TMA descriptors address
    the PUBLIC contiguous (B, H, S, D) layout directly (viewed as
    (B*H, S, D) planes), skipping the BTHD materialization the dense
    entry point above requires. See fa4_fwd_kernel.mojo/fa4_fwd_launch.mojo
    ``bhsd``/``bhsd_qkv`` comptime params."""
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    _check_bhsd_args(batch, seqlen, nheads, q_addr, k_addr, v_addr, out_addr)

    launch_fwd_fa4[
        DType.bfloat16,
        128,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        bhsd_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_fwd_f16_d128_causal_bhsd(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same BHSD-native dense causal d128 fwd as the bf16 entry point
    above, only the comptime ``dtype`` differs."""
    var q_addr = Int(py=args[0])
    var k_addr = Int(py=args[1])
    var v_addr = Int(py=args[2])
    var out_addr = Int(py=args[3])
    var lse_addr = Int(py=args[4])
    var batch = Int(py=args[5])
    var seqlen = Int(py=args[6])
    var nheads = Int(py=args[7])
    var softmax_scale = Float32(py=args[8])
    var ctx_addr = Int(py=args[9])

    _check_bhsd_args(batch, seqlen, nheads, q_addr, k_addr, v_addr, out_addr)

    launch_fwd_fa4[
        DType.float16,
        128,
        False,
        True,
        1,
        False,
        False,
        False,
        0,
        bhsd_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        out_addr,
        lse_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_bwd_bf16_d128_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Zero-copy bwd: Q/K/V are strided (B, S, H, 128) views described
    by per-tensor runtime element strides (b, s, h, d); out/dout/lse,
    the dq/dk/dv outputs and all scratch keep the contiguous layouts
    of the dense entry point."""
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var dout_addr = Int(py=args[16])
    var lse_addr = Int(py=args[17])
    var dq_addr = Int(py=args[18])
    var dk_addr = Int(py=args[19])
    var dv_addr = Int(py=args[20])
    var dpsum_addr = Int(py=args[21])
    var lse_log2_addr = Int(py=args[22])
    var dq_accum_addr = Int(py=args[23])
    var batch = Int(py=args[24])
    var seqlen = Int(py=args[25])
    var nheads = Int(py=args[26])
    var softmax_scale = Float32(py=args[27])
    var ctx_addr = Int(py=args[28])

    # The whole layout contract is validated up front so preprocess
    # never launches for an unsupported layout.
    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
        128,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
        128,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
        128,
    )

    launch_bwd_preprocess[
        DType.bfloat16, 128, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.bfloat16,
        128,
        False,
        True,
        1,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    launch_bwd_convert[
        DType.bfloat16, 128, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


def flash_attention_bwd_f16_d128_causal_strided_qkv(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Same zero-copy strided bwd as the bf16 entry point above (see its
    docstring for the layout contract), only the comptime ``dtype`` differs.
    """
    var q_addr = Int(py=args[0])
    var q_b_stride = Int(py=args[1])
    var q_s_stride = Int(py=args[2])
    var q_h_stride = Int(py=args[3])
    var q_d_stride = Int(py=args[4])
    var k_addr = Int(py=args[5])
    var k_b_stride = Int(py=args[6])
    var k_s_stride = Int(py=args[7])
    var k_h_stride = Int(py=args[8])
    var k_d_stride = Int(py=args[9])
    var v_addr = Int(py=args[10])
    var v_b_stride = Int(py=args[11])
    var v_s_stride = Int(py=args[12])
    var v_h_stride = Int(py=args[13])
    var v_d_stride = Int(py=args[14])
    var out_addr = Int(py=args[15])
    var dout_addr = Int(py=args[16])
    var lse_addr = Int(py=args[17])
    var dq_addr = Int(py=args[18])
    var dk_addr = Int(py=args[19])
    var dv_addr = Int(py=args[20])
    var dpsum_addr = Int(py=args[21])
    var lse_log2_addr = Int(py=args[22])
    var dq_accum_addr = Int(py=args[23])
    var batch = Int(py=args[24])
    var seqlen = Int(py=args[25])
    var nheads = Int(py=args[26])
    var softmax_scale = Float32(py=args[27])
    var ctx_addr = Int(py=args[28])

    # The whole layout contract is validated up front so preprocess
    # never launches for an unsupported layout.
    _check_strided_qkv_args(batch, seqlen, nheads)
    _check_strided_qkv_layout(
        "q",
        q_addr,
        q_b_stride,
        q_s_stride,
        q_h_stride,
        q_d_stride,
        seqlen,
        nheads,
        128,
    )
    _check_strided_qkv_layout(
        "k",
        k_addr,
        k_b_stride,
        k_s_stride,
        k_h_stride,
        k_d_stride,
        seqlen,
        nheads,
        128,
    )
    _check_strided_qkv_layout(
        "v",
        v_addr,
        v_b_stride,
        v_s_stride,
        v_h_stride,
        v_d_stride,
        seqlen,
        nheads,
        128,
    )

    launch_bwd_preprocess[
        DType.float16, 128, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        out_addr,
        dout_addr,
        lse_addr,
        dpsum_addr,
        lse_log2_addr,
        dq_accum_addr,
        0,
        0,
        0,
        ctx_addr,
    )
    launch_bwd_main[
        DType.float16,
        128,
        False,
        True,
        1,
        False,
        False,
        0,
        strided_qkv=True,
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        q_addr,
        k_addr,
        v_addr,
        dout_addr,
        dk_addr,
        dv_addr,
        lse_log2_addr,
        dpsum_addr,
        dq_accum_addr,
        0,
        ctx_addr,
        q_s_stride=q_s_stride,
        q_h_stride=q_h_stride,
        q_d_stride=q_d_stride,
        k_s_stride=k_s_stride,
        k_h_stride=k_h_stride,
        k_d_stride=k_d_stride,
        v_s_stride=v_s_stride,
        v_h_stride=v_h_stride,
        v_d_stride=v_d_stride,
    )
    launch_bwd_convert[
        DType.float16, 128, False, True, 1, False
    ](
        batch,
        seqlen,
        nheads,
        softmax_scale,
        dq_accum_addr,
        dq_addr,
        0,
        ctx_addr,
    )
    return PythonObject(None)


@export
def PyInit_fa4_ops() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("fa4_ops")
        module.def_py_function[flash_attention_fwd_bf16_d64_causal](
            "flash_attention_fwd_bf16_d64_causal"
        )
        module.def_py_function[flash_attention_fwd_f16_d64_causal](
            "flash_attention_fwd_f16_d64_causal"
        )
        module.def_py_function[flash_attention_bwd_bf16_d64_causal](
            "flash_attention_bwd_bf16_d64_causal"
        )
        module.def_py_function[flash_attention_bwd_f16_d64_causal](
            "flash_attention_bwd_f16_d64_causal"
        )
        module.def_py_function[flash_attention_fwd_bf16_d64_causal_strided_qkv](
            "flash_attention_fwd_bf16_d64_causal_strided_qkv"
        )
        module.def_py_function[flash_attention_fwd_f16_d64_causal_strided_qkv](
            "flash_attention_fwd_f16_d64_causal_strided_qkv"
        )
        module.def_py_function[flash_attention_fwd_bf16_d64_causal_bhsd](
            "flash_attention_fwd_bf16_d64_causal_bhsd"
        )
        module.def_py_function[flash_attention_fwd_f16_d64_causal_bhsd](
            "flash_attention_fwd_f16_d64_causal_bhsd"
        )
        module.def_py_function[flash_attention_bwd_bf16_d64_causal_strided_qkv](
            "flash_attention_bwd_bf16_d64_causal_strided_qkv"
        )
        module.def_py_function[flash_attention_bwd_f16_d64_causal_strided_qkv](
            "flash_attention_bwd_f16_d64_causal_strided_qkv"
        )
        module.def_py_function[flash_attention_fwd_bf16_d128_causal](
            "flash_attention_fwd_bf16_d128_causal"
        )
        module.def_py_function[flash_attention_fwd_f16_d128_causal](
            "flash_attention_fwd_f16_d128_causal"
        )
        module.def_py_function[flash_attention_bwd_bf16_d128_causal](
            "flash_attention_bwd_bf16_d128_causal"
        )
        module.def_py_function[flash_attention_bwd_f16_d128_causal](
            "flash_attention_bwd_f16_d128_causal"
        )
        module.def_py_function[flash_attention_fwd_bf16_d128_causal_strided_qkv](
            "flash_attention_fwd_bf16_d128_causal_strided_qkv"
        )
        module.def_py_function[flash_attention_fwd_f16_d128_causal_strided_qkv](
            "flash_attention_fwd_f16_d128_causal_strided_qkv"
        )
        module.def_py_function[flash_attention_fwd_bf16_d128_causal_bhsd](
            "flash_attention_fwd_bf16_d128_causal_bhsd"
        )
        module.def_py_function[flash_attention_fwd_f16_d128_causal_bhsd](
            "flash_attention_fwd_f16_d128_causal_bhsd"
        )
        module.def_py_function[flash_attention_bwd_bf16_d128_causal_strided_qkv](
            "flash_attention_bwd_bf16_d128_causal_strided_qkv"
        )
        module.def_py_function[flash_attention_bwd_f16_d128_causal_strided_qkv](
            "flash_attention_bwd_f16_d128_causal_strided_qkv"
        )
        return module.finalize()
    except error:
        abort(String("failed to create FA4 Python module: ", error))
