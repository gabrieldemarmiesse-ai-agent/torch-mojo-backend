# ===----------------------------------------------------------------------=== #
# Self-load race soak, CI-lite (ported from agent A's phase-2c harness,
# /scratch/fa4-fwd-harness-2c/soak_v7.mojo -- the FULL soak with more
# shapes, more reps, and multi-clock-pin coverage stays in that harness;
# this is the permanent-suite regression guard, run by
# tests/test_fa4_selfload_soak.py).
#
# The self-loading kernel (fa4_fwd_selfload_kernel.mojo) deletes the
# empty[] mbarriers the phase-2b producer/consumer kernel used to prove a
# slot's smem read had retired before the next TMA refill into that same
# slot; the substitute proof is `wgmma.wait_group` plus the PREFETCH
# invariant (see the comment next to PREFETCH's definition in that file).
# A violation would be TIMING-dependent, so a single pass is weak
# evidence. This soak attacks it the same way the harness did:
#
#   * many iterations per shape (deterministic inputs -> the output must
#     be BITWISE identical every rep; any bit that moves is
#     nondeterminism, which for this kernel can only be a race),
#   * bursts of back-to-back launches with NO sync between them, so CTAs
#     of consecutive kernels are co-resident and warp scheduling gets
#     perturbed,
#   * L2-resident K/V (same buffers reused across the whole burst) so the
#     refill TMA lands as fast as the hardware can make it -- the worst
#     case for a write-after-read window,
#   * every rep also gets a tolerance check against the phase-2b
#     producer/consumer kernel (structurally different, same math), so a
#     corruption that happens to be bitwise-stable across reps is still
#     caught.
#
# Build (never under the GPU lock):
#   <mojo> build tests/fa4_selfload_soak_probe.mojo \
#       -I torch_mojo_backend/eager_flash_attention -o <out>
# Run (real GPU; no flock needed for a correctness probe -- see
# AGENTS.md, the lock is for benchmark timing precision only):
#   REPS=8 ./<out>
# ===----------------------------------------------------------------------=== #

from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory.unsafe_pointer import pointer_to_int

from fa4_fwd_launch import launch_fwd_fa4
from fa4_fwd_selfload_launch import launch_fwd_fa4_selfload

comptime D: Int = 64
comptime BURST: Int = 4


def _hash_unit(idx: Int, seed: Int) -> Float32:
    """Deterministic pseudo-random value in [-0.5, 0.5) for (idx, seed)."""
    var h = (UInt64(idx) + 1) * 0x9E3779B97F4A7C15 + UInt64(
        seed
    ) * 0xC2B2AE3D27D4EB4F
    h = (h ^ (h >> 29)) * 0xBF58476D1CE4E5B9
    h = h ^ (h >> 32)
    return Float32(Int(h & 0xFFFF)) / 65536.0 - 0.5


def fill_qkv(
    dst: DeviceBuffer[DType.bfloat16], total: Int, seed: Int, ctx: DeviceContext
) raises:
    """Host-filled, deterministic-hash, then copied to device -- a
    correctness probe has no need for a GPU fill kernel."""
    var host = ctx.enqueue_create_host_buffer[DType.bfloat16](total)
    ctx.synchronize()
    for i in range(total):
        host[i] = _hash_unit(i, seed).cast[DType.bfloat16]()
    ctx.enqueue_copy(dst, host)


def _soak_shape(
    batch: Int,
    heads: Int,
    seq: Int,
    reps: Int,
    ctx: DeviceContext,
    ctx_addr: Int,
) raises -> Int:
    var total = batch * heads * seq * D
    var lse_total = batch * heads * seq
    var scale = Float32(0.125)

    var q_b = ctx.enqueue_create_buffer[DType.bfloat16](total)
    var k_b = ctx.enqueue_create_buffer[DType.bfloat16](total)
    var v_b = ctx.enqueue_create_buffer[DType.bfloat16](total)
    var o_ref = ctx.enqueue_create_buffer[DType.bfloat16](total)
    var lse_ref = ctx.enqueue_create_buffer[DType.float32](lse_total)
    var o_a = ctx.enqueue_create_buffer[DType.bfloat16](total)
    var o_c = ctx.enqueue_create_buffer[DType.bfloat16](total)
    var lse_a = ctx.enqueue_create_buffer[DType.float32](lse_total)
    var lse_c = ctx.enqueue_create_buffer[DType.float32](lse_total)

    var s0 = 1000 + batch * 31 + heads * 7 + seq
    fill_qkv(q_b, total, s0, ctx)
    fill_qkv(k_b, total, s0 + 1, ctx)
    fill_qkv(v_b, total, s0 + 2, ctx)
    ctx.synchronize()

    # Phase-2b reference leg: structurally different (producer
    # warpgroup + empty[] barriers), same math -- a corruption that
    # happens to be bitwise-stable across self-load reps still shows up
    # against this leg.
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
        seq,
        heads,
        scale,
        Int(q_b.unsafe_ptr()),
        Int(k_b.unsafe_ptr()),
        Int(v_b.unsafe_ptr()),
        Int(o_ref.unsafe_ptr()),
        Int(lse_ref.unsafe_ptr()),
        0,
        ctx_addr,
    )
    ctx.synchronize()

    var host_ref = ctx.enqueue_create_host_buffer[DType.bfloat16](total)
    ctx.enqueue_copy(host_ref, o_ref)
    var host_gold = ctx.enqueue_create_host_buffer[DType.bfloat16](total)
    var host_x = ctx.enqueue_create_host_buffer[DType.bfloat16](total)
    ctx.synchronize()

    var launches = 0
    var bit_mismatch = 0
    var tol_mismatch = 0

    for rep in range(reps):
        # BURST back-to-back launches, no sync in between: CTAs of
        # consecutive kernels overlap on the SMs.
        for j in range(BURST):
            var o_ptr = Int(o_a.unsafe_ptr()) if (j & 1) == 0 else Int(
                o_c.unsafe_ptr()
            )
            var l_ptr = Int(lse_a.unsafe_ptr()) if (j & 1) == 0 else Int(
                lse_c.unsafe_ptr()
            )
            launch_fwd_fa4_selfload[
                DType.bfloat16,
                64,
                False,
                True,
                1,
                0,
            ](
                batch,
                seq,
                heads,
                scale,
                Int(q_b.unsafe_ptr()),
                Int(k_b.unsafe_ptr()),
                Int(v_b.unsafe_ptr()),
                o_ptr,
                l_ptr,
                0,
                ctx_addr,
            )
            launches += 1

        for which in range(2):
            if which == 0:
                ctx.enqueue_copy(host_x, o_a)
            else:
                ctx.enqueue_copy(host_x, o_c)
            ctx.synchronize()
            if rep == 0 and which == 0:
                for i in range(total):
                    host_gold[i] = host_x[i]
            else:
                for i in range(total):
                    if host_gold[i] != host_x[i]:
                        bit_mismatch += 1
                        if bit_mismatch <= 3:
                            print(
                                "  BIT MISMATCH rep",
                                rep,
                                "buf",
                                which,
                                "flat",
                                i,
                                ":",
                                host_gold[i],
                                "vs",
                                host_x[i],
                            )
            for i in range(total):
                var a = host_ref[i].cast[DType.float64]()
                var b = host_x[i].cast[DType.float64]()
                if abs(a - b) > 0.05:
                    tol_mismatch += 1
                    if tol_mismatch <= 3:
                        print(
                            "  TOL MISMATCH rep",
                            rep,
                            "flat",
                            i,
                            ":",
                            host_ref[i],
                            "vs",
                            host_x[i],
                        )

    print(
        "  B",
        batch,
        "H",
        heads,
        "S",
        seq,
        "| self-load launches",
        launches,
        "| checked",
        2 * reps,
        "| bit-mismatch",
        bit_mismatch,
        "| tol-mismatch",
        tol_mismatch,
    )

    _ = q_b^
    _ = k_b^
    _ = v_b^
    _ = o_ref^
    _ = lse_ref^
    _ = o_a^
    _ = o_c^
    _ = lse_a^
    _ = lse_c^
    _ = host_ref^
    _ = host_gold^
    _ = host_x^
    return bit_mismatch + tol_mismatch


def main() raises:
    var ctx = DeviceContext()
    var ctx_addr = pointer_to_int(ctx._handle)
    var reps_env = getenv("REPS")
    var reps = 8 if reps_env.byte_length() == 0 else Int(reps_env)
    print("self-load soak-lite: REPS =", reps, "BURST =", BURST)

    var bad = 0
    # Two shapes: enough kv trips to wrap the 4-slot ring several times,
    # one round B/H count and one awkward one (matches the harness's P4
    # regime -- the shape closest to the self-load kernel's own
    # break-even wave count, so its ring runs the shallowest margin).
    bad += _soak_shape(2, 4, 1024, reps, ctx, ctx_addr)
    bad += _soak_shape(3, 5, 1152, reps, ctx, ctx_addr)
    if bad != 0:
        raise Error(String("SOAK FAILED: ", bad, " mismatches"))
    print("SOAK PASS")
