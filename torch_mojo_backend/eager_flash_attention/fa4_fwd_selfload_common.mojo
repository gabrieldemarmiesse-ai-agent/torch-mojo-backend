# ============================================================================
# PHASE 2c: self-loading single-warpgroup CTA, d64 dense-causal BHSD ONLY.
# A NEW module (not a replacement): fa4_fwd_common.mojo (phase 2b) still
# serves head_dim=128 and, below the wave-gate threshold in fa4_ops.mojo,
# the few-wave d64 shapes too. See NOTES.md "Phase 2c" (agent A's harness,
# /scratch/fa4-fwd-harness-2c) for the measurements this module encodes.
# ============================================================================
"""Comptime constants for the self-loading FA4 fwd kernel (phase 2c).

The phase-2b profile at P1 (B32 H12 S1024) says the d64 kernel is
latency-bound, not pipe- or bandwidth-bound (tensor 36%, XU 39%, L2 42%,
issue 44% of peak) with `wait` at 1.31 warps per issue-active cycle and
only TWO consumer warps per SMSP (2 CTAs/SM x 1 MMA warpgroup). Phase 2
measured the same lever from the other side: dropping to 1 CTA/SM costs
55%. So the target is a THIRD CTA per SM.

At phase 2b's 256-thread CTA (1 MMA warpgroup + 1 producer warpgroup)
that is impossible: `nvvm.minctasm`=3 gives ptxas a static budget of
65536/(3*256) = 80 registers per thread and it refuses outright
("Insufficient registers (80) ... try 90 or higher") -- setmaxnreg
redistributes registers BETWEEN warps at runtime, it does not lift the
CTA's static allocation.

This module's kernel therefore drops the producer warpgroup: the CTA is
ONE warpgroup of 128 threads that issues its own TMA loads (see the
PREFETCH invariant comment in fa4_fwd_selfload_kernel.mojo for the
correctness argument this enables). 65536/(3*128) = 170 -> 168 registers
per thread, comfortably above the ~155 the consumer needs, and 4 ring
stages keep smem at 8 KiB Q + 4*16 KiB = 72 KiB, so 3*72 = 216 KiB fits
the 227 KiB opt-in cap.

Swept on H100 PCIe at 1395 MHz (phase 2c, NOTES.md): 3 CTAs/SM with a
4-slot ring is the optimum measured there. 2 CTAs/SM with a 6-slot ring
(the phase-2b ring depth, same self-loading body) reads 268 us at P1
against 230 for the 3-CTA/4-slot config -- the third CTA is the whole
win, and 4 slots is the deepest ring that still fits three CTAs in
227 KiB. FITTED ON H100 PCIe (114 SMs, 227 KiB smem opt-in cap, 64 K
32-bit registers per SM) -- re-derive both constants on any other card
before trusting them there.
"""

comptime FA_SELFLOAD_STAGES: Int = 4
comptime FA_SELFLOAD_CTAS: Int = 3


def kFa4BlockM(head_dim: Int) -> Int:
    return 64 if head_dim == 64 else 128


comptime kFa4BlockN: Int = 128


def kFa4NMmaWarpgroups(head_dim: Int) -> Int:
    return 1


def kFa4NThreads(head_dim: Int) -> Int:
    return 128


def kFa4CtasPerSm(head_dim: Int) -> Int:
    return FA_SELFLOAD_CTAS


# K/V shared-memory ring: K(n) lives in slot (2n) % kFa4KVStages, V(n) in
# slot (2n+1) % kFa4KVStages -- same incremental scheme as the phase-2b
# ring, just 4 slots deep instead of 6 (the deepest ring 3 CTAs/SM fits
# in the 227 KiB opt-in cap at 128 threads; see the module docstring).
comptime kFa4KVStages: Int = FA_SELFLOAD_STAGES
