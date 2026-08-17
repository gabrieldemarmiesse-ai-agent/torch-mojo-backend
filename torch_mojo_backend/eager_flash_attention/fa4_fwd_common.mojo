"""V2 comptime constants for the FA4 fwd kernel (phase 2 deliverable).

head_dim 64 moves from tile 192x128 (3 MMA warpgroups + producer, 512
threads, setmaxnreg 32/160, 1 CTA/SM) to tile 64x128 (1 MMA warpgroup +
producer, 256 threads, setmaxnreg 24/232, TWO CTAs PER SM via
`nvvm.minctasm`=2 — the pool 2*(128*232 + 128*24) = 65536 is exactly the
register file).

WHY (measured, see NOTES.md phase 2): causal tile-quantization
overcompute. BM=192 > BN=128 makes the diagonal band span two 128-col
tiles per 192 rows: at S=1024 the kernel computes 1.500x the causal
triangle vs 1.125x at BM=64 (cuDNN's 66.7us P0 kernel runs 64x128 at
1.125x). The second CTA per SM replaces the third warpgroup's latency
hiding (a 2-WG BM=128 single-CTA variant measured WORSE per unit: each
WG's ~850-cycle softmax overruns its 512-cycle tensor slot).

head_dim 128 keeps the production config (tile 128x128, 2 MMA
warpgroups + producer, 384 threads, 24/240) — evaluations must stay
byte-identical.

SCOPE: these constants are a pure function of head_dim and feed every
d64 instantiation in fa4_ops.mojo — dense causal, the strided-QKV ABI,
and the BHSD-native path all share this one kernel body, so all three
move to BM=64/1-warpgroup/2-CTAs-per-SM together (verified bitwise
against the BM=192 kernel on 11 edge shapes, benchmarked on the
dense/bhsd routes). `varlen` and `window` are comptime parameters on
`fwd_fa4_kernel`/`launch_fwd_fa4` but, as of this change, no entry
point in fa4_ops.mojo instantiates either at ANY head_dim — there is
no shipped varlen/window route today for these constants to affect.
The causal mask arms most sensitive to BM (see the diagonal-predicate
comment in fa4_fwd_kernel.mojo) already fall back to the pre-existing,
unoptimized-but-safe `kv_trips <= 2` rule whenever `varlen or window`,
so a future varlen/window entry point inherits a correct (if
unaccelerated) mask for free; re-measure it before relying on its
performance, and re-run this file's edge-shape tests before trusting
its BM=64 tiling once one is added.
"""


def kFa4BlockM(head_dim: Int) -> Int:
    return 64 if head_dim == 64 else 128


comptime kFa4BlockN: Int = 128


def kFa4NMmaWarpgroups(head_dim: Int) -> Int:
    return 1 if head_dim == 64 else 2


def kFa4NThreads(head_dim: Int) -> Int:
    return (kFa4NMmaWarpgroups(head_dim) + 1) * 128


def kFa4ProducerRegs(head_dim: Int) -> Int:
    return 24


def kFa4ConsumerRegs(head_dim: Int) -> Int:
    return 232 if head_dim == 64 else 240


# K/V shared-memory ring: K(n) lives in slot (2n) % kFa4KVStages,
# V(n) in slot (2n+1) % kFa4KVStages. (Ring depth is not the limiter:
# 8 stages measured within noise of 6 at BM=128; 6 fits 2 CTAs/SM at
# BM=64: 8 KiB Q + 96 KiB ring + mbar < 113.5 KiB per CTA.)
comptime kFa4KVStages: Int = 6
