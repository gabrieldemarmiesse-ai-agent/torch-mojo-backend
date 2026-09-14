"""Persistent bf16 GEMM with rolling TMA ring stage and phase counters.

The producer and consumer advance explicit counters across all output
work, avoiding division/modulo by the non-power-of-two stage count; both
counters span output-work boundaries, so the barrier sequence is the
parent's with the repeated stage-index arithmetic removed.

Derived from upstream gemm16_nn_v4_kernels.mojo.  Only the scalar-pair
shared-store loop changes: four 8x8 matrices are packed per st.matrix,
using the original BM-high, 64-column swizzled TMA boxes.  The existing
pipeline, layouts, consumer barriers, C descriptor, and TMA-store launches
remain unchanged.  The upstream col_a/kmaj_b/ragged_n parameters remain
available, so the candidate serves NN/TN/NT/TT with runtime M/N/K.

Tuning provenance: BK=64, raster group=4, and the parent's intended
BM=128/BN=256/stages=3/cluster_m=2/consumers=2 regime are the upstream
H100-tuned configuration.  Problem dimensions are never compile-time
constants.  The exported enqueuer has the upstream generic signature;
its caller is responsible for the existing SM90/TMA regime guards.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.compute.mma import (
    st_matrix,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import (
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from std.memory import AddressSpace
from max.gpu.sync import named_barrier
from max.gpu.primitives import (
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
)
from std.memory import bitcast, stack_allocation
from std.sys import size_of
from std.sys.info import _has_sm_9x, _is_sm_9x
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple

from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    TensorCoreAsync,
    _convert_cfrags_to_simd,
    _convert_cfrags_to_tuple,
    _wgmma_descriptor,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
    warpgroup_fence,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from std.sys import get_defined_bool, get_defined_int
from gemm16_dtype import _GEMM16_DT, _GEMM16_TAG

from op_utils import _enqueue_cached

from gemm16_nn_v4_kernels import (
    _V4_DT,
    _V4_F32,
    _V4_PTR,
    _V4_BK,
    _V4_SWIZZLE,
    _v4_dyn_smem_tile,
    _v4_persistent_smem_bytes,
    _v4_mma_tile,
    _v4_persistent_layout_tag,
    _v4_persistent_ragged_tag,
)

# H100 cache-reuse experiment: macro rows grouped before advancing N.
# Runtime M/N/K still determine the entire work census.
comptime _V4_GROUP = get_defined_int["TUNE_GROUP", 4]()

# Named here only so the launch cache key can spell the build identity of the
# epilogue: the selected instruction sequence differs between the two values,
# so two builds of this family must not share one cached DeviceFunction.
comptime _ROLL_PAIR_CAST = get_defined_bool["PAIR_CAST", False]()


@always_inline
def _pack_accum_pair(x: Float32, y: Float32) -> Float32:
    # Optional instruction-selection experiment, retaining identical rounding.
    comptime if get_defined_bool["PAIR_CAST", False]():
        return bitcast[DType.float32, 1](
            SIMD[DType.float32, 2](x, y).cast[_V4_DT]()
        )
    else:
        return bitcast[DType.float32, 1](
            SIMD[_V4_DT, 2](x.cast[_V4_DT](), y.cast[_V4_DT]())
        )


@always_inline
def _store_accum_bm_boxes_stmatrix[
    bm: Int, bn: Int
](
    wg_half: Pointer[
        Scalar[_V4_DT], MutAnyOrigin, address_space=AddressSpace.SHARED
    ],
    accum: LayoutTensor[
        _V4_F32,
        Layout.row_major(1, 64 * bn // 128),
        MutAnyOrigin,
        address_space=AddressSpace.LOCAL,
    ],
    warp: Int,
    lane: Int,
    warp_group_idx: Int,
):
    """Store one warp group's WGMMA accumulator fragments into the 128B-
    swizzled TMA staging tile using `st.matrix.x4` (bn // 16 instructions
    per thread instead of bn // 4 scalar pair stores).

    For instruction t, matrix j holds fragment pair q = 4t + j: row half
    j % 2, column block 2t + j // 2.  Lane group l // 8 supplies the
    address of matrix (l // 8), row l % 8, which this function maps through
    the canonical SWIZZLE_128B 64x64 box layout.
    """
    comptime CFRAG = 64 * bn // 128
    var mi = lane // 8
    # Every consumer owns 64 rows in the SAME BM-high C box.  In contrast
    # to NT's separate consumer slices, successive column boxes stride by
    # bm*64 and each consumer's rows begin at (warp_group_idx-1)*64.
    var row = (warp_group_idx - 1) * 64 + warp * 16 + (lane % 8) + 8 * (mi % 2)
    var row_base = row * 64
    var row_mod = row % 8
    var c0 = mi // 2
    comptime for t in range(CFRAG // 8):
        var col = 16 * t + 8 * c0
        var off = (
            (col // 64) * (bm * 64)
            + row_base
            + (((col % 64) // 8) ^ row_mod) * 8
        )
        var data = SIMD[DType.float32, 4](
            _pack_accum_pair(
                accum.ptr[unsafe_offset=8 * t],
                accum.ptr[unsafe_offset=8 * t + 1],
            ),
            _pack_accum_pair(
                accum.ptr[unsafe_offset=8 * t + 2],
                accum.ptr[unsafe_offset=8 * t + 3],
            ),
            _pack_accum_pair(
                accum.ptr[unsafe_offset=8 * t + 4],
                accum.ptr[unsafe_offset=8 * t + 5],
            ),
            _pack_accum_pair(
                accum.ptr[unsafe_offset=8 * t + 6],
                accum.ptr[unsafe_offset=8 * t + 7],
            ),
        )
        st_matrix[simd_width=4](wg_half.unsafe_offset(off), data)


@__llvm_arg_metadata(a_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(128 * (consumers + 1))
    ),
    `nvvm.cluster_dim`=StaticTuple[Int32, 3](
        Int32(cluster_m), Int32(1), Int32(1)
    ),
)
# One kernel symbol per layout: the TN and NN instantiations of this body
# would otherwise share one base name (differing only by mangling hash), so
# GPU profiles could not tell them apart and scripts/compare_kernel_asm.py --
# which pairs kernels by hash-stripped name -- would collide them.  The
# ragged tag does the same for the n-clip TT instantiation while keeping
# every exact-n symbol byte-identical to its pre-existing name.
@__name(
    t"{_GEMM16_TAG}_gemm_{_v4_persistent_layout_tag[col_a, kmaj_b]()}_v4_persistent_stmatrix_rolling_m{bm}n{bn}_s{stages}c{cluster_m}wg{consumers}g{_V4_GROUP}{_v4_persistent_ragged_tag[ragged_n]()}"
)
def _rolling_persistent_ws[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool,
    # col_a extends the persistent body to the TN (wgrad) layout: A is
    # physically (K, M), TMA-loaded into an MN-major shared tile and
    # consumed through WGMMA's col-major A mode via _v4_mma_tile.  kmaj_b
    # does the same for B: physically (N, K), TMA-loaded into a K-major
    # shared tile for WGMMA's col-major B mode; col_a + kmaj_b is the TT
    # instantiation.  The trailing shape parameters exist because the TMA
    # boxes follow each operand's majorness; their defaults keep every
    # pre-existing NN and TN instantiation (and its generated code)
    # unchanged.
    col_a: Bool = False,
    kmaj_b: Bool = False,
    # ragged_n admits n % bn != 0 (still n % 64 == 0): blocks_n becomes a
    # ceil-div, the B TMA reads clamp past the n edge (zero-fill, zero
    # contributions) and the C TMA store's partial last column box clips
    # against the (m, n) descriptor -- the same machinery the ragged-m path
    # uses, on the other axis.  The NN, TN and TT routes all instantiate
    # it.
    ragged_n: Bool = False,
    a_tile_shape: IndexList[2] = Index(_V4_BK, bm) if col_a else Index(
        bm, _V4_BK
    ),
    a_desc_shape: IndexList[2] = Index(_V4_BK, 64) if col_a else Index(
        bm, _V4_BK
    ),
    b_tile_shape: IndexList[2] = Index(64, _V4_BK) if kmaj_b else Index(
        _V4_BK, 64
    ),
    b_desc_shape: IndexList[2] = Index(64, _V4_BK) if kmaj_b else Index(
        _V4_BK, 64
    ),
](
    a_tma: TMATensorTile[_V4_DT, 2, a_tile_shape, a_desc_shape],
    b_tma: TMATensorTile[_V4_DT, 2, b_tile_shape, b_desc_shape],
    c_tma: TMATensorTile[_V4_DT, 2, Index(bm, 64), Index(bm, 64)],
    output: _V4_PTR,
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    comptime if _is_sm_9x():
        comptime A_LAYOUT = tile_layout_mn_major[
            _V4_DT, bm, _V4_BK, _V4_SWIZZLE
        ]() if col_a else tile_layout_k_major[_V4_DT, bm, _V4_BK, _V4_SWIZZLE]()
        comptime B_LAYOUT = tile_layout_k_major[
            _V4_DT, bn, _V4_BK, _V4_SWIZZLE
        ]() if kmaj_b else tile_layout_mn_major[
            _V4_DT, bn, _V4_BK, _V4_SWIZZLE
        ]()
        # For both majornesses a 64-row chunk of the bn-row tile is one
        # contiguous 64 * BK block at offset chunk * 64 * BK (BK = 64 bf16 is
        # exactly one 128B swizzle atom row, so the K-major layout is a plain
        # stack of 8-row atoms; the NT kernel's half-tile multicast relies on
        # the same decomposition).
        comptime B_CHUNK_LAYOUT = tile_layout_k_major[
            _V4_DT, 64, _V4_BK, _V4_SWIZZLE
        ]() if kmaj_b else tile_layout_mn_major[
            _V4_DT, 64, _V4_BK, _V4_SWIZZLE
        ]()
        comptime A_PIPE_LAYOUT = Layout.row_major(stages, bm * _V4_BK)
        comptime B_PIPE_LAYOUT = Layout.row_major(stages, bn * _V4_BK)
        # Three carvings of one extern slab -- see `_v4_dyn_smem_tile`.
        var a_pipeline = _v4_dyn_smem_tile[A_PIPE_LAYOUT, 128, 0]()
        var b_pipeline = _v4_dyn_smem_tile[
            B_PIPE_LAYOUT, 128, stages * bm * _V4_BK
        ]()
        # C staging tile for the TMA-store epilogue (swizzled 128B rows of
        # 64 elements, bn // 64 chunks).  A dummy allocation when disabled.
        comptime C_SMEM_ELEMS = bm * bn if tma_store else 512
        comptime C_SMEM_OFFSET = stages * (bm + bn) * _V4_BK
        var c_smem = _v4_dyn_smem_tile[
            Layout.row_major(1, C_SMEM_ELEMS), 1024, C_SMEM_OFFSET
        ]()
        # The C tile is the last carving, so its end IS the slab size the
        # launch must ask for; keeping the two in step is not left to a
        # comment.
        comptime assert (
            _v4_persistent_smem_bytes[stages, bm, bn, tma_store]()
            == (C_SMEM_OFFSET + C_SMEM_ELEMS) * 2
        ), "persistent-body smem carve and launch size disagree"
        var full_barriers = stack_allocation[
            stages,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        var empty_barriers = stack_allocation[
            stages,
            SharedMemBarrier,
            address_space=AddressSpace.SHARED,
            alignment=8,
        ]()
        if thread_idx.x == 0:
            comptime for stage in range(stages):
                full_barriers[unsafe_offset=stage].init()
                # Released by every consumer warp group of every CTA in the
                # cluster: the multicast source must not overwrite a peer's
                # tile while that peer is still reading it.
                empty_barriers[unsafe_offset=stage].init(
                    Int32(consumers * cluster_m)
                )
            a_tma.prefetch_descriptor()
            b_tma.prefetch_descriptor()
            comptime if tma_store:
                c_tma.prefetch_descriptor()
            fence_mbarrier_init()
        # All barriers must be initialized cluster-wide before any arrival
        # (the consumers below arrive at peer CTAs' empty barriers).
        cluster_sync_relaxed()

        comptime CFRAG = 64 * bn // 128
        comptime MACRO_BM = bm * cluster_m
        comptime TMA_BYTES = (bm + bn) * _V4_BK * 2
        comptime MCAST_MASK = UInt16((1 << cluster_m) - 1)
        comptime B_CHUNKS = bn // 64
        var warp_group_idx = Int(thread_idx.x) // 128
        var warp_group_thread_idx = Int(thread_idx.x) % 128
        var rank = Int(block_rank_in_cluster())
        var cluster_id = Int(block_idx.x) // cluster_m
        var num_clusters = Int(grid_dim.x) // cluster_m
        # m may be ragged: TMA A reads clamp out-of-bounds rows and the
        # epilogue stores are row-predicated.  With ragged_n, n may be too
        # (see the parameter comment above).
        var macro_rows = (m + MACRO_BM - 1) // MACRO_BM
        var blocks_n = n // bn
        comptime if ragged_n:
            blocks_n = (n + bn - 1) // bn
        var total_works = macro_rows * blocks_n
        var num_tiles = k // _V4_BK
        var group_span = _V4_GROUP * blocks_n

        # Release every pipeline slot to the producers (cluster-wide).
        if warp_group_idx > 0 and warp_group_thread_idx < cluster_m:
            comptime for stage in range(stages):
                empty_barriers[unsafe_offset=stage].arrive_cluster(
                    UInt32(warp_group_thread_idx)
                )

        if warp_group_idx == 0:
            warpgroup_reg_dealloc[24]()
            if warp_group_thread_idx == 0:
                var ring_stage = 0
                var ring_phase = UInt32(0)
                var w = cluster_id
                while w < total_works:
                    var group = w // group_span
                    var rem = w % group_span
                    var rows_in_group = min(
                        _V4_GROUP, macro_rows - group * _V4_GROUP
                    )
                    var macro_row = group * _V4_GROUP + rem % rows_in_group
                    var n0 = (rem // rows_in_group) * bn
                    var m0 = macro_row * MACRO_BM + rank * bm
                    var t = 0
                    while t < num_tiles:
                        var stage = ring_stage
                        var phase = ring_phase
                        empty_barriers[unsafe_offset=stage].wait(phase)
                        full_barriers[unsafe_offset=stage].expect_bytes(
                            Int32(TMA_BYTES)
                        )
                        var a_tile = LayoutTensor[
                            _V4_DT,
                            A_LAYOUT,
                            MutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                            alignment=128,
                        ](a_pipeline.ptr.unsafe_offset(stage * bm * _V4_BK))
                        var k0 = t * _V4_BK
                        # TMA coordinates are (fastest dim, slower dim) of
                        # the global tensor the descriptor was built over:
                        # (m, k) for the col-major (K, M) wgrad operand.
                        comptime if col_a:
                            a_tma.async_copy(
                                a_tile,
                                full_barriers[unsafe_offset=stage],
                                (m0, k0),
                            )
                        else:
                            a_tma.async_copy(
                                a_tile,
                                full_barriers[unsafe_offset=stage],
                                (k0, m0),
                            )
                        # Cooperative B load: each cluster rank reads its
                        # share of the 64-column chunks once from L2 and
                        # multicasts it to every peer, so the per-SM TMA
                        # engines split the shared-tile traffic instead of
                        # rank 0 funneling all of B (nvjet's "coopB").
                        var cc = rank * B_CHUNKS // cluster_m
                        var cend = (rank + 1) * B_CHUNKS // cluster_m
                        while cc < cend:
                            var b_chunk = LayoutTensor[
                                _V4_DT,
                                B_CHUNK_LAYOUT,
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](
                                b_pipeline.ptr.unsafe_offset(
                                    stage * bn * _V4_BK + cc * 64 * _V4_BK
                                )
                            )
                            # B TMA coordinates follow the descriptor's
                            # global tensor: (k, n) for the K-major (N, K)
                            # kmaj_b operand, (n, k) for the row-major
                            # (K, N) one.
                            comptime if cluster_m > 1:
                                comptime if kmaj_b:
                                    b_tma.async_multicast_load(
                                        b_chunk,
                                        full_barriers[unsafe_offset=stage],
                                        (k0, n0 + cc * 64),
                                        MCAST_MASK,
                                    )
                                else:
                                    b_tma.async_multicast_load(
                                        b_chunk,
                                        full_barriers[unsafe_offset=stage],
                                        (n0 + cc * 64, k0),
                                        MCAST_MASK,
                                    )
                            else:
                                comptime if kmaj_b:
                                    b_tma.async_copy(
                                        b_chunk,
                                        full_barriers[unsafe_offset=stage],
                                        (k0, n0 + cc * 64),
                                    )
                                else:
                                    b_tma.async_copy(
                                        b_chunk,
                                        full_barriers[unsafe_offset=stage],
                                        (n0 + cc * 64, k0),
                                    )
                            cc += 1
                        t += 1
                        ring_stage += 1
                        if ring_stage == stages:
                            ring_stage = 0
                            ring_phase = ring_phase ^ UInt32(1)
                    w += num_clusters
        else:
            # Consumer registers: three warp groups fit 65536 regs/SM only
            # at 160 regs/thread (96 accumulator + addressing); two fit 232.
            comptime if consumers >= 3:
                warpgroup_reg_alloc[160]()
            else:
                warpgroup_reg_alloc[232]()
            var accum = LayoutTensor[
                _V4_F32,
                Layout.row_major(1, CFRAG),
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            comptime wgmma = TensorCoreAsync[
                _V4_F32,
                _V4_DT,
                _V4_DT,
                Index(64, bn, 16),
                a_swizzle=_V4_SWIZZLE,
                b_swizzle=_V4_SWIZZLE,
                transpose_b=False,
            ]()

            var ring_stage = 0
            var ring_phase = UInt32(0)
            var w = cluster_id
            while w < total_works:
                var group = w // group_span
                var rem = w % group_span
                var rows_in_group = min(
                    _V4_GROUP, macro_rows - group * _V4_GROUP
                )
                var macro_row = group * _V4_GROUP + rem % rows_in_group
                var n0 = (rem // rows_in_group) * bn
                var m0 = macro_row * MACRO_BM + rank * bm
                _ = accum.fill(0.0)
                var t = 0
                while t < num_tiles:
                    var stage = ring_stage
                    var phase = ring_phase
                    full_barriers[unsafe_offset=stage].wait(phase)
                    var a_tile = LayoutTensor[
                        _V4_DT,
                        A_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](a_pipeline.ptr.unsafe_offset(stage * bm * _V4_BK))
                    var b_tile = LayoutTensor[
                        _V4_DT,
                        B_LAYOUT,
                        MutAnyOrigin,
                        address_space=AddressSpace.SHARED,
                        alignment=128,
                    ](b_pipeline.ptr.unsafe_offset(stage * bn * _V4_BK))
                    comptime if col_a or kmaj_b:
                        # Raw descriptor path: TensorCoreAsync has no
                        # col-major A mode (and the TT instantiation's
                        # K-major B rides the same majorness-generic
                        # helper).
                        _v4_mma_tile[bn, col_a, kmaj_b, A_LAYOUT, B_LAYOUT](
                            a_tile.ptr, b_tile.ptr, accum, warp_group_idx
                        )
                    else:
                        warpgroup_fence(accum)
                        wgmma.arrive()
                        wgmma.wgmma[consumers](
                            a_tile, b_tile, accum, warp_group_idx - 1
                        )
                        wgmma.commit_group()
                        warpgroup_fence(accum)
                        wgmma.wait_group()
                    if warp_group_thread_idx < cluster_m:
                        empty_barriers[unsafe_offset=stage].arrive_cluster(
                            UInt32(warp_group_thread_idx)
                        )
                    t += 1
                    ring_stage += 1
                    if ring_stage == stages:
                        ring_stage = 0
                        ring_phase = ring_phase ^ UInt32(1)

                var tid = warp_group_thread_idx
                var warp = tid // 32
                var lane = tid % 32
                var base_row = warp * 16 + lane // 4
                var base_col = (lane % 4) * 2
                comptime if tma_store:
                    # Stage the tile in shared memory and hand it to TMA;
                    # the store drains in the background of the next work's
                    # mainloop, and TMA clips rows past a ragged m edge.
                    comptime NCONS = Int32(consumers * 128)
                    if warp_group_idx == 1 and warp_group_thread_idx == 0:
                        # Previous work's store must fully drain before the
                        # staging tile is overwritten.
                        c_tma.wait_group[0]()
                    named_barrier[NCONS](1)
                    _store_accum_bm_boxes_stmatrix[bm, bn](
                        c_smem.ptr, accum, warp, lane, warp_group_idx
                    )
                    fence_async_view_proxy()
                    named_barrier[NCONS](1)
                    if warp_group_idx == 1 and warp_group_thread_idx == 0:
                        comptime for chunk in range(bn // 64):
                            var c_chunk = LayoutTensor[
                                _V4_DT,
                                Layout.row_major(bm, 64),
                                MutAnyOrigin,
                                address_space=AddressSpace.SHARED,
                                alignment=128,
                            ](c_smem.ptr.unsafe_offset(chunk * bm * 64))
                            c_tma.async_store(c_chunk, (n0 + chunk * 64, m0))
                        c_tma.commit_group()
                else:
                    comptime for q in range(CFRAG // 2):
                        var e = q * 2
                        var row = (
                            (warp_group_idx - 1) * 64 + base_row + (q % 2) * 8
                        )
                        var col = base_col + (q // 2) * 8
                        var pair = SIMD[_V4_DT, 2](
                            accum.ptr[unsafe_offset=e].cast[_V4_DT](),
                            accum.ptr[unsafe_offset=e + 1].cast[_V4_DT](),
                        )
                        if m0 + row < m and n0 + col + 1 < n:
                            output.unsafe_store[alignment=4](
                                (m0 + row) * n + n0 + col, pair
                            )
                w += num_clusters
            comptime if tma_store:
                # Outstanding bulk stores must complete before kernel exit.
                if warp_group_idx == 1 and warp_group_thread_idx == 0:
                    c_tma.wait_group[0]()

        # Peer CTAs receive multicast writes into this CTA's shared memory;
        # do not tear the block down while any cluster member is running.
        cluster_sync()


def enqueue_rolling_persistent[
    stages: Int,
    cluster_m: Int,
    bm: Int,
    bn: Int,
    consumers: Int,
    tma_store: Bool = False,
    col_a: Bool = False,
    kmaj_b: Bool = False,
    ragged_n: Bool = False,
](
    output: _V4_PTR,
    a: _V4_PTR,
    b: _V4_PTR,
    m: Int,
    n: Int,
    k: Int,
    sm_count: Int,
    ctx: DeviceContext,
) raises:
    # Each descriptor follows its operand's physical layout: (M, K) row-major
    # with a whole-tile box, or -- for the TN/wgrad and TT col_a routes --
    # (K, M) row-major with a (BK, 64) box feeding the MN-major shared tile;
    # likewise (K, N) row-major for B, or -- for the TT kmaj_b route --
    # (N, K) row-major with a (64, BK) box feeding the K-major shared tile.
    comptime A_TILE = Index(_V4_BK, bm) if col_a else Index(bm, _V4_BK)
    comptime A_DESC = Index(_V4_BK, 64) if col_a else Index(bm, _V4_BK)
    comptime B_TILE = Index(64, _V4_BK) if kmaj_b else Index(_V4_BK, 64)
    var a_dim0 = k if col_a else m
    var a_dim1 = m if col_a else k
    var a_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            a.unsafe_address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](a_dim0, a_dim1),
        IndexList[2](a_dim1, 1),
        IndexList[2](A_DESC[0], A_DESC[1]),
    )
    var b_dim0 = n if kmaj_b else k
    var b_dim1 = k if kmaj_b else n
    var b_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            b.unsafe_address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](b_dim0, b_dim1),
        IndexList[2](b_dim1, 1),
        IndexList[2](B_TILE[0], B_TILE[1]),
    )
    var c_desc = create_tma_descriptor[_V4_DT, 2, _V4_SWIZZLE](
        DeviceBuffer(
            ctx,
            output.unsafe_address_space_cast[AddressSpace.GENERIC](),
            1,
            owning=False,
        ),
        IndexList[2](m, n),
        IndexList[2](n, 1),
        IndexList[2](bm, 64),
    )
    var a_tma = TMATensorTile[_V4_DT, 2, A_TILE, A_DESC](a_desc)
    var b_tma = TMATensorTile[_V4_DT, 2, B_TILE, B_TILE](b_desc)
    var c_tma = TMATensorTile[_V4_DT, 2, Index(bm, 64), Index(bm, 64)](c_desc)
    var macro_rows = (m + bm * cluster_m - 1) // (bm * cluster_m)
    var blocks_n = n // bn
    comptime if ragged_n:
        blocks_n = (n + bn - 1) // bn
    var total_works = macro_rows * blocks_n
    var num_clusters = min(sm_count // cluster_m, total_works)
    var grid_x = num_clusters * cluster_m
    comptime DYN_SMEM = _v4_persistent_smem_bytes[stages, bm, bn, tma_store]()
    # Compiled once per process and context: `ctx.enqueue_function[kernel]`
    # re-runs compile_function on every launch. The key names what selects the
    # code -- dtype, geometry, stages, layout, epilogue and the PAIR_CAST /
    # raster-group build defines -- and nothing about this call's pointers or
    # its m/n/k, which travel as arguments. The cluster shape rides on the
    # kernel's own `nvvm.cluster_dim` metadata, as it did before.
    _enqueue_cached[
        _rolling_persistent_ws[
            stages,
            cluster_m,
            bm,
            bn,
            consumers,
            tma_store,
            col_a,
            kmaj_b,
            ragged_n,
        ],
        dyn_smem=DYN_SMEM,
    ](
        ctx,
        String(
            t"g16roll_{_GEMM16_TAG}_s{stages}c{cluster_m}m{bm}n{bn}w{consumers}_{Int(tma_store)}{Int(col_a)}{Int(kmaj_b)}{Int(ragged_n)}_g{_V4_GROUP}_p{Int(_ROLL_PAIR_CAST)}"
        ),
        grid_x,
        1,
        1,
        128 * (consumers + 1),
        a_tma,
        b_tma,
        c_tma,
        output,
        Int64(m),
        Int64(n),
        Int64(k),
    )
