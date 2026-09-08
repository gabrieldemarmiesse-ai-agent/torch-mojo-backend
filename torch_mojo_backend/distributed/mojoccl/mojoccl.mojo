# mojoccl: a Mojo shared library exporting NCCL's C ABI (nccl.h), so
# torch_mojo_backend/distributed/nccl.py can dlopen it exactly like
# libnccl.so.2/librccl.so.1 (TORCH_MOJO_BACKEND_CCL=mojo), and so external
# tools (nccl-tests) can link it as a libnccl.so drop-in.
#
# Signatures, enum values and ncclResult_t codes are pinned to
# /home/gabriel/projects/nccl/src/nccl.h.in (2.31.2) -- the source of truth
# is nccl.py's `_declare()`, which this library's exports were written
# against. AllReduce/Broadcast/AllGather are real; Reduce/ReduceScatter/
# Send/Recv return ncclInvalidUsage (GPT-2 DDP needs only the first three;
# tests/ddp_worker.py skips the checks that need them when
# TORCH_MOJO_BACKEND_CCL=mojo is set).
#
# One process per GPU. Within a node, one region per rank of raw
# driver-owned memory shared with legacy IPC (driver.mojo) -- MAX's own
# allocator memory cannot be exported that way (see
# docs/mojo_collectives_feasibility.md, section 5.6) -- and the collectives
# of collectives_kernels.mojo run over the peer mappings. Across nodes,
# GPUDirect RDMA written here over libibverbs (ibverbs.mojo,
# internode.mojo): no vendor collective library takes part at any level.
#
# A multi-node communicator runs each collective hierarchically:
#
#   allreduce  reduce_scatter_stage (node-local)  ->  one RDMA exchange of
#              this rank's 1/local_world shard with the SAME local_rank on
#              every other node, summed by inbox_add  ->  allgather_finish
#   broadcast  root stages and RDMA-writes to its same-local_rank peers,
#              then every node runs the node-local broadcast from its own
#              local root
#   allgather  node-local allgather into a node block, one RDMA exchange of
#              node blocks, then place each block by GLOBAL rank (read out
#              of the bootstrap table, not assumed to be
#              node * local_world + local_rank)
#
# Broadcast and allgather run once at init and are written for clarity
# rather than speed; allreduce is the one the DDP step waits on.
#
# Single-node communicators never touch libibverbs at all -- same fused
# kernels, same numbers as before this file learned about nodes.

from std.collections import Dict
from std.ffi import OwnedDLHandle
from std.gpu import global_idx
from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.sys import size_of
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceBuffer, DeviceStream

from driver import (
    HANDLE_BYTES,
    alloc_region,
    close_handle,
    current_device_ordinal,
    free_region,
    get_handle,
    open_driver,
    open_handle,
)
from bootstrap import (
    UID_BYTES,
    BootstrapConn,
    bootstrap_allgather,
    bootstrap_barrier,
    bootstrap_connect,
    derive_topology,
    host_hash,
    make_unique_id,
)
from collectives_kernels import (
    MAX_WORLD,
    allgather,
    allgather_finish,
    allreduce,
    broadcast,
    error_offset,
    region_init,
    reduce_scatter_stage,
    shard_range,
    signal_bytes,
)
from internode import (
    IB_BLOB_BYTES,
    MAX_NODES,
    ib_connect,
    ib_enqueue,
    ib_error,
    ib_local_info,
    ib_next_seq,
    ib_npeers,
    ib_port_lid,
    ib_port_mtu,
    ib_setup,
    ib_teardown,
)
from internode_kernels import copy_bytes, inbox_add, place_blocks


# ncclResult_t (nccl.h.in:44-53)
comptime NCCL_SUCCESS: Int32 = 0
comptime NCCL_UNHANDLED_CUDA_ERROR: Int32 = 1
comptime NCCL_SYSTEM_ERROR: Int32 = 2
comptime NCCL_INTERNAL_ERROR: Int32 = 3
comptime NCCL_INVALID_ARGUMENT: Int32 = 4
comptime NCCL_INVALID_USAGE: Int32 = 5
comptime NCCL_REMOTE_ERROR: Int32 = 6
comptime NCCL_IN_PROGRESS: Int32 = 7

# ncclDataType_t (nccl.h.in:466-479). AllReduce runs a real kernel and only
# instantiates the five below; Broadcast/AllGather move raw bytes (item size
# x count -> nbytes) and accept every type in the enum.
comptime NCCL_INT8: Int32 = 0
comptime NCCL_UINT8: Int32 = 1
comptime NCCL_INT32: Int32 = 2
comptime NCCL_UINT32: Int32 = 3
comptime NCCL_INT64: Int32 = 4
comptime NCCL_UINT64: Int32 = 5
comptime NCCL_FLOAT16: Int32 = 6
comptime NCCL_FLOAT32: Int32 = 7
comptime NCCL_FLOAT64: Int32 = 8
comptime NCCL_BFLOAT16: Int32 = 9

# ncclRedOp_t (nccl.h.in:448-463) -- values this library implements.
comptime NCCL_SUM: Int32 = 0
comptime NCCL_AVG: Int32 = 4

# Payload a rank sends when it has nothing to contribute to an exchange.
#
# Every exchange is all-to-all among the ranks sharing a local_rank, even
# when only one of them has data (a broadcast, or an allreduce whose shard
# table leaves this rank empty), and that is a correctness requirement, not
# tidiness. The inbox is double buffered by the exchange counter's parity,
# so what has to be true is: peer B's write for exchange e+2 lands after MY
# add kernel for exchange e has read that half. All-to-all makes it a proof
# -- B cannot post e+2 before its callback for e+1 returned, which needed MY
# e+1 message, which my stream sent only after running my add kernel for e.
# Let one exchange be one-directional and that chain breaks, leaving nothing
# but a timing margin between B's write and my read.
comptime CREDIT_BYTES = 16

comptime DEFAULT_REGION_MB = 256
comptime DEFAULT_BOOTSTRAP_TIMEOUT_S: Float64 = 120.0


def _region_cap_bytes() -> Int:
    var s = getenv("MOJOCCL_REGION_MB", String(DEFAULT_REGION_MB))
    try:
        return Int(s) * 1024 * 1024
    except:
        return DEFAULT_REGION_MB * 1024 * 1024


def _bootstrap_timeout_s() -> Float64:
    var s = getenv(
        "MOJOCCL_BOOTSTRAP_TIMEOUT_S", String(DEFAULT_BOOTSTRAP_TIMEOUT_S)
    )
    try:
        return Float64(s)
    except:
        return DEFAULT_BOOTSTRAP_TIMEOUT_S


def _dtype_item_bytes(nccl_dtype: Int32) -> Int:
    """Item size for the five dtypes AllReduce's kernel is instantiated for."""
    if nccl_dtype == NCCL_INT32 or nccl_dtype == NCCL_FLOAT32:
        return 4
    elif nccl_dtype == NCCL_INT64:
        return 8
    elif nccl_dtype == NCCL_FLOAT16 or nccl_dtype == NCCL_BFLOAT16:
        return 2
    else:
        return 0


def _any_dtype_item_bytes(nccl_dtype: Int32) -> Int:
    """Item size for every ncclDataType_t -- Broadcast/AllGather move raw
    bytes and never look at the dtype beyond this."""
    if nccl_dtype == NCCL_INT8 or nccl_dtype == NCCL_UINT8:
        return 1
    elif (
        nccl_dtype == NCCL_INT32
        or nccl_dtype == NCCL_UINT32
        or nccl_dtype == NCCL_FLOAT32
    ):
        return 4
    elif (
        nccl_dtype == NCCL_INT64
        or nccl_dtype == NCCL_UINT64
        or nccl_dtype == NCCL_FLOAT64
    ):
        return 8
    elif nccl_dtype == NCCL_FLOAT16 or nccl_dtype == NCCL_BFLOAT16:
        return 2
    else:
        return 0


# ---------------------------------------------------------------------------
# Communicator state, behind the opaque `ncclComm_t` (void*) every exported
# function after ncclCommInitRank receives. Heap-allocated once per
# communicator and never freed on destroy (the struct itself is a few
# hundred bytes; only the GPU region, the peer IPC mappings and the IB
# resources -- the ones that matter -- are released there).
#
# `regions` is indexed by LOCAL rank and padded to MAX_WORLD with zeros:
# only same-node peers are IPC-mapped, and a node contributes at most
# MAX_WORLD ranks however large the communicator is.
# ---------------------------------------------------------------------------


struct CommState(Movable):
    var rank: Int
    var world: Int
    var ordinal: Int
    var ctx: DeviceContext
    var driver: OwnedDLHandle
    var cap_bytes: Int
    var regions: StaticTuple[Int, MAX_WORLD]
    var owned_base: Int
    var generation: Int
    var last_stream: Int64
    var aborted: Bool
    var stream_cache: Dict[Int64, DeviceStream]
    var local_rank: Int
    var local_world: Int
    var my_node: Int
    var nnodes: Int
    var rank_at: List[Int]
    var ib: Int
    var net_off: Int

    def __init__(
        out self,
        rank: Int,
        world: Int,
        ordinal: Int,
        ctx: DeviceContext,
        var driver: OwnedDLHandle,
        cap_bytes: Int,
        regions: StaticTuple[Int, MAX_WORLD],
        owned_base: Int,
        local_rank: Int,
        local_world: Int,
        my_node: Int,
        nnodes: Int,
        var rank_at: List[Int],
        ib: Int,
        net_off: Int,
    ):
        self.rank = rank
        self.world = world
        self.ordinal = ordinal
        self.ctx = ctx
        self.driver = driver^
        self.cap_bytes = cap_bytes
        self.regions = regions
        self.owned_base = owned_base
        self.generation = 0
        self.last_stream = 0
        self.aborted = False
        self.stream_cache = Dict[Int64, DeviceStream]()
        self.local_rank = local_rank
        self.local_world = local_world
        self.my_node = my_node
        self.nnodes = nnodes
        self.rank_at = rank_at^
        self.ib = ib
        self.net_off = net_off


@always_inline
def _align_up(x: Int, a: Int) -> Int:
    return (x + a - 1) // a * a


@always_inline
def _comm_ptr(comm: Int64) -> Pointer[CommState, MutAnyOrigin]:
    return Pointer[CommState, MutAnyOrigin](unsafe_from_address=Int(comm))


@always_inline
def _any(p: Pointer[UInt8, MutUntrackedOrigin]) -> Pointer[UInt8, MutAnyOrigin]:
    """Rebinds an `unsafe_alloc`-returned pointer's origin to `MutAnyOrigin`,
    matching what driver.mojo/bootstrap.mojo (and the exported ABI functions
    they share signatures with) declare."""
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p))


def _ensure_stream_cached(mut state: CommState, handle: Int64) raises:
    """`create_external_stream` wraps a raw `cudaStream_t`/`hipStream_t` in a
    `DeviceStream`; every collective call was re-wrapping the SAME handle
    (torch hands this library one stable stream per torch.Stream for the
    whole communicator's life -- it is not reused across streams), so cache
    the wrapper in `state.stream_cache` keyed by the raw handle instead of
    re-wrapping on every call. A handle not seen before is wrapped and
    inserted; callers then read `state.stream_cache[handle]` directly (a
    `ref`, no copy). If a handle were ever reused for a different stream this
    would keep returning the stale wrapper -- not something a torch process
    does, but worth knowing if that assumption ever breaks.
    """
    if handle not in state.stream_cache:
        var wrapped = state.ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(handle))
        )
        _ = state.stream_cache.insert(handle, wrapped^)


def _copy_error_word(
    dst: Pointer[UInt64, MutAnyOrigin], src: Pointer[UInt64, MutAnyOrigin]
):
    """One-thread device kernel: copies the error word out of a region's
    signal area -- raw driver memory (`cuMemAlloc`/`hipExtMallocWithFlags`),
    not host-accessible -- into a proper `DeviceBuffer` `enqueue_copy` can
    D2H-copy from. Mirrors the production kernel harness's `_copy_u64`
    (kernel/harness.mojo, `check_error`): the error word cannot be read by
    dereferencing a host pointer at the region address, that faults or reads
    garbage.
    """
    if global_idx.x == 0:
        dst[unsafe_offset=0] = src[unsafe_offset=0]


def _read_error_word(
    ctx: DeviceContext, stream: DeviceStream, region: Int
) raises -> UInt64:
    """The UInt64 error word at `region + error_offset()`, fetched to the
    host via a real device-to-host copy (see `_copy_error_word`).

    Enqueues the copy kernel on `stream` -- ordered after whatever
    collective on it last wrote the word -- then a D2H `enqueue_copy`, then
    blocks for both. Callers that need the fully up-to-date word (a peer's
    in-flight barrier timeout, not just what already landed) must
    synchronize `stream` themselves first, as `ncclCommGetAsyncError` does.
    """
    var src = Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=region + error_offset()
    )
    var dev_word = ctx.enqueue_create_buffer[DType.uint64](1)
    var compiled = ctx.compile_function[_copy_error_word]()
    stream.enqueue_function(
        compiled,
        dev_word.unsafe_ptr(),
        src,
        grid_dim=(1, 1, 1),
        block_dim=(32, 1, 1),
    )
    var host_word = unsafe_alloc[UInt64](1)
    ctx.enqueue_copy(host_word, dev_word)
    ctx.synchronize()
    return host_word[unsafe_offset=0]


# ---------------------------------------------------------------------------
# Version / error string / unique id
# ---------------------------------------------------------------------------


@export
def ncclGetVersion(version: Pointer[Int32, MutAnyOrigin]) abi("C") -> Int32:
    # 2.31.2, encoded per nccl.h.in's NCCL_VERSION macro: X*10000+Y*100+Z for
    # Y>8 (true from 2.9 on) -- 2*10000 + 31*100 + 2 = 23102, matching the
    # pinned header this ABI was written against. (A prior value here, 22031,
    # did not decode to 2.31.2 under that formula.)
    version[] = 23102
    return NCCL_SUCCESS


comptime _ERR_SUCCESS: StaticString = "no error"
comptime _ERR_UNHANDLED_CUDA: StaticString = "unhandled cuda/hip error"
comptime _ERR_SYSTEM: StaticString = "system error"
comptime _ERR_INTERNAL: StaticString = "internal error"
comptime _ERR_INVALID_ARGUMENT: StaticString = "invalid argument"
comptime _ERR_INVALID_USAGE: StaticString = (
    "invalid usage (not implemented by mojoccl)"
)
comptime _ERR_REMOTE: StaticString = "remote error (a peer's barrier timed out)"
comptime _ERR_IN_PROGRESS: StaticString = "operation in progress"
comptime _ERR_UNKNOWN: StaticString = "unknown result code"


@export
def ncclGetErrorString(
    result: Int32,
) abi("C") -> Pointer[UInt8, ImmStaticOrigin]:
    if result == NCCL_SUCCESS:
        return _ERR_SUCCESS.unsafe_ptr()
    elif result == NCCL_UNHANDLED_CUDA_ERROR:
        return _ERR_UNHANDLED_CUDA.unsafe_ptr()
    elif result == NCCL_SYSTEM_ERROR:
        return _ERR_SYSTEM.unsafe_ptr()
    elif result == NCCL_INTERNAL_ERROR:
        return _ERR_INTERNAL.unsafe_ptr()
    elif result == NCCL_INVALID_ARGUMENT:
        return _ERR_INVALID_ARGUMENT.unsafe_ptr()
    elif result == NCCL_INVALID_USAGE:
        return _ERR_INVALID_USAGE.unsafe_ptr()
    elif result == NCCL_REMOTE_ERROR:
        return _ERR_REMOTE.unsafe_ptr()
    elif result == NCCL_IN_PROGRESS:
        return _ERR_IN_PROGRESS.unsafe_ptr()
    else:
        return _ERR_UNKNOWN.unsafe_ptr()


@export
def ncclGetUniqueId(uid_out: Pointer[UInt8, MutAnyOrigin]) abi("C") -> Int32:
    try:
        make_unique_id(uid_out)
        return NCCL_SUCCESS
    except:
        return NCCL_SYSTEM_ERROR


# ---------------------------------------------------------------------------
# ncclCommInitRank -- the one struct-by-value export. `ncclUniqueId commId`
# is 128 bytes, SysV MEMORY class: it consumes no integer register and is
# pushed on the stack by a real C caller, so `rank` (declared after it) gets
# the next FREE register (rdx), not the one following `nranks` textually.
# Mirrored here by 3 real leading params (comm/nranks/rank -> rdi/esi/edx),
# 3 Int64 dummies exhausting rcx/r8/r9, then 16 UInt64 stack params that ARE
# the struct -- the callee-side twin of the caller-side shim
# proto/ipc_probe.mojo uses for cuIpcOpenMemHandle. Verified host-only with
# ctypes against this exact parameter shape (register/stack layout, no GPU
# needed): a probe function of this shape decoded nranks, rank and a
# checksum of the 16 id words correctly when called exactly as
# ncclCommInitRank(comm, nranks, ncclUniqueId, rank) would be.
# ---------------------------------------------------------------------------


@export
def ncclCommInitRank(
    comm_out: Pointer[Int64, MutAnyOrigin],
    nranks: Int32,
    rank: Int32,
    _r1: Int64,
    _r2: Int64,
    _r3: Int64,
    id0: UInt64,
    id1: UInt64,
    id2: UInt64,
    id3: UInt64,
    id4: UInt64,
    id5: UInt64,
    id6: UInt64,
    id7: UInt64,
    id8: UInt64,
    id9: UInt64,
    id10: UInt64,
    id11: UInt64,
    id12: UInt64,
    id13: UInt64,
    id14: UInt64,
    id15: UInt64,
) abi("C") -> Int32:
    # `regions` (CommState and every collective kernel's argument list) is a
    # fixed StaticTuple[.., MAX_WORLD]: a larger world would index past its
    # end. Guard explicitly rather than rely on StaticTuple's own bounds
    # check, whose behavior under a release (non-debug) build is not this
    # library's contract to depend on.
    if Int(nranks) < 1 or Int(rank) < 0 or Int(rank) >= Int(nranks):
        return NCCL_INVALID_ARGUMENT
    try:
        var idbuf = unsafe_alloc[UInt8](UID_BYTES)
        var idbuf64 = idbuf.unsafe_bitcast[UInt64]()
        idbuf64[unsafe_offset=0] = id0
        idbuf64[unsafe_offset=1] = id1
        idbuf64[unsafe_offset=2] = id2
        idbuf64[unsafe_offset=3] = id3
        idbuf64[unsafe_offset=4] = id4
        idbuf64[unsafe_offset=5] = id5
        idbuf64[unsafe_offset=6] = id6
        idbuf64[unsafe_offset=7] = id7
        idbuf64[unsafe_offset=8] = id8
        idbuf64[unsafe_offset=9] = id9
        idbuf64[unsafe_offset=10] = id10
        idbuf64[unsafe_offset=11] = id11
        idbuf64[unsafe_offset=12] = id12
        idbuf64[unsafe_offset=13] = id13
        idbuf64[unsafe_offset=14] = id14
        idbuf64[unsafe_offset=15] = id15
        return _init_rank(_any(idbuf), Int(rank), Int(nranks), comm_out)
    except e:
        print("mojoccl: ncclCommInitRank failed:", e)
        return NCCL_INTERNAL_ERROR


def _init_rank(
    uid: Pointer[UInt8, MutAnyOrigin],
    rank: Int,
    nranks: Int,
    comm_out: Pointer[Int64, MutAnyOrigin],
) raises -> Int32:
    """Three bootstrap rounds and everything they gate.

    Round 1 gathers host identity, from which every rank derives the same
    node/local_rank table. Round 2 gathers the 64-byte IPC handle plus, on a
    multi-node communicator, this rank's IB connection data. Round 3 is a
    barrier: past it every peer's region is zeroed, its IPC handles are open
    and its queue pairs are in RTS, so the first collective may write into
    it.
    """
    var timeout_s = _bootstrap_timeout_s()
    var conn = bootstrap_connect(uid, rank, nranks, timeout_s)

    # Round 1: host identity.
    var b1 = unsafe_alloc[UInt8](16)
    var b1w = b1.unsafe_bitcast[UInt64]()
    b1w[unsafe_offset=0] = host_hash()
    b1w[unsafe_offset=1] = UInt64(rank)
    var t1 = unsafe_alloc[UInt8](16 * nranks)
    bootstrap_allgather(conn, _any(b1), 16, _any(t1), timeout_s)
    var t1w = t1.unsafe_bitcast[UInt64]()
    var hashes = List[UInt64]()
    for r in range(nranks):
        hashes.append(t1w[unsafe_offset = 2 * r])
    var topo = derive_topology(hashes, rank)
    if topo.local_world > MAX_WORLD:
        conn.close()
        raise Error(
            "mojoccl: "
            + String(topo.local_world)
            + " ranks on one node, but the intra-node kernels are built for at"
            " most "
            + String(MAX_WORLD)
        )
    if topo.nnodes > MAX_NODES:
        conn.close()
        raise Error(
            "mojoccl: " + String(topo.nnodes) + " nodes exceeds the "
            + String(MAX_NODES) + "-node limit of the inbox layout"
        )

    var lib = open_driver()
    var ordinal = current_device_ordinal(lib)
    var ctx = DeviceContext(device_id=ordinal)
    var cap_bytes = _region_cap_bytes()
    # A positive multiple of 4096 (the kernels' own precondition,
    # RESULTS.md section 9) is what keeps every per-chunk offset the
    # collectives form 16-byte aligned for every supported dtype.
    if cap_bytes <= 0 or cap_bytes % 4096 != 0:
        conn.close()
        raise Error("mojoccl: MOJOCCL_REGION_MB must be a positive 4 KiB multiple")
    # A multi-node communicator gets a third cap-sized area on top of
    # [signal | stage_in | stage_out]: everything the network touches --
    # the inbox the peers RDMA into, and the staging broadcast and
    # allgather need because user buffers are not registered -- lives
    # there and nowhere else. Aliasing it onto stage_in would have been
    # free in memory and wrong in fact: a peer node writes my inbox as
    # soon as ITS reduce-scatter is done, which is not ordered against
    # MY reduce-scatter still using stage_in, nor against a local peer
    # still reading the previous generation's staging. A single-node
    # communicator allocates none of it and keeps exactly the old
    # region.
    var net_bytes = cap_bytes if topo.nnodes > 1 else 0
    var net_off = signal_bytes() + 2 * cap_bytes
    var region_bytes = net_off + net_bytes
    var base = alloc_region(lib, region_bytes)
    region_init(ctx, base)

    var ib = 0
    if topo.nnodes > 1:
        ib = ib_setup(
            lib,
            ordinal,
            topo.my_local_rank,
            topo.my_node,
            topo.nnodes,
            base,
            region_bytes,
        )

    # Round 2: IPC handle + IB connection data.
    comptime BLOB2 = HANDLE_BYTES + IB_BLOB_BYTES
    var b2 = unsafe_alloc[UInt8](BLOB2)
    for i in range(BLOB2):
        b2[unsafe_offset=i] = 0
    get_handle(lib, base, _any(b2))
    var my_lid = 0
    var my_mtu = 0
    if ib != 0:
        my_lid = ib_port_lid(ib)
        my_mtu = ib_port_mtu(ib)
        ib_local_info(
            ib,
            Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(b2) + HANDLE_BYTES
            ),
            my_lid,
            my_mtu,
        )
    var t2 = unsafe_alloc[UInt8](BLOB2 * nranks)
    bootstrap_allgather(conn, _any(b2), BLOB2, _any(t2), timeout_s)

    # Same-node peers only: an IPC handle from another host is meaningless.
    var regions = StaticTuple[Int, MAX_WORLD](fill=0)
    regions[topo.my_local_rank] = base
    for r in range(nranks):
        if topo.node_of[r] != topo.my_node or r == rank:
            continue
        var lr = topo.local_rank_of[r]
        regions[lr] = open_handle(
            lib,
            Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(t2) + r * BLOB2
            ),
        )

    if ib != 0:
        var peer_rank_of_node = List[Int]()
        for j in range(topo.nnodes):
            peer_rank_of_node.append(
                topo.rank_at[j * topo.local_world + topo.my_local_rank]
            )
        ib_connect(
            ib,
            Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(t2) + HANDLE_BYTES
            ),
            BLOB2,
            peer_rank_of_node,
            my_mtu,
        )

    bootstrap_barrier(conn, timeout_s)
    conn.close()

    var rank_at = List[Int]()
    for i in range(len(topo.rank_at)):
        rank_at.append(topo.rank_at[i])
    var state = CommState(
        rank=rank,
        world=nranks,
        ordinal=ordinal,
        ctx=ctx,
        driver=lib^,
        cap_bytes=cap_bytes,
        regions=regions,
        owned_base=base,
        local_rank=topo.my_local_rank,
        local_world=topo.local_world,
        my_node=topo.my_node,
        nnodes=topo.nnodes,
        rank_at=rank_at^,
        ib=ib,
        net_off=net_off,
    )
    var handle_ptr = unsafe_alloc[CommState](1)
    handle_ptr.unsafe_write(state^)
    comm_out[] = Int64(Int(handle_ptr))
    return NCCL_SUCCESS


# ---------------------------------------------------------------------------
# Communicator lifecycle
# ---------------------------------------------------------------------------


@export
def ncclCommDestroy(comm: Int64) abi("C") -> Int32:
    try:
        var ptr = _comm_ptr(comm)
        ref state = ptr[]
        if not state.aborted:
            # The inter-node callbacks were enqueued on the CALLER's stream,
            # not on the context's own, and they dereference the IbState this
            # tears down -- so drain that stream too before touching it.
            if state.last_stream != 0:
                _ensure_stream_cached(state, state.last_stream)
                state.stream_cache[state.last_stream].synchronize()
            state.ctx.synchronize()
            ib_teardown(state.ib)
            for r in range(state.local_world):
                if r != state.local_rank:
                    close_handle(state.driver, state.regions[r])
            free_region(state.driver, state.owned_base)
        return NCCL_SUCCESS
    except:
        return NCCL_INTERNAL_ERROR


@export
def ncclCommAbort(comm: Int64) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    # No wait, no cleanup of GPU or IB resources here (a dead peer may hang
    # forever inside its own barrier spin, and a queue pair torn down under
    # an in-flight RDMA write is worse than one left alone) -- matches
    # ncclCommAbort's documented "don't wait" contract. ncclCommDestroy is
    # never called after abort() by nccl.py's NcclComm.
    state.aborted = True
    return NCCL_SUCCESS


@export
def ncclCommGetAsyncError(
    comm: Int64, err_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    try:
        ref state = _comm_ptr(comm)[]
        if state.aborted:
            err_out[] = NCCL_SYSTEM_ERROR
            return NCCL_SUCCESS
        if state.ib != 0 and ib_error(state.ib) != 0:
            # A host callback gave up (post failed, a completion came back
            # with a bad status, or nothing arrived inside the deadline).
            # Host-side state, so it needs no device read.
            err_out[] = NCCL_REMOTE_ERROR
            return NCCL_SUCCESS
        # Synchronize first: the freshest read this cheaply-checkable word
        # can give is "everything enqueued so far landed", same as before --
        # only the read itself changes, from a host dereference of device
        # memory (wrong) to a real D2H copy (_read_error_word).
        # Not cached: an error poll, not a collective -- rare enough that the
        # wrap cost this library's cache exists to avoid does not matter here,
        # and this is the one call site that needs a *fresh* wrap or the
        # comm's own default stream depending on whether a collective has
        # run yet, which does not fit the single-handle cache lookup below.
        var s = (
            state.ctx.create_external_stream(
                OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(state.last_stream))
            )
            if state.last_stream != 0
            else DeviceStream(state.ctx)
        )
        s.synchronize()
        var word = _read_error_word(
            state.ctx, s, state.regions[state.local_rank]
        )
        err_out[] = NCCL_REMOTE_ERROR if Int(word) != 0 else NCCL_SUCCESS
        return NCCL_SUCCESS
    except:
        return NCCL_INTERNAL_ERROR


@export
def ncclCommCount(
    comm: Int64, count_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    count_out[] = Int32(state.world)
    return NCCL_SUCCESS


@export
def ncclCommUserRank(
    comm: Int64, rank_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    rank_out[] = Int32(state.rank)
    return NCCL_SUCCESS


# ---------------------------------------------------------------------------
# Group semantics: every op here already executes on the same stream, in
# issue order, so aggregating a group buys nothing -- immediate execution is
# correct, per the design brief.
# ---------------------------------------------------------------------------


@export
def ncclGroupStart() abi("C") -> Int32:
    return NCCL_SUCCESS


@export
def ncclGroupEnd() abi("C") -> Int32:
    return NCCL_SUCCESS


# ---------------------------------------------------------------------------
# Collectives
# ---------------------------------------------------------------------------


def _inter_node_exchange[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    numel: Int,
) raises:
    """The middle third of a multi-node allreduce, for one chunk.

    On entry this rank's node-local sum of its own shard sits in its
    stage_out (that is `reduce_scatter_stage`'s contract). This enqueues, on
    the caller's stream: a host callback that RDMA-writes that shard to the
    same-local_rank rank of every other node and waits for theirs, then the
    kernel that adds what arrived. On exit the shard holds the sum over the
    WHOLE communicator, ready for `allgather_finish`.

    Inbox geometry is derived from `(numel, local_world, item)` alone so that
    a sender computes the same addresses as its receiver: `net_off` holds two
    halves of `(nnodes-1)` slots of one shard each, and the exchange
    counter's parity picks the half. `_max_chunk_bytes` keeps both halves
    inside the area.
    """
    comptime item = size_of[dtype]()
    var sr = shard_range(numel, state.local_world, state.local_rank, item)
    var off_e = sr[0]
    var cnt_e = sr[1]
    var seq = ib_next_seq(state.ib)
    var npeers = ib_npeers(state.ib)
    var nbytes = cnt_e * item
    var slot_bytes = _align_up(max(nbytes, CREDIT_BYTES), 16)
    var half = npeers * slot_bytes
    if 2 * half > state.cap_bytes:
        raise Error(
            "mojoccl: the inter-node inbox overflows the network area; this"
            " is a chunking bug"
        )
    var inbox_base = state.net_off + (seq & 1) * half
    var shard = (
        state.owned_base + signal_bytes() + state.cap_bytes + off_e * item
    )
    # A rank with an empty shard (numel below local_world's rounded-up shard
    # size -- DDP's 4-byte AVG allreduce does exactly this on 7 of 8 local
    # ranks) still exchanges, with CREDIT_BYTES of ignored payload. See
    # `CREDIT_BYTES` for why every exchange has to be all-to-all.
    ib_enqueue(
        state.ib,
        state.driver,
        Int(raw_stream),
        shard if cnt_e > 0 else state.owned_base,
        nbytes if cnt_e > 0 else CREDIT_BYTES,
        inbox_base,
        slot_bytes,
        True,
        npeers,
        state.owned_base + inbox_base if cnt_e > 0 else 0,
        seq,
    )
    if cnt_e > 0:
        inbox_add[dtype](
            state.ctx,
            stream,
            shard,
            state.owned_base + inbox_base,
            cnt_e,
            slot_bytes,
            npeers,
        )


def _max_chunk_bytes(state: CommState) -> Int:
    """Largest chunk whose staging and inbox both fit in stage_in.

    A chunk of `B` bytes puts `B/L` bytes in each of the `2(N-1)` inbox
    slots, so `B <= cap * L / (2(N-1))`, less a page of alignment slop --
    and never more than `cap`, which the intra-node kernels require anyway.
    At the shapes DDP produces (27 MiB buckets against a 256 MiB region)
    the cap never binds: 256 MiB at 2 nodes, 146 MiB at 8.
    """
    if state.nnodes <= 1:
        return state.cap_bytes
    var denom = 2 * (state.nnodes - 1)
    var b = (state.cap_bytes * state.local_world // denom) - 2 * 4096
    b = min(b, state.cap_bytes)
    if b < 4096:
        return 4096
    return b // 4096 * 4096


def _do_allreduce[
    dtype: DType
](
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    count: Int,
    scale: Float32,
) raises:
    comptime item = size_of[dtype]()
    var max_elems = max(1, _max_chunk_bytes(state) // item)
    var done = 0
    while done < count:
        var chunk = min(max_elems, count - done)
        var off = done * item
        if state.nnodes == 1:
            state.generation += 1
            allreduce[dtype](
                state.ctx,
                stream,
                state.local_rank,
                state.local_world,
                state.regions,
                sendbuff + off,
                recvbuff + off,
                chunk,
                state.cap_bytes,
                scale,
                state.generation,
            )
        else:
            state.generation += 1
            reduce_scatter_stage[dtype](
                state.ctx,
                stream,
                state.local_rank,
                state.local_world,
                state.regions,
                sendbuff + off,
                chunk,
                state.cap_bytes,
                state.generation,
            )
            _inter_node_exchange[dtype](state, stream, raw_stream, chunk)
            state.generation += 1
            allgather_finish[dtype](
                state.ctx,
                stream,
                state.local_rank,
                state.local_world,
                state.regions,
                recvbuff + off,
                chunk,
                state.cap_bytes,
                scale,
                state.generation,
            )
        done += chunk


@export
def ncclAllReduce(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    try:
        var item = _dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        if op != NCCL_SUM and op != NCCL_AVG:
            return NCCL_INVALID_USAGE
        # The payload loops use 16-byte vector loads/stores on
        # in_ptr/out_ptr and fault on a misaligned address (RESULTS.md
        # section 9). Every allocator-returned pointer and every chunk
        # offset this function forms satisfy that (the chunk size is a
        # multiple of 4096 bytes) -- only a mid-tensor view the caller
        # passes directly can violate it, so reject that case here with a
        # clear error instead of letting the kernel raise.
        if Int(sendbuff) % 16 != 0 or Int(recvbuff) % 16 != 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        if state.aborted:
            return NCCL_INVALID_USAGE
        if state.ib != 0 and ib_error(state.ib) != 0:
            return NCCL_REMOTE_ERROR
        state.last_stream = stream
        _ensure_stream_cached(state, stream)
        ref s = state.stream_cache[stream]
        var scale = Float32(1.0)
        if op == NCCL_AVG:
            scale = Float32(1.0) / Float32(state.world)
        if datatype == NCCL_INT32:
            _do_allreduce[DType.int32](
                state, s, stream, Int(sendbuff), Int(recvbuff), Int(count), scale
            )
        elif datatype == NCCL_INT64:
            _do_allreduce[DType.int64](
                state, s, stream, Int(sendbuff), Int(recvbuff), Int(count), scale
            )
        elif datatype == NCCL_FLOAT16:
            _do_allreduce[DType.float16](
                state, s, stream, Int(sendbuff), Int(recvbuff), Int(count), scale
            )
        elif datatype == NCCL_FLOAT32:
            _do_allreduce[DType.float32](
                state, s, stream, Int(sendbuff), Int(recvbuff), Int(count), scale
            )
        else:  # NCCL_BFLOAT16, ruled in by _dtype_item_bytes above
            _do_allreduce[DType.bfloat16](
                state, s, stream, Int(sendbuff), Int(recvbuff), Int(count), scale
            )
        return NCCL_SUCCESS
    except e:
        print("mojoccl: ncclAllReduce failed:", e)
        return NCCL_INTERNAL_ERROR


@export
def ncclBroadcast(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    root: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    try:
        var item = _any_dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        if state.aborted:
            return NCCL_INVALID_USAGE
        if Int(root) < 0 or Int(root) >= state.world:
            return NCCL_INVALID_ARGUMENT
        if state.ib != 0 and ib_error(state.ib) != 0:
            return NCCL_REMOTE_ERROR
        state.last_stream = stream
        _ensure_stream_cached(state, stream)
        ref s = state.stream_cache[stream]
        var total_bytes = Int(count) * item
        if state.nnodes == 1:
            var max_bytes = max(item, state.cap_bytes)
            var done = 0
            while done < total_bytes:
                var chunk = min(max_bytes, total_bytes - done)
                state.generation += 1
                broadcast(
                    state.ctx,
                    s,
                    state.local_rank,
                    Int(root),
                    state.local_world,
                    state.regions,
                    Int(sendbuff) + done,
                    Int(recvbuff) + done,
                    chunk,
                    state.cap_bytes,
                    state.generation,
                )
                done += chunk
            return NCCL_SUCCESS
        _broadcast_multinode(
            state, s, stream, Int(sendbuff), Int(recvbuff), total_bytes, Int(root)
        )
        return NCCL_SUCCESS
    except e:
        print("mojoccl: ncclBroadcast failed:", e)
        return NCCL_INTERNAL_ERROR


def _broadcast_multinode(
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    total_bytes: Int,
    root: Int,
) raises:
    """Root -> one rank per node over IB, then a node-local broadcast.

    The rank that receives on node j is the one sharing the root's
    local_rank, because that is the only rank the root has a queue pair to.
    Every OTHER rank still runs a credit-only exchange with its own
    same-local_rank peers (CREDIT_BYTES). Correct and simple beats fast
    here: broadcast runs at DDP init, on parameters, not in the step.
    """
    var lw = state.local_world
    var root_node = 0
    var root_lr = 0
    for j in range(state.nnodes):
        for l in range(lw):
            if state.rank_at[j * lw + l] == root:
                root_node = j
                root_lr = l
    var i_am_root = state.rank == root
    var i_am_local_root = state.local_rank == root_lr
    var receives = i_am_local_root and state.my_node != root_node
    # The network area holds the root's staged chunk plus both inbox halves,
    # and a half is one slot PER PEER now that the exchange is all-to-all.
    var npeers = ib_npeers(state.ib)
    var max_bytes = max(
        4096,
        (state.cap_bytes // (1 + 2 * npeers) - 2 * 4096) // 4096 * 4096,
    )
    var done = 0
    while done < total_bytes:
        var chunk = min(max_bytes, total_bytes - done)
        var seq = ib_next_seq(state.ib)
        var slot_bytes = _align_up(chunk, 16)
        var inbox_off = state.net_off + _align_up(chunk, 4096)
        var half = npeers * slot_bytes
        if inbox_off + 2 * half > state.net_off + state.cap_bytes:
            raise Error("mojoccl: broadcast inbox does not fit; chunking bug")
        var inbox_base = inbox_off + (seq & 1) * half
        var recv_slot = 0
        if receives:
            recv_slot = root_node if root_node < state.my_node else root_node - 1
        var send_ptr = recvbuff + done
        # Everyone in this local_rank group exchanges; only the root's
        # payload is real (see CREDIT_BYTES).
        var send_addr = state.owned_base
        var send_bytes = CREDIT_BYTES
        var flush_addr = 0
        if i_am_root:
            # The user buffer is not registered, so the root's payload has to
            # be copied into the region before the NIC can read it.
            copy_bytes(
                state.ctx,
                stream,
                state.owned_base + state.net_off,
                sendbuff + done,
                chunk,
            )
            send_addr = state.owned_base + state.net_off
            send_bytes = chunk
            send_ptr = sendbuff + done
        elif receives:
            flush_addr = (
                state.owned_base + inbox_base + recv_slot * slot_bytes
            )
            send_ptr = flush_addr
        ib_enqueue(
            state.ib,
            state.driver,
            Int(raw_stream),
            send_addr,
            send_bytes,
            inbox_base,
            slot_bytes,
            True,
            npeers,
            flush_addr,
            seq,
        )
        state.generation += 1
        broadcast(
            state.ctx,
            stream,
            state.local_rank,
            root_lr,
            lw,
            state.regions,
            send_ptr,
            recvbuff + done,
            chunk,
            state.cap_bytes,
            state.generation,
        )
        done += chunk


@export
def ncclAllGather(
    sendbuff: Int64,
    recvbuff: Int64,
    sendcount: Int64,
    datatype: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    try:
        var item = _any_dtype_item_bytes(datatype)
        if item == 0:
            return NCCL_INVALID_ARGUMENT
        ref state = _comm_ptr(comm)[]
        if state.aborted:
            return NCCL_INVALID_USAGE
        if state.ib != 0 and ib_error(state.ib) != 0:
            return NCCL_REMOTE_ERROR
        state.last_stream = stream
        _ensure_stream_cached(state, stream)
        ref s = state.stream_cache[stream]
        var per_rank_bytes = Int(sendcount) * item
        if state.nnodes == 1:
            var max_bytes = max(item, state.cap_bytes)
            var done = 0
            while done < per_rank_bytes:
                var chunk = min(max_bytes, per_rank_bytes - done)
                state.generation += 1
                allgather(
                    state.ctx,
                    s,
                    state.local_rank,
                    state.local_world,
                    state.regions,
                    Int(sendbuff) + done,
                    Int(recvbuff) + done,
                    chunk,
                    state.cap_bytes,
                    state.generation,
                    stride_bytes=per_rank_bytes,
                )
                done += chunk
            return NCCL_SUCCESS
        _allgather_multinode(
            state, s, stream, Int(sendbuff), Int(recvbuff), per_rank_bytes
        )
        return NCCL_SUCCESS
    except e:
        print("mojoccl: ncclAllGather failed:", e)
        return NCCL_INTERNAL_ERROR


def _allgather_multinode(
    mut state: CommState,
    stream: DeviceStream,
    raw_stream: Int64,
    sendbuff: Int,
    recvbuff: Int,
    per_rank_bytes: Int,
) raises:
    """Node-local allgather, one RDMA exchange of node blocks, then place.

    Every rank ships its whole node block (`local_world * chunk` bytes)
    rather than a share of it: at DDP's 8-byte allgather that is 64 bytes on
    the wire and the simpler code is worth more than the bandwidth. Placement
    reads the global rank of (node, local_rank) out of the bootstrap table --
    torchrun makes it `node * local_world + local_rank`, but nothing here
    assumes so.
    """
    var lw = state.local_world
    var npeers = ib_npeers(state.ib)
    var block_stage = state.owned_base + state.net_off
    # The network area holds one node block (lw * chunk) plus both inbox
    # halves (npeers node blocks each).
    var denom = lw * (1 + 2 * npeers)
    var max_bytes = max(16, (state.cap_bytes // denom - 8192) // 16 * 16)
    var done = 0
    while done < per_rank_bytes:
        var chunk = min(max_bytes, per_rank_bytes - done)
        state.generation += 1
        allgather(
            state.ctx,
            stream,
            state.local_rank,
            lw,
            state.regions,
            sendbuff + done,
            block_stage,
            chunk,
            state.cap_bytes,
            state.generation,
            stride_bytes=chunk,
        )
        var seq = ib_next_seq(state.ib)
        var block = lw * chunk
        var slot_bytes = _align_up(block, 16)
        var inbox_off = state.net_off + _align_up(block, 4096)
        var half = npeers * slot_bytes
        if inbox_off + 2 * half > state.net_off + state.cap_bytes:
            raise Error("mojoccl: allgather inbox does not fit; chunking bug")
        var inbox_base = inbox_off + (seq & 1) * half
        ib_enqueue(
            state.ib,
            state.driver,
            Int(raw_stream),
            block_stage,
            block,
            inbox_base,
            slot_bytes,
            True,
            npeers,
            state.owned_base + inbox_base,
            seq,
        )
        _place_node_block(
            state, stream, recvbuff, block_stage, state.my_node, chunk, done,
            per_rank_bytes,
        )
        var slot = 0
        for j in range(state.nnodes):
            if j == state.my_node:
                continue
            _place_node_block(
                state,
                stream,
                recvbuff,
                state.owned_base + inbox_base + slot * slot_bytes,
                j,
                chunk,
                done,
                per_rank_bytes,
            )
            slot += 1
        done += chunk


def _place_node_block(
    mut state: CommState,
    stream: DeviceStream,
    recvbuff: Int,
    src: Int,
    node: Int,
    chunk: Int,
    done: Int,
    per_rank_bytes: Int,
) raises:
    var offs = StaticTuple[Int64, MAX_WORLD](fill=0)
    for l in range(state.local_world):
        offs[l] = Int64(
            state.rank_at[node * state.local_world + l] * per_rank_bytes + done
        )
    place_blocks(
        state.ctx, stream, recvbuff, src, offs, chunk, state.local_world
    )


# ---------------------------------------------------------------------------
# Not implemented: DDP on GPT-2 needs only AllReduce/Broadcast/AllGather (+
# barrier, which routes to gloo -- see process_group.py). Returning
# ncclInvalidUsage rather than silently mis-computing is the point.
# ---------------------------------------------------------------------------


@export
def ncclReduce(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    root: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return NCCL_INVALID_USAGE


@export
def ncclReduceScatter(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return NCCL_INVALID_USAGE


@export
def ncclSend(
    sendbuff: Int64,
    count: Int64,
    datatype: Int32,
    peer: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return NCCL_INVALID_USAGE


@export
def ncclRecv(
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    peer: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return NCCL_INVALID_USAGE
