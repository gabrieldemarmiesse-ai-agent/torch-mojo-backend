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
# One process per GPU, one region per rank, raw driver-owned memory shared
# with legacy IPC (driver.mojo) -- MAX's own allocator memory cannot be
# exported that way (see docs/mojo_collectives_feasibility.md in the main
# worktree, §5.6). Bootstrap (the handle exchange ncclCommInitRank itself
# must do) rides a /dev/shm directory encoded in the 128-byte unique id
# (bootstrap.mojo); the collectives are the STAND-IN kernel
# (collectives_kernels.mojo), swapped out wholesale once the production
# kernel lands.

from std.collections import Dict
from std.ffi import OwnedDLHandle
from std.gpu import global_idx
from std.memory.alloc import unsafe_alloc
from std.os import getenv
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
    create_rendezvous_dir,
    decode_dir,
    encode_dir,
    read_rank_handle,
    remove_rendezvous_dir,
    wait_for_done,
    wait_for_rank,
    write_rank_done,
    write_rank_handle,
)
from collectives_kernels import (
    MAX_WORLD,
    allgather,
    allreduce,
    broadcast,
    error_offset,
    region_init,
    signal_bytes,
)

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
# hundred bytes; only the GPU region and peer IPC mappings -- the resources
# that matter -- are released there). `regions` is padded to MAX_WORLD with
# zeros; only indices < world are ever read.
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
        var dir = create_rendezvous_dir()
        encode_dir(dir, uid_out)
        return NCCL_SUCCESS
    except e:
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
    if Int(nranks) > MAX_WORLD or Int(nranks) < 1:
        return NCCL_INVALID_ARGUMENT
    if Int(rank) < 0 or Int(rank) >= Int(nranks):
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
        var dir = decode_dir(_any(idbuf))

        # Everything below either succeeds together or is unwound together:
        # a failure past this point (region alloc, handle exchange, the
        # peer-open loop, the done-marker wait) is caught here so rank 0 can
        # remove the rendezvous directory it created in ncclGetUniqueId --
        # otherwise it and every rank's handle/done file leak in /dev/shm
        # forever, since no other rank will ever revisit this path.
        try:
            var lib = open_driver()
            var ordinal = current_device_ordinal(lib)
            var ctx = DeviceContext(device_id=ordinal)

            var cap_bytes = _region_cap_bytes()
            # A positive multiple of 4096 (the production kernel's own
            # precondition, RESULTS.md §9) is what keeps every per-chunk
            # offset `ncclAllReduce` forms (multiples of cap_bytes, one full
            # chunk at a time) 16-byte aligned for every supported dtype --
            # 4096 divides evenly by 2, 4 and 8. A misconfigured
            # MOJOCCL_REGION_MB (e.g. 0) would otherwise degrade chunk
            # offsets to non-16-byte-aligned single-element steps.
            if cap_bytes <= 0 or cap_bytes % 4096 != 0:
                if Int(rank) == 0:
                    try:
                        remove_rendezvous_dir(dir, Int(nranks))
                    except:
                        pass
                return NCCL_INVALID_ARGUMENT
            var region_bytes = signal_bytes() + 2 * cap_bytes
            var base = alloc_region(lib, region_bytes)
            region_init(ctx, base)

            var my_handle = unsafe_alloc[UInt8](HANDLE_BYTES)
            get_handle(lib, base, _any(my_handle))
            write_rank_handle(dir, Int(rank), ordinal, _any(my_handle))

            var timeout_s = _bootstrap_timeout_s()
            var regions = StaticTuple[Int, MAX_WORLD](fill=0)
            regions[Int(rank)] = base
            var peer_handle = unsafe_alloc[UInt8](HANDLE_BYTES)
            for r in range(Int(nranks)):
                if r == Int(rank):
                    continue
                try:
                    wait_for_rank(dir, r, timeout_s)
                    _ = read_rank_handle(dir, r, _any(peer_handle))
                    regions[r] = open_handle(lib, _any(peer_handle))
                except:
                    # A peer past this one never got opened: close whatever
                    # peers < r WERE opened and free our own region before
                    # propagating, so a failed init leaks no IPC mappings.
                    for r2 in range(Int(nranks)):
                        if r2 != Int(rank) and regions[r2] != 0:
                            try:
                                close_handle(lib, regions[r2])
                            except:
                                pass
                    try:
                        free_region(lib, base)
                    except:
                        pass
                    raise

            # Done protocol: every rank marks itself done once it has opened
            # every peer's handle; rank 0 (and only rank 0) waits for all
            # `nranks` markers and then removes `dir` -- otherwise
            # /dev/shm/mojoccl-* directories and their per-rank handle files
            # accumulate forever on a shared node (mkdtemp never cleans up
            # after itself). Other ranks return as soon as their own marker
            # is written; they do not wait for the directory to disappear.
            write_rank_done(dir, Int(rank))
            if Int(rank) == 0:
                for r in range(Int(nranks)):
                    if r != 0:
                        wait_for_done(dir, r, timeout_s)
                try:
                    remove_rendezvous_dir(dir, Int(nranks))
                except:
                    pass  # a leftover /dev/shm dir is a nuisance, not a correctness bug

            var state = CommState(
                rank=Int(rank),
                world=Int(nranks),
                ordinal=ordinal,
                ctx=ctx,
                driver=lib^,
                cap_bytes=cap_bytes,
                regions=regions,
                owned_base=base,
            )
            var handle_ptr = unsafe_alloc[CommState](1)
            handle_ptr.unsafe_write(state^)
            comm_out[] = Int64(Int(handle_ptr))
            return NCCL_SUCCESS
        except:
            if Int(rank) == 0:
                try:
                    remove_rendezvous_dir(dir, Int(nranks))
                except:
                    pass
            raise
    except:
        return NCCL_INTERNAL_ERROR


# ---------------------------------------------------------------------------
# Communicator lifecycle
# ---------------------------------------------------------------------------


@export
def ncclCommDestroy(comm: Int64) abi("C") -> Int32:
    try:
        var ptr = _comm_ptr(comm)
        ref state = ptr[]
        if not state.aborted:
            state.ctx.synchronize()
            for r in range(state.world):
                if r != state.rank:
                    close_handle(state.driver, state.regions[r])
            free_region(state.driver, state.owned_base)
        return NCCL_SUCCESS
    except e:
        return NCCL_INTERNAL_ERROR


@export
def ncclCommAbort(comm: Int64) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    # No wait, no cleanup of GPU resources here (a dead peer may hang
    # forever inside its own barrier spin) -- matches ncclCommAbort's
    # documented "don't wait" contract. ncclCommDestroy is never called
    # after abort() by nccl.py's NcclComm.
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
        var word = _read_error_word(state.ctx, s, state.regions[state.rank])
        err_out[] = NCCL_REMOTE_ERROR if Int(word) != 0 else NCCL_SUCCESS
        return NCCL_SUCCESS
    except e:
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
        # The production kernel's payload loops use 16-byte vector
        # loads/stores on in_ptr/out_ptr and fault on a misaligned address
        # (RESULTS.md §9). Every allocator-returned pointer and every chunk
        # offset this function forms satisfy that (cap_bytes is validated a
        # multiple of 4096 at init) -- only a mid-tensor view the caller
        # passes directly can violate it, so reject that case here with a
        # clear error instead of letting the kernel raise.
        if Int(sendbuff) % 16 != 0 or Int(recvbuff) % 16 != 0:
            return NCCL_INVALID_ARGUMENT
        var ptr = _comm_ptr(comm)
        ref state = ptr[]
        if state.aborted:
            return NCCL_INVALID_USAGE
        state.last_stream = stream
        _ensure_stream_cached(state, stream)
        ref s = state.stream_cache[stream]
        var scale = Float32(1.0)
        if op == NCCL_AVG:
            scale = Float32(1.0) / Float32(state.world)
        var max_elems = max(1, state.cap_bytes // item)
        var total = Int(count)
        var done = 0
        while done < total:
            var chunk = min(max_elems, total - done)
            state.generation += 1
            var off = done * item
            if datatype == NCCL_INT32:
                allreduce[DType.int32](
                    state.ctx,
                    s,
                    state.rank,
                    state.world,
                    state.regions,
                    Int(sendbuff) + off,
                    Int(recvbuff) + off,
                    chunk,
                    state.cap_bytes,
                    scale,
                    state.generation,
                )
            elif datatype == NCCL_INT64:
                allreduce[DType.int64](
                    state.ctx,
                    s,
                    state.rank,
                    state.world,
                    state.regions,
                    Int(sendbuff) + off,
                    Int(recvbuff) + off,
                    chunk,
                    state.cap_bytes,
                    scale,
                    state.generation,
                )
            elif datatype == NCCL_FLOAT16:
                allreduce[DType.float16](
                    state.ctx,
                    s,
                    state.rank,
                    state.world,
                    state.regions,
                    Int(sendbuff) + off,
                    Int(recvbuff) + off,
                    chunk,
                    state.cap_bytes,
                    scale,
                    state.generation,
                )
            elif datatype == NCCL_FLOAT32:
                allreduce[DType.float32](
                    state.ctx,
                    s,
                    state.rank,
                    state.world,
                    state.regions,
                    Int(sendbuff) + off,
                    Int(recvbuff) + off,
                    chunk,
                    state.cap_bytes,
                    scale,
                    state.generation,
                )
            else:  # NCCL_BFLOAT16, ruled in by _dtype_item_bytes above
                allreduce[DType.bfloat16](
                    state.ctx,
                    s,
                    state.rank,
                    state.world,
                    state.regions,
                    Int(sendbuff) + off,
                    Int(recvbuff) + off,
                    chunk,
                    state.cap_bytes,
                    scale,
                    state.generation,
                )
            done += chunk
        return NCCL_SUCCESS
    except:
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
        var ptr = _comm_ptr(comm)
        ref state = ptr[]
        if state.aborted:
            return NCCL_INVALID_USAGE
        if Int(root) < 0 or Int(root) >= state.world:
            return NCCL_INVALID_ARGUMENT
        state.last_stream = stream
        _ensure_stream_cached(state, stream)
        ref s = state.stream_cache[stream]
        var total_bytes = Int(count) * item
        var max_bytes = max(item, state.cap_bytes)
        var done = 0
        while done < total_bytes:
            var chunk = min(max_bytes, total_bytes - done)
            state.generation += 1
            broadcast(
                state.ctx,
                s,
                state.rank,
                Int(root),
                state.world,
                state.regions,
                Int(sendbuff) + done,
                Int(recvbuff) + done,
                chunk,
                state.cap_bytes,
                state.generation,
            )
            done += chunk
        return NCCL_SUCCESS
    except:
        return NCCL_INTERNAL_ERROR


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
        var ptr = _comm_ptr(comm)
        ref state = ptr[]
        if state.aborted:
            return NCCL_INVALID_USAGE
        state.last_stream = stream
        _ensure_stream_cached(state, stream)
        ref s = state.stream_cache[stream]
        var per_rank_bytes = Int(sendcount) * item
        var max_bytes = max(item, state.cap_bytes)
        var done = 0
        while done < per_rank_bytes:
            var chunk = min(max_bytes, per_rank_bytes - done)
            state.generation += 1
            allgather(
                state.ctx,
                s,
                state.rank,
                state.world,
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
    except:
        return NCCL_INTERNAL_ERROR


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
