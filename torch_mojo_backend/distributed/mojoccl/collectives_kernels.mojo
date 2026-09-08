# STAND-IN collectives kernel for mojoccl.
#
# Correctness over speed, as directed: single-thread-block copy/reduce
# kernels plus a page-sized signal area and a generation-based barrier built
# on plain system-scope Atomic release/acquire (the flag pattern
# proto/ipc_probe.mojo measured working, PASS at §5.3 of
# docs/mojo_collectives_feasibility.md in the main worktree) -- not MAX's
# comm.Signal, whose embedded Lamport region alone is ~24.75 MiB/rank, far
# more than this needs. This file is expected to be replaced wholesale by
# the production kernel (kernel/collectives_kernels.mojo, written in
# parallel against this same function-signature contract); nothing outside
# this file should need to change on that swap.
#
# Region layout per rank (see mojoccl.mojo): [0, signal_bytes()) signal area,
# then stage_in of cap_bytes, then stage_out of cap_bytes. `regions[r]` is
# rank r's region base as mapped in THIS process (own allocation for
# r == rank, an opened IPC mapping otherwise).

from std.atomic import Atomic, Ordering
from std.builtin.device_passable import DevicePassable
from std.ffi import _get_global_or_null, external_call
from max.gpu.sync import barrier
from std.gpu import thread_idx, MAX_THREADS_PER_BLOCK_METADATA
from std.memory.alloc import unsafe_alloc
from std.sys import get_accum_type
from std.time import global_perf_counter_ns
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream, DeviceBuffer

from driver import zero_bytes

comptime MAX_WORLD = 8
comptime BLOCK = 256
# One page: [0] error word (UInt64), [8] barrier counter (UInt64), padding.
comptime SIG_BYTES = 4096
comptime ERR_OFF = 0
comptime BAR_OFF = 8
comptime BARRIER_TIMEOUT_NS: UInt64 = 30_000_000_000


def signal_bytes() -> Int:
    return SIG_BYTES


def error_offset() -> Int:
    return ERR_OFF


def region_init(ctx: DeviceContext, region: Int) raises:
    """Zeros the signal area of a freshly allocated region.

    No `stream` parameter (unlike the collectives below): this runs once per
    rank at communicator creation, on `ctx`'s own queue, before any user
    stream touches the region -- zero_bytes blocks, so every later
    collective (issued on the caller's external stream) is guaranteed to see
    the zeroed area.
    """
    zero_bytes(ctx, region, SIG_BYTES)


@always_inline
def _err_ptr(region: Int64) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=Int(region) + ERR_OFF
    )


@always_inline
def _bar_ptr(region: Int64) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](
        unsafe_from_address=Int(region) + BAR_OFF
    )


@always_inline
def _gen_barrier(
    regions: StaticTuple[Int64, MAX_WORLD],
    rank: Int32,
    world: Int32,
    target: UInt64,
):
    """A full, symmetric cross-rank barrier: every participant publishes
    `target` into its own region and waits for every OTHER participant to
    reach at least `target` too, before any of them may proceed.

    Symmetric on purpose (publish AND wait, both phases of every call): a
    one-way "publish and move on" would let a fast rank start overwriting
    its stage buffers for the NEXT generation while a slow peer is still
    reading THIS generation's data out of them. `target` is derived from the
    caller-supplied generation counter, strictly increasing per
    communicator, so no double-buffering of the counter itself is needed --
    every call in the process's lifetime gets a fresh target.
    """
    if thread_idx.x == 0:
        Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
            _bar_ptr(regions[Int(rank)]), target
        )
    if Int(thread_idx.x) < Int(world):
        var peer_ptr = _bar_ptr(regions[Int(thread_idx.x)])
        var t0 = global_perf_counter_ns()
        while (
            Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](peer_ptr)
            < target
        ):
            if global_perf_counter_ns() - t0 > BARRIER_TIMEOUT_NS:
                Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
                    _err_ptr(regions[Int(rank)]), UInt64(1)
                )
                break
    barrier()


@always_inline
def _enqueue_cached_stream[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *Ts: DevicePassable,
](
    ctx: DeviceContext,
    stream: DeviceStream,
    key: String,
    threads: Int,
    *args: *Ts,
) raises:
    """Enqueue `func` (grid of 1 block) on the caller's `stream`, compiling
    it at most once per process and context.

    Self-contained copy of eager_kernels/op_utils's `_enqueue_cached`
    pattern (this library builds and loads standalone, outside that
    package): compile once via `ctx`, cache the `DeviceFunction` in the
    process-global registry, enqueue onto `stream` on every call after.
    """
    var name = String(t"MOJOCCL_KERNEL_{key}_{ctx.id()}")
    comptime FuncT = type_of(ctx.compile_function[func]())
    var global_ptr = _get_global_or_null(name)
    if global_ptr:
        var fptr = global_ptr.value().unsafe_bitcast[FuncT]()
        stream.enqueue_function(
            fptr[], *args, grid_dim=(1, 1, 1), block_dim=(threads,)
        )
        return
    var compiled = ctx.compile_function[func]()
    var fptr = unsafe_alloc[FuncT](1)
    fptr.unsafe_write(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), fptr.unsafe_bitcast[NoneType]()
    )
    stream.enqueue_function(
        fptr[], *args, grid_dim=(1, 1, 1), block_dim=(threads,)
    )


# ---------------------------------------------------------------------------
# allreduce: copy-in, barrier, every rank independently sums every peer's
# stage_in (no reduce-scatter partitioning -- simplest correct thing, per
# "plain copy-in / RS+AG / copy-out is fine"), barrier, copy-out. Generic
# Float64 accumulation covers all five dtypes with one code path; SUM is
# exact for the small integer values these tests use, AVG divides via
# `scale`.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
def _allreduce_kernel[
    dtype: DType
](
    regions: StaticTuple[Int64, MAX_WORLD],
    rank: Int32,
    world: Int32,
    in_ptr: Int64,
    out_ptr: Int64,
    numel: Int32,
    cap_bytes: Int32,
    scale: Float32,
    generation: UInt64,
):
    var my_region = Int(regions[Int(rank)])
    var stage_in = Pointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=my_region + SIG_BYTES
    )
    var src = Pointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(in_ptr)
    )
    var n = Int(numel)
    var tid = Int(thread_idx.x)
    for i in range(tid, n, BLOCK):
        stage_in[unsafe_offset=i] = src[unsafe_offset=i]

    _gen_barrier(regions, rank, world, 2 * generation)

    var stage_out = Pointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=my_region + SIG_BYTES + Int(cap_bytes)
    )
    var w = Int(world)
    var fscale = Float64(scale)
    for i in range(tid, n, BLOCK):
        var acc: Float64 = 0
        for p in range(w):
            var peer_in = Pointer[Scalar[dtype], MutAnyOrigin](
                unsafe_from_address=Int(regions[p]) + SIG_BYTES
            )
            acc += Float64(peer_in[unsafe_offset=i])
        stage_out[unsafe_offset=i] = Scalar[dtype](acc * fscale)

    _gen_barrier(regions, rank, world, 2 * generation + 1)

    var dst = Pointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(out_ptr)
    )
    for i in range(tid, n, BLOCK):
        dst[unsafe_offset=i] = stage_out[unsafe_offset=i]


def allreduce[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    numel: Int,
    cap_bytes: Int,
    scale: Float32,
    generation: Int,
) raises:
    var regions64 = StaticTuple[Int64, MAX_WORLD](fill=0)
    for i in range(MAX_WORLD):
        regions64[i] = Int64(regions[i])
    comptime kern = _allreduce_kernel[dtype]
    _enqueue_cached_stream[kern](
        ctx,
        stream,
        String("allreduce_") + String(dtype),
        BLOCK,
        regions64,
        Int32(rank),
        Int32(world),
        Int64(in_ptr),
        Int64(out_ptr),
        Int32(numel),
        Int32(cap_bytes),
        scale,
        UInt64(generation),
    )


# ---------------------------------------------------------------------------
# broadcast: root copies its send buffer into its own stage_in, barrier,
# every rank (root included) copies root's stage_in into its own recv
# buffer, barrier. Byte-granular: NCCL's broadcast moves opaque bytes, no
# reduction, so there is no dtype to specialize on.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
def _broadcast_kernel(
    regions: StaticTuple[Int64, MAX_WORLD],
    rank: Int32,
    root: Int32,
    world: Int32,
    send_ptr: Int64,
    recv_ptr: Int64,
    nbytes: Int32,
    generation: UInt64,
):
    var my_region = Int(regions[Int(rank)])
    var stage = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=my_region + SIG_BYTES
    )
    var n = Int(nbytes)
    var tid = Int(thread_idx.x)
    if rank == root:
        var src = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(send_ptr)
        )
        for i in range(tid, n, BLOCK):
            stage[unsafe_offset=i] = src[unsafe_offset=i]

    _gen_barrier(regions, rank, world, 2 * generation)

    var root_stage = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(regions[Int(root)]) + SIG_BYTES
    )
    var dst = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(recv_ptr))
    for i in range(tid, n, BLOCK):
        dst[unsafe_offset=i] = root_stage[unsafe_offset=i]

    _gen_barrier(regions, rank, world, 2 * generation + 1)


def broadcast(
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    root: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    send_ptr: Int,
    recv_ptr: Int,
    nbytes: Int,
    cap_bytes: Int,
    generation: Int,
) raises:
    if nbytes > cap_bytes:
        raise Error("mojoccl: broadcast nbytes exceeds the region capacity")
    var regions64 = StaticTuple[Int64, MAX_WORLD](fill=0)
    for i in range(MAX_WORLD):
        regions64[i] = Int64(regions[i])
    comptime kern = _broadcast_kernel
    _enqueue_cached_stream[kern](
        ctx,
        stream,
        String("broadcast"),
        BLOCK,
        regions64,
        Int32(rank),
        Int32(root),
        Int32(world),
        Int64(send_ptr),
        Int64(recv_ptr),
        Int32(nbytes),
        UInt64(generation),
    )


# ---------------------------------------------------------------------------
# allgather: every rank copies its chunk into its own stage_in, barrier,
# every rank copies every peer's stage_in into the right slice of its own
# output, barrier.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
def _allgather_kernel(
    regions: StaticTuple[Int64, MAX_WORLD],
    rank: Int32,
    world: Int32,
    in_ptr: Int64,
    out_ptr: Int64,
    nbytes: Int32,
    stride_bytes: Int32,
    generation: UInt64,
):
    var my_region = Int(regions[Int(rank)])
    var stage = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=my_region + SIG_BYTES
    )
    var src = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(in_ptr))
    var n = Int(nbytes)
    var tid = Int(thread_idx.x)
    for i in range(tid, n, BLOCK):
        stage[unsafe_offset=i] = src[unsafe_offset=i]

    _gen_barrier(regions, rank, world, 2 * generation)

    var dst = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(out_ptr))
    var w = Int(world)
    var stride = Int(stride_bytes)
    for p in range(w):
        var peer_stage = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(regions[p]) + SIG_BYTES
        )
        var base = p * stride
        for i in range(tid, n, BLOCK):
            dst[unsafe_offset=base + i] = peer_stage[unsafe_offset=i]

    _gen_barrier(regions, rank, world, 2 * generation + 1)


def allgather(
    ctx: DeviceContext,
    stream: DeviceStream,
    rank: Int,
    world: Int,
    regions: StaticTuple[Int, MAX_WORLD],
    in_ptr: Int,
    out_ptr: Int,
    nbytes_per_rank: Int,
    cap_bytes: Int,
    generation: Int,
    stride_bytes: Int = -1,
) raises:
    """`stride_bytes` (default `nbytes_per_rank`) is the OUTPUT layout's true
    per-rank size, in bytes, separate from `nbytes_per_rank` (how much THIS
    call moves). They differ only when the caller chunks one rank's
    contribution across several calls: each chunk still lands at
    `peer * stride_bytes + chunk_offset` in the flat `[rank0 | rank1 | ...]`
    output, never at `peer * nbytes_per_rank`. A one-call (unchunked)
    allgather needs no stride argument at all.
    """
    if nbytes_per_rank > cap_bytes:
        raise Error(
            "mojoccl: allgather nbytes_per_rank exceeds the region capacity"
        )
    var stride = nbytes_per_rank if stride_bytes < 0 else stride_bytes
    var regions64 = StaticTuple[Int64, MAX_WORLD](fill=0)
    for i in range(MAX_WORLD):
        regions64[i] = Int64(regions[i])
    comptime kern = _allgather_kernel
    _enqueue_cached_stream[kern](
        ctx,
        stream,
        String("allgather"),
        BLOCK,
        regions64,
        Int32(rank),
        Int32(world),
        Int64(in_ptr),
        Int64(out_ptr),
        Int32(nbytes_per_rank),
        Int32(stride),
        UInt64(generation),
    )
