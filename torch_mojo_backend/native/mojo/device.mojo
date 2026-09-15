"""Devices, streams, events and memory of the native backend.

One `Dev` per mojo index: the accelerators MAX enumerates, then the MAX CPU
device last (so `mojo:N` is the CPU on an N-GPU box, as before). A stream is a
MAX stream of the device's base context; kernels get the context *view* bound
to it (`DeviceContext.select_stream`), so real multi-stream execution costs
nothing at launch time. Everything here runs under the shim's mutex.
"""
from std.ffi import _get_global_or_null, c_char, c_size_t, external_call
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.atomic.atomic import Atomic
from std.time import perf_counter_ns
from std.os import getenv

from max.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    DeviceEvent,
    DeviceStream,
    HostBuffer,
)

from vendor import Vendor, raw_stream

comptime BufP = Pointer[Buf, MutUntrackedOrigin]
comptime PinnedP = Pointer[Pinned, MutUntrackedOrigin]
comptime EvP = Pointer[Ev, MutUntrackedOrigin]
comptime StagingP = Pointer[Staging, MutUntrackedOrigin]
comptime U8P = Pointer[UInt8, MutUntrackedOrigin]
comptime POOL_STREAMS = 4


struct MemoryStat(Copyable, Movable):
    var current: Int64
    var peak: Int64
    var allocated: Int64
    var freed: Int64

    def __init__(out self):
        self.current = 0
        self.peak = 0
        self.allocated = 0
        self.freed = 0

    def add(mut self, n: Int64):
        self.current += n
        self.peak = max(self.peak, self.current)
        self.allocated += n

    def remove(mut self, n: Int64):
        self.current -= n
        self.freed += n

    def reset_accumulated(mut self):
        self.allocated = 0
        self.freed = 0

    def write(self, dst: Pointer[Int64, MutUntrackedOrigin]):
        dst[unsafe_offset=0] = self.current
        dst[unsafe_offset=1] = self.peak
        dst[unsafe_offset=2] = self.allocated
        dst[unsafe_offset=3] = self.freed


@fieldwise_init
struct Properties(Movable):
    var values: List[Int64]
    var text: String


def _warn(what: StaticString, e: Error):
    print("torch-mojo-backend: ", what, ": ", String(e))


struct Dev(Movable):
    var ctx: DeviceContext
    var views: List[
        DeviceContext
    ]  # views[s] submits to stream s; views[0] is ctx
    var streams: List[DeviceStream]  # created streams, kept alive
    var raw: List[Int]  # vendor stream handle per stream (0 when unknown)
    var api: String
    var is_cpu: Bool
    var pool: List[Int]
    var pool_next: Int
    var staging: List[Int]  # pending pageable-H2D staging boxes (addresses)
    var allocated_bytes: MemoryStat
    var allocation: MemoryStat
    var num_device_alloc: Int64
    var num_device_free: Int64
    var num_alloc_retries: Int64
    var num_ooms: Int64
    var properties: Optional[Properties]
    var peers: Dict[Int, Bool]  # access from this device to each peer
    var quarantine: Bool  # failed drain: allocations must not be reused

    def __init__(out self, var ctx: DeviceContext, is_cpu: Bool) raises:
        self.api = ctx.api()
        self.is_cpu = is_cpu
        self.views = List[DeviceContext]()
        self.views.append(ctx)
        self.streams = List[DeviceStream]()
        self.raw = List[Int]()
        self.raw.append(
            0
        )  # filled once the vendor driver exists (init_backend)
        self.pool = List[Int]()
        self.pool_next = 0
        self.staging = List[Int]()
        self.allocated_bytes = MemoryStat()
        self.allocation = MemoryStat()
        self.num_device_alloc = 0
        self.num_device_free = 0
        self.num_alloc_retries = 0
        self.num_ooms = 0
        self.properties = None
        self.peers = Dict[Int, Bool]()
        self.quarantine = False
        self.ctx = ctx^

    def view(self, s: Int) raises -> DeviceContext:
        if s < 0 or s >= len(self.views):
            raise Error("invalid stream id ", s, " on mojo device ", self.api)
        return self.views[s]


@fieldwise_init
struct Backend(Movable):
    var devices: List[Dev]
    var vendor: Optional[Vendor]
    var n_accel: Int
    var test_peer_copy: String
    var test_peer_gate: Int
    var pinned: List[Int]  # Pinned box addresses, sorted by buffer base


comptime BACKEND_GLOBAL = "TMB_NATIVE_BACKEND"


@always_inline
def be() -> Pointer[Backend, MutUntrackedOrigin]:
    var p = _get_global_or_null(BACKEND_GLOBAL)
    return p.value().unsafe_bitcast[Backend]()


def dev(i: Int) raises -> Pointer[Dev, MutUntrackedOrigin]:
    ref devs = be()[].devices
    if i < 0 or i >= len(devs):
        raise Error("invalid mojo device index ", i)
    return Pointer(to=devs[i]).unsafe_origin_cast[MutUntrackedOrigin]()


def stream_ctx(device: Int, stream: Int) raises -> DeviceContext:
    return dev(device)[].view(stream)


def _probe_api() -> Tuple[String, Int]:
    """Which accelerator api MAX drives here and how many devices it has,
    asked at run time: this library is built once for every accelerator
    (the compile-time default api would tie it to the build machine's)."""
    var apis = List[String]()
    apis.append("cuda")
    apis.append("hip")
    apis.append("metal")
    for api in apis:
        # 0 when MAX has no support for that api on this machine
        var n = DeviceContext.number_of_devices(api=api)
        if n > 0:
            return (api, n)
    return (String("cpu"), 0)


def init_backend() raises -> Int:
    """Enumerate devices once; returns the mojo device count (GPUs + CPU)."""
    if _get_global_or_null(BACKEND_GLOBAL):
        return len(be()[].devices)
    var probed = _probe_api()
    var api = probed[0]
    var n = probed[1]
    var vendor: Optional[Vendor] = None
    if api == "cuda" or api == "hip":
        try:
            vendor = Vendor(api)
        except e:
            _warn(
                "vendor driver unavailable, events fall back to host timing", e
            )
    var devs = List[Dev]()
    for i in range(n):
        devs.append(Dev(DeviceContext(i, api=api), False))
    devs.append(Dev(DeviceContext(api="cpu"), True))
    var gate = getenv("TORCH_MOJO_BACKEND_TEST_PEER_GATE_FD")
    var gate_fd = Int(gate) if gate != "" else -1
    var modes = getenv("TORCH_MOJO_BACKEND_TEST_PEER_COPY")
    # Delimit once so mode checks match whole tokens without getenv or splitting.
    var test_peer_copy = "," + modes + "," if modes != "" else ""
    var box = unsafe_alloc[Backend](1)
    box.unsafe_write(
        Backend(
            devs^,
            vendor^,
            n,
            test_peer_copy^,
            gate_fd,
            List[Int](),
        )
    )
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(BACKEND_GLOBAL), box.unsafe_bitcast[NoneType]()
    )
    if be()[].vendor:
        for i in range(n):
            var d = dev(i)
            try:
                d[].raw[0] = raw_stream(be()[].vendor.value(), d[].ctx)
            except e:
                _warn("no raw stream handle", e)
    return n + 1


# --- memory -------------------------------------------------------------------


@fieldwise_init
struct Buf(Movable):
    var buf: DeviceBuffer[DType.uint8]
    var device: Int
    var nbytes: Int
    var stream: Int  # owner stream: where MAX orders the release
    var users: List[Int]  # other streams that used the buffer (record_stream)


def _create(
    ctx: DeviceContext, nbytes: Int
) raises -> DeviceBuffer[DType.uint8]:
    return ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))


def _create_retry(
    d: Pointer[Dev, MutUntrackedOrigin], ctx: DeviceContext, nbytes: Int
) raises -> DeviceBuffer[DType.uint8]:
    """MAX's allocator can fail transiently on a full arena (modular#6801):
    drain the device once and retry before reporting out-of-memory."""
    try:
        return _create(ctx, nbytes)
    except e:
        for i in range(len(d[].views)):
            d[].views[i].synchronize()
        d[].num_alloc_retries += 1
        try:
            return _create(ctx, nbytes)
        except retry_error:
            d[].num_ooms += 1
            raise retry_error^


def h_alloc(
    nbytes: Int,
    device: Int32,
    stream: Int64,
    data: Pointer[Int, MutUntrackedOrigin],
) abi("C") -> Int:
    try:
        var d = dev(Int(device))
        var ctx = d[].view(Int(stream))
        var buf = _create_retry(d, ctx, nbytes)
        data[] = Int(buf.unsafe_ptr())
        var box = unsafe_alloc[Buf](1)
        box.unsafe_write(
            Buf(buf^, Int(device), nbytes, Int(stream), List[Int]())
        )
        d[].allocated_bytes.add(Int64(nbytes))
        d[].allocation.add(1)
        d[].num_device_alloc += 1
        if be()[].test_peer_copy != "":
            print("P2P_ALLOC", Int(device), data[], nbytes)
        return Int(box)
    except e:
        set_error(String(e))
        return 0


def h_free(handle: Int) abi("C"):
    if handle == 0:
        return
    var box = BufP(unsafe_from_address=handle)
    if be()[].devices[box[].device].quarantine:
        if be()[].test_peer_copy != "":
            print("P2P_RETAIN", box[].device, Int(box[].buf.unsafe_ptr()))
        return
    # MAX releases the block stream-ordered on the OWNER stream only. Every
    # other stream that used it (torch.Tensor.record_stream) gets fenced now,
    # at release time: the owner waits for all of that stream's work so far.
    if len(box[].users) > 0:
        try:
            var d = dev(box[].device)
            var owner = d[].view(box[].stream)
            for i in range(len(box[].users)):
                owner.enqueue_wait_for(d[].view(box[].users[i]))
        except e:
            # Ordering could not be established: leak the block rather than
            # let MAX reuse memory another stream may still be reading.
            set_error(String(e))
            return
    # Account for storage returned to MAX, whose own arena and stream-ordered
    # release are opaque to us. A deliberately leaked block above stays live.
    ref d = be()[].devices[box[].device]
    d.allocated_bytes.remove(Int64(box[].nbytes))
    d.allocation.remove(1)
    d.num_device_free += 1
    if be()[].test_peer_copy != "":
        print("P2P_FREE", box[].device, Int(box[].buf.unsafe_ptr()))
    var moved = box.unsafe_take_pointee()
    box.unsafe_free()
    _ = moved^


def h_mem_stats(
    device: Int32, dst: Pointer[Int64, MutUntrackedOrigin], n: Int32
) abi("C"):
    try:
        var d = dev(Int(device))
        if n != 20:
            raise Error("memory stats ABI mismatch: expected 20 slots")
        # tmb.h's flat contract. No rounding, splitting or arena of our own:
        # requested and reserved equal allocated, entirely in the 'all' pool.
        d[].allocated_bytes.write(dst)
        d[].allocation.write(dst.unsafe_offset(4))
        d[].allocated_bytes.write(dst.unsafe_offset(8))
        d[].allocated_bytes.write(dst.unsafe_offset(12))
        dst[unsafe_offset=16] = d[].num_device_alloc
        dst[unsafe_offset=17] = d[].num_device_free
        dst[unsafe_offset=18] = d[].num_alloc_retries
        dst[unsafe_offset=19] = d[].num_ooms
    except e:
        set_error(String(e))


def h_mem_reset_peak(device: Int32) abi("C"):
    try:
        var d = dev(Int(device))
        d[].allocated_bytes.peak = d[].allocated_bytes.current
        d[].allocation.peak = d[].allocation.current
    except e:
        set_error(String(e))


def h_mem_reset_accumulated(device: Int32) abi("C"):
    try:
        var d = dev(Int(device))
        d[].allocated_bytes.reset_accumulated()
        d[].allocation.reset_accumulated()
        d[].num_device_alloc = 0
        d[].num_device_free = 0
        d[].num_alloc_retries = 0
        d[].num_ooms = 0
    except e:
        set_error(String(e))


def h_empty_cache() abi("C"):
    # Only release completed host staging buffers. MAX owns the device arena
    # and exposes no trim API; never synchronize to imitate a cache flush.
    for i in range(len(be()[].devices)):
        _drain_staging(
            Pointer(to=be()[].devices[i]).unsafe_origin_cast[
                MutUntrackedOrigin
            ]()
        )


def h_mem_get_info(
    device: Int32,
    free: Pointer[c_size_t, MutUntrackedOrigin],
    total: Pointer[c_size_t, MutUntrackedOrigin],
) abi("C") -> Int32:
    try:
        var d = dev(Int(device))
        var info: Tuple[c_size_t, c_size_t]
        try:
            info = d[].ctx.get_memory_info()
        except e:
            set_error(
                "MAX "
                + d[].api
                + " device memory information is unavailable: "
                + String(e)
            )
            return 2
        if info[1] == 0:
            set_error("MAX does not expose memory capacity on this device")
            return 2
        free[] = info[0]
        total[] = info[1]
        return 0
    except e:
        set_error(String(e))
        return 1


def _attribute(ctx: DeviceContext, attr: DeviceAttribute) -> Int64:
    try:
        return Int64(ctx.get_attribute(attr))
    except:
        return -1


def _properties(d: Pointer[Dev, MutUntrackedOrigin]) -> Properties:
    var ctx = d[].ctx
    var values = List[Int64]()
    # Capability is CUDA-specific; HIP uses the architecture string instead.
    values.append(
        _attribute(ctx, DeviceAttribute.COMPUTE_CAPABILITY_MAJOR) if d[].api
        == "cuda" else -1
    )
    values.append(
        _attribute(ctx, DeviceAttribute.COMPUTE_CAPABILITY_MINOR) if d[].api
        == "cuda" else -1
    )
    var total = Int64(-1)
    try:
        var info = ctx.get_memory_info()
        if info[1] > 0:
            total = Int64(info[1])
    except:
        # A MAX runtime that cannot report capacity leaves it unknown; the
        # properties call itself still succeeds, and mem_get_info() is the
        # entry point that surfaces the error.
        total = Int64(-1)
    values.append(total)
    var attrs: List[DeviceAttribute] = [
        DeviceAttribute.MULTIPROCESSOR_COUNT,
        DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR,
        DeviceAttribute.WARP_SIZE,
        DeviceAttribute.MAX_REGISTERS_PER_MULTIPROCESSOR,
        DeviceAttribute.MAX_THREADS_PER_BLOCK,
        DeviceAttribute.MAX_REGISTERS_PER_BLOCK,
        DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK,
        DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR,
        DeviceAttribute.CLOCK_RATE,
        DeviceAttribute.MAX_GRID_DIM_X,
    ]
    for attr in attrs:
        # MAX CPU returns successful zeroes for GPU attributes. They do not
        # describe CPU hardware, so expose them as unknown, not as capacities.
        values.append(-1 if d[].is_cpu else _attribute(ctx, attr))
    values.append(Int64(d[].is_cpu))
    var arch: String
    try:
        arch = ctx.arch_name()
    except:
        # Only AMD reports one; elsewhere the empty string means "no gfx
        # architecture", which Python exposes as None.
        arch = String()
    var text = ctx.name() + "\0" + d[].api + "\0" + arch + "\0"
    return Properties(values^, text^)


def h_device_props(
    device: Int32,
    dst: Pointer[Int64, MutUntrackedOrigin],
    n: Int32,
    text: Pointer[c_char, MutUntrackedOrigin],
    text_cap: Int32,
) abi("C") -> Int32:
    try:
        var d = dev(Int(device))
        if n != 16:
            raise Error("device properties ABI mismatch: expected 16 slots")
        if not d[].properties:
            d[].properties = _properties(d)
        ref props = d[].properties.value()
        if Int(text_cap) < props.text.byte_length():
            raise Error("device properties text buffer too small")
        for i in range(16):
            dst[unsafe_offset=i] = props.values[i]
        unsafe_memcpy(
            dest=text,
            src=props.text.as_c_string_slice().unsafe_ptr(),
            count=props.text.byte_length(),
        )
        return 0
    except e:
        set_error(String(e))
        return 1


@fieldwise_init
struct Pinned(Movable):
    var buf: Optional[HostBuffer[DType.uint8]]
    var base: Int
    var nbytes: Int
    var device: Int  # host memory is not necessarily portable across devices


def _pinned_lower_bound(base: Int) -> Int:
    ref blocks = be()[].pinned
    var lo = 0
    var hi = len(blocks)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if PinnedP(unsafe_from_address=blocks[mid])[].base < base:
            lo = mid + 1
        else:
            hi = mid
    return lo


def h_host_alloc(
    nbytes: Int, device: Int32, data: Pointer[Int, MutUntrackedOrigin]
) abi("C") -> Int:
    try:
        var index = Int(device)
        if index < 0 or index >= len(be()[].devices):
            index = len(be()[].devices) - 1
        var buf = Optional[HostBuffer[DType.uint8]]()
        var base = 0
        if nbytes != 0:
            var ctx = dev(index)[].ctx
            buf = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
            base = Int(buf.value().unsafe_ptr())
        # Zero bytes still need a handle to free, but have no address to pin.
        var box = unsafe_alloc[Pinned](1)
        box.unsafe_write(Pinned(buf^, base, nbytes, index))
        if nbytes != 0:
            be()[].pinned.insert(_pinned_lower_bound(base), Int(box))
        data[] = base
        return Int(box)
    except e:
        set_error(String(e))
        return 0


def h_host_free(handle: Int) abi("C"):
    if handle == 0:
        return
    var box = PinnedP(unsafe_from_address=handle)
    if box[].nbytes != 0:
        ref blocks = be()[].pinned
        var i = _pinned_lower_bound(box[].base)
        # A missing or mismatched handle must not remove another live allocation.
        if i >= len(blocks) or blocks[i] != handle:
            return
        _ = blocks.pop(i)
    # No async copies use this block yet. The async-transfer follow-up needs
    # deferred free keyed on outstanding async copies before direct DMA.
    var moved = box.unsafe_take_pointee()
    box.unsafe_free()
    _ = moved^


def h_is_pinned_ptr(ptr: Int) abi("C") -> Int32:
    ref blocks = be()[].pinned
    var i = _pinned_lower_bound(ptr)
    if i < len(blocks) and PinnedP(unsafe_from_address=blocks[i])[].base == ptr:
        return 1
    if i > 0:
        var box = PinnedP(unsafe_from_address=blocks[i - 1])
        if ptr - box[].base < box[].nbytes:
            return 1
    return external_call["tmb_cuda_is_pinned_ptr", Int32](ptr)


def h_copy_data(
    dst: Int, src: Int, nbytes: Int, device: Int32, stream: Int64
) abi("C"):
    try:
        copy_d2d(stream_ctx(Int(device), Int(stream)), dst, src, nbytes)
    except e:
        set_error(String(e))


def h_device_of_ptr(ptr: Int) abi("C") -> Int32:
    return -1


def h_record_stream(handle: Int, device: Int32, stream: Int64) abi("C"):
    """torch.Tensor.record_stream: remember that `stream` uses the buffer; the
    fence happens when the buffer is released (h_free), covering every use
    enqueued on that stream up to then, as CUDA's caching allocator does."""
    if handle == 0:
        return
    var box = BufP(unsafe_from_address=handle)
    if Int(device) != box[].device:
        set_error("record_stream: stream of another device")
        return
    if Int(stream) == box[].stream:
        return
    for i in range(len(box[].users)):
        if box[].users[i] == Int(stream):
            return
    box[].users.append(Int(stream))


def wrap_raw(
    ctx: DeviceContext, addr: Int, nbytes: Int
) -> DeviceBuffer[DType.uint8]:
    return DeviceBuffer[DType.uint8](
        ctx, U8P(unsafe_from_address=addr), nbytes, owning=False
    )


def copy_d2d(ctx: DeviceContext, dst: Int, src: Int, nbytes: Int) raises:
    """Stream-ordered device copy. The CPU device runs copies on a worker pool
    that is not ordered with kernel execution, so there it completes first."""
    if nbytes == 0:
        return
    var d = wrap_raw(ctx, dst, nbytes)
    var s = wrap_raw(ctx, src, nbytes)
    d.enqueue_copy_from(s)
    if ctx.api() == "cpu":
        ctx.synchronize()


def _peer_access(device: Int, peer: Int) raises -> Bool:
    var d = dev(device)
    if peer in d[].peers:
        return d[].peers[peer]
    var other = dev(peer)
    var enabled = False
    if (d[].api == "cuda" or d[].api == "hip") and d[].api == other[].api:
        try:
            if ",host," not in be()[].test_peer_copy and d[].ctx.can_access(
                other[].ctx
            ):
                if ",enable_error," in be()[].test_peer_copy:
                    raise Error("injected peer enable failure")
                d[].ctx.enable_peer_access(other[].ctx)
                enabled = True
        except e:
            if be()[].test_peer_copy != "":
                print("peer enable failed", String(e))
            enabled = False
    if be()[].test_peer_copy != "":
        print("peer probe", device, peer, enabled)
    d[].peers[peer] = enabled
    return enabled


def _test_peer_gate(p: Pointer[NoneType, MutAnyOrigin]):
    # The test releases the pipe even on failure.
    var byte = UInt8(0)
    _ = external_call["read", Int](Int(p), Pointer(to=byte), 1)


def _test_peer_submit(
    dst_ctx: DeviceContext,
    src_ctx: DeviceContext,
    d: DeviceBuffer[DType.uint8],
    s: DeviceBuffer[DType.uint8],
) raises:
    var mode = be()[].test_peer_copy
    if ",gate," in mode:
        dst_ctx.stream().enqueue_host_func(
            _test_peer_gate,
            Pointer[NoneType, MutAnyOrigin](
                unsafe_from_address=be()[].test_peer_gate
            ),
        )
    elif ",submit_error," in mode or ",drain_error," in mode:
        dst_ctx.enqueue_wait_for(src_ctx)
        dst_ctx.enqueue_copy_no_cross_stream_sync(d, s)
        print("P2P_SUBMITTED", Int(d.unsafe_ptr()), Int(s.unsafe_ptr()))
        raise Error("injected failure after DMA, before reverse event")


def copy_peer(
    dst_device: Int, dst: Int, src_device: Int, src: Int, nbytes: Int
) raises -> Bool:
    """False requests host staging; caller fences storage and drains on error.
    """
    if nbytes == 0:
        return True
    if not _peer_access(dst_device, src_device):
        if be()[].test_peer_copy != "":
            print("peer copy host", dst_device, src_device)
        return False
    var dst_ctx = ctx_for(dst_device)
    var src_ctx = ctx_for(src_device)
    var d = wrap_raw(dst_ctx, dst, nbytes)
    var s = wrap_raw(src_ctx, src, nbytes)
    if be()[].test_peer_copy != "":
        _test_peer_submit(dst_ctx, src_ctx, d, s)
    d.enqueue_copy_from(s)
    if be()[].test_peer_copy != "":
        print("peer copy direct", dst_device, src_device)
    _ = s^
    _ = d^
    return True


def drain_copy(dst_device: Int, src_device: Int) -> Bool:
    """Error cleanup only; a failed drain quarantines both devices' storage."""
    var drained = True
    for i in range(2):
        var device = dst_device if i == 0 else src_device
        try:
            if ",drain_error," in be()[].test_peer_copy and i == 0:
                raise Error("injected destination drain failure")
            ctx_for(device).synchronize()
            if be()[].test_peer_copy != "":
                print("P2P_DRAINED", device)
        except e:
            _warn("transfer drain failed; retaining storage", e)
            drained = False
    if not drained:
        be()[].devices[dst_device].quarantine = True
        be()[].devices[src_device].quarantine = True
    return drained


def copy_to_host(
    ctx: DeviceContext, dev_ptr: Int, host_ptr: Int, nbytes: Int
) raises:
    """Blocking D2H."""
    if nbytes == 0:
        return
    var s = wrap_raw(ctx, dev_ptr, nbytes)
    s.enqueue_copy_to(U8P(unsafe_from_address=host_ptr))
    ctx.synchronize()


@fieldwise_init
struct Staging(Movable):
    var buf: HostBuffer[DType.uint8]
    var done: Atomic[DType.int32]  # set from the driver's callback thread


def _staging_done(p: Pointer[NoneType, MutAnyOrigin]):
    p.unsafe_bitcast[Staging]()[].done.store(1)


def _drain_staging(d: Pointer[Dev, MutUntrackedOrigin]):
    var keep = List[Int]()
    for i in range(len(d[].staging)):
        var box = StagingP(unsafe_from_address=d[].staging[i])
        if box[].done.load() != 0:
            var moved = box.unsafe_take_pointee()
            box.unsafe_free()
            _ = moved^
        else:
            keep.append(d[].staging[i])
    d[].staging = keep^


def copy_from_host(
    device: Int, ctx: DeviceContext, dev_ptr: Int, host_ptr: Int, nbytes: Int
) raises:
    """H2D from pageable host memory: on GPUs, stage through a MAX pinned
    buffer (a synchronous memcpy, then an asynchronous DMA) and release the
    staging buffer once a host callback reports the copy done. The CPU
    device copies synchronously."""
    if nbytes == 0:
        return
    var dst = wrap_raw(ctx, dev_ptr, nbytes)
    if ctx.api() == "cpu" or ctx.api() == "metal":
        # Metal: unified memory, and MAX's Metal streams have no host
        # callbacks (enqueue_host_func), which the staged path below needs
        dst.enqueue_copy_from(U8P(unsafe_from_address=host_ptr))
        ctx.synchronize()
        return
    var d = dev(device)
    _drain_staging(d)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    unsafe_memcpy(
        dest=host.unsafe_ptr(),
        src=U8P(unsafe_from_address=host_ptr),
        count=nbytes,
    )
    var box = unsafe_alloc[Staging](1)
    box.unsafe_write(Staging(host^, Atomic[DType.int32](0)))
    # Own pinned staging before submission, including partial-submit errors.
    d[].staging.append(Int(box))
    try:
        dst.enqueue_copy_from(box[].buf)
        ctx.stream().enqueue_host_func(
            _staging_done,
            box.unsafe_bitcast[NoneType]().unsafe_origin_cast[MutAnyOrigin](),
        )
    except e:
        try:
            ctx.synchronize()
            box[].done.store(1)
        except drain_error:
            d[].quarantine = True
            _warn("H2D drain failed; retaining staging", drain_error)
        raise e


def read_bytes_sync(
    ctx: DeviceContext, dev_ptr: Int, dst_addr: Int, nbytes: Int
) raises:
    """Blocking readback of a few bytes (the .item() primitive)."""
    var s = wrap_raw(ctx, dev_ptr, nbytes)
    s.enqueue_copy_to(U8P(unsafe_from_address=dst_addr))
    ctx.synchronize()


def memset_bytes(
    ctx: DeviceContext, dev_ptr: Int, value: UInt8, nbytes: Int
) raises:
    if nbytes == 0:
        return
    var d = wrap_raw(ctx, dev_ptr, nbytes)
    ctx.enqueue_memset(d, value)


def memset_typed[
    dt: DType
](ctx: DeviceContext, dev_ptr: Int, value: Scalar[dt], count: Int) raises:
    if count == 0:
        return
    var d = DeviceBuffer[dt](
        ctx,
        Pointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=dev_ptr),
        count,
        owning=False,
    )
    ctx.enqueue_memset(d, value)


# --- devices and streams -----------------------------------------------------


def h_device_count() abi("C") -> Int32:
    return Int32(len(be()[].devices))


def h_synchronize_device(device: Int32) abi("C"):
    try:
        var d = dev(Int(device))
        for i in range(len(d[].views)):
            d[].views[i].synchronize()
        _drain_staging(d)
    except e:
        set_error(String(e))


def _add_stream(
    d: Pointer[Dev, MutUntrackedOrigin], priority: Int
) raises -> Int:
    var s = d[].ctx.create_stream(priority=priority)
    var idx = d[].ctx.num_streams() - 1
    var view = d[].ctx.select_stream(idx)
    var r = 0
    if be()[].vendor:
        try:
            r = raw_stream(be()[].vendor.value(), view)
        except e:
            _warn("no raw stream handle", e)
    d[].streams.append(s^)
    d[].views.append(view^)
    d[].raw.append(r)
    return idx


def h_new_stream(device: Int32, priority: Int32) abi("C") -> Int64:
    try:
        var d = dev(Int(device))
        if d[].is_cpu:
            return 0
        return Int64(_add_stream(d, Int(priority)))
    except e:
        set_error(String(e))
        return 0


def h_stream_from_pool(device: Int32, high_priority: Int32) abi("C") -> Int64:
    try:
        var d = dev(Int(device))
        if d[].is_cpu:
            return 0
        if len(d[].pool) < POOL_STREAMS:
            d[].pool.append(_add_stream(d, 0))
            return Int64(d[].pool[len(d[].pool) - 1])
        var s = d[].pool[d[].pool_next]
        d[].pool_next = (d[].pool_next + 1) % POOL_STREAMS
        return Int64(s)
    except e:
        set_error(String(e))
        return 0


def h_synchronize_stream(device: Int32, stream: Int64) abi("C"):
    try:
        stream_ctx(Int(device), Int(stream)).synchronize()
    except e:
        set_error(String(e))


def h_query_stream(device: Int32, stream: Int64) abi("C") -> Int32:
    try:
        var d = dev(Int(device))
        var r = d[].raw[Int(stream)] if Int(stream) < len(d[].raw) else 0
        if r != 0 and be()[].vendor:
            return 1 if be()[].vendor.value().stream_query(r) else 0
        d[].view(Int(stream)).synchronize()
        return 1
    except e:
        set_error(String(e))
        return 1


def h_stream_native_handle(device: Int32, stream: Int64) abi("C") -> Int:
    try:
        var d = dev(Int(device))
        if Int(stream) < len(d[].raw):
            return d[].raw[Int(stream)]
    except e:
        set_error(String(e))
    return 0


# --- events ---------------------------------------------------------------------


@fieldwise_init
struct Ev(Movable):
    var device: Int
    var timing: Bool
    var recorded: Bool
    var max_ev: DeviceEvent
    var raw: Int
    var host_ns: Int


def h_event_create(device: Int32, enable_timing: Int32) abi("C") -> Int:
    try:
        var d = dev(Int(device))
        var raw = 0
        if be()[].vendor and d[].raw[0] != 0:
            raw = (
                be()[].vendor.value().event_create(d[].ctx, enable_timing != 0)
            )
        var box = unsafe_alloc[Ev](1)
        box.unsafe_write(
            Ev(
                Int(device),
                enable_timing != 0,
                False,
                d[].ctx.create_event(),
                raw,
                0,
            )
        )
        return Int(box)
    except e:
        set_error(String(e))
        return 0


def h_event_destroy(ev: Int, device: Int32) abi("C"):
    if ev == 0:
        return
    var box = EvP(unsafe_from_address=ev)
    if box[].raw != 0 and be()[].vendor:
        try:
            be()[].vendor.value().event_destroy(box[].raw)
        except e:
            _warn("event destroy", e)
    var moved = box.unsafe_take_pointee()
    box.unsafe_free()
    _ = moved^


def h_event_record(ev: Int, device: Int32, stream: Int64) abi("C"):
    try:
        var box = EvP(unsafe_from_address=ev)
        var d = dev(Int(device))
        var ctx = d[].view(Int(stream))
        if box[].raw != 0:
            be()[].vendor.value().event_record(box[].raw, d[].raw[Int(stream)])
        ctx.stream().record_event(
            box[].max_ev
        )  # after the vendor event: waiting on it covers both
        box[].host_ns = perf_counter_ns()
        box[].recorded = True
    except e:
        set_error(String(e))


def h_event_block(ev: Int, device: Int32, stream: Int64) abi("C"):
    try:
        var box = EvP(unsafe_from_address=ev)
        if not box[].recorded:
            return
        stream_ctx(Int(device), Int(stream)).stream().enqueue_wait_for(
            box[].max_ev
        )
    except e:
        set_error(String(e))


def h_event_query(ev: Int) abi("C") -> Int32:
    try:
        var box = EvP(unsafe_from_address=ev)
        if not box[].recorded:
            return 1
        if box[].raw != 0:
            return 1 if be()[].vendor.value().event_query(box[].raw) else 0
        box[].max_ev.synchronize()
        return 1
    except e:
        set_error(String(e))
        return 1


def h_event_synchronize(ev: Int) abi("C"):
    try:
        var box = EvP(unsafe_from_address=ev)
        if box[].recorded:
            box[].max_ev.synchronize()
            if box[].raw != 0:
                be()[].vendor.value().event_synchronize(box[].raw)
    except e:
        set_error(String(e))


def h_event_elapsed_ms(start: Int, end: Int) abi("C") -> Float64:
    try:
        var a = EvP(unsafe_from_address=start)
        var b = EvP(unsafe_from_address=end)
        if not a[].timing or not b[].timing:
            raise Error(
                "elapsed_time needs events created with enable_timing=True"
            )
        if a[].raw != 0 and b[].raw != 0:
            return be()[].vendor.value().event_elapsed_ms(a[].raw, b[].raw)
        if not dev(a[].device)[].is_cpu:
            raise Error(
                "elapsed_time: device timing needs the vendor driver (not"
                " available on this device)"
            )
        a[].max_ev.synchronize()
        b[].max_ev.synchronize()
        return Float64(b[].host_ns - a[].host_ns) / 1.0e6
    except e:
        set_error(String(e))
        return 0.0


# --- the table --------------------------------------------------------------


def set_error(msg: String):
    var tmp = String(msg)
    external_call["tmb_set_error", NoneType](
        tmp.as_c_string_slice().unsafe_ptr()
    )


def hooks_table() -> Pointer[Int, MutUntrackedOrigin]:
    """TmbBackendHooks (tmb.h): a u32 size padded to 8 bytes, then 31 pointers.
    """
    comptime N = 32
    var t = unsafe_alloc[Int](N)
    for i in range(N):
        t[unsafe_offset=i] = 0
    t[unsafe_offset=0] = N * 8
    var f_alloc: def(
        Int, Int32, Int64, Pointer[Int, MutUntrackedOrigin]
    ) thin abi("C") -> Int = h_alloc
    var f_free: def(Int) thin abi("C") -> None = h_free
    var f_copy: def(Int, Int, Int, Int32, Int64) thin abi(
        "C"
    ) -> None = h_copy_data
    var f_devof: def(Int) thin abi("C") -> Int32 = h_device_of_ptr
    var f_rec: def(Int, Int32, Int64) thin abi("C") -> None = h_record_stream
    var f_count: def() thin abi("C") -> Int32 = h_device_count
    var f_syncdev: def(Int32) thin abi("C") -> None = h_synchronize_device
    var f_newstream: def(Int32, Int32) thin abi("C") -> Int64 = h_new_stream
    var f_pool: def(Int32, Int32) thin abi("C") -> Int64 = h_stream_from_pool
    var f_syncstream: def(Int32, Int64) thin abi(
        "C"
    ) -> None = h_synchronize_stream
    var f_qstream: def(Int32, Int64) thin abi("C") -> Int32 = h_query_stream
    var f_native: def(Int32, Int64) thin abi(
        "C"
    ) -> Int = h_stream_native_handle
    var f_evc: def(Int32, Int32) thin abi("C") -> Int = h_event_create
    var f_evd: def(Int, Int32) thin abi("C") -> None = h_event_destroy
    var f_evr: def(Int, Int32, Int64) thin abi("C") -> None = h_event_record
    var f_evb: def(Int, Int32, Int64) thin abi("C") -> None = h_event_block
    var f_evq: def(Int) thin abi("C") -> Int32 = h_event_query
    var f_evs: def(Int) thin abi("C") -> None = h_event_synchronize
    var f_eve: def(Int, Int) thin abi("C") -> Float64 = h_event_elapsed_ms
    var f_stats: def(Int32, Pointer[Int64, MutUntrackedOrigin], Int32) thin abi(
        "C"
    ) -> None = h_mem_stats
    var f_peak: def(Int32) thin abi("C") -> None = h_mem_reset_peak
    var f_accum: def(Int32) thin abi("C") -> None = h_mem_reset_accumulated
    var f_empty: def() thin abi("C") -> None = h_empty_cache
    var f_info: def(
        Int32,
        Pointer[c_size_t, MutUntrackedOrigin],
        Pointer[c_size_t, MutUntrackedOrigin],
    ) thin abi("C") -> Int32 = h_mem_get_info
    var f_props: def(
        Int32,
        Pointer[Int64, MutUntrackedOrigin],
        Int32,
        Pointer[c_char, MutUntrackedOrigin],
        Int32,
    ) thin abi("C") -> Int32 = h_device_props
    var f_host_alloc: def(
        Int, Int32, Pointer[Int, MutUntrackedOrigin]
    ) thin abi("C") -> Int = h_host_alloc
    var f_host_free: def(Int) thin abi("C") -> None = h_host_free
    var f_pinned: def(Int) thin abi("C") -> Int32 = h_is_pinned_ptr
    t[unsafe_offset=1] = Pointer(to=f_alloc).unsafe_bitcast[Int]()[]
    t[unsafe_offset=2] = Pointer(to=f_free).unsafe_bitcast[Int]()[]
    t[unsafe_offset=3] = Pointer(to=f_copy).unsafe_bitcast[Int]()[]
    t[unsafe_offset=4] = Pointer(to=f_devof).unsafe_bitcast[Int]()[]
    t[unsafe_offset=5] = Pointer(to=f_rec).unsafe_bitcast[Int]()[]
    t[unsafe_offset=6] = Pointer(to=f_count).unsafe_bitcast[Int]()[]
    t[unsafe_offset=7] = Pointer(to=f_syncdev).unsafe_bitcast[Int]()[]
    t[unsafe_offset=8] = Pointer(to=f_newstream).unsafe_bitcast[Int]()[]
    t[unsafe_offset=9] = Pointer(to=f_pool).unsafe_bitcast[Int]()[]
    t[unsafe_offset=10] = Pointer(to=f_syncstream).unsafe_bitcast[Int]()[]
    t[unsafe_offset=11] = Pointer(to=f_qstream).unsafe_bitcast[Int]()[]
    t[unsafe_offset=12] = Pointer(to=f_native).unsafe_bitcast[Int]()[]
    t[unsafe_offset=13] = Pointer(to=f_evc).unsafe_bitcast[Int]()[]
    t[unsafe_offset=14] = Pointer(to=f_evd).unsafe_bitcast[Int]()[]
    t[unsafe_offset=15] = Pointer(to=f_evr).unsafe_bitcast[Int]()[]
    t[unsafe_offset=16] = Pointer(to=f_evb).unsafe_bitcast[Int]()[]
    t[unsafe_offset=17] = Pointer(to=f_evq).unsafe_bitcast[Int]()[]
    t[unsafe_offset=18] = Pointer(to=f_evs).unsafe_bitcast[Int]()[]
    t[unsafe_offset=19] = Pointer(to=f_eve).unsafe_bitcast[Int]()[]
    # 20..22: prof_mark / prof_range_push / prof_range_pop stay NULL for now
    t[unsafe_offset=23] = Pointer(to=f_stats).unsafe_bitcast[Int]()[]
    t[unsafe_offset=24] = Pointer(to=f_peak).unsafe_bitcast[Int]()[]
    t[unsafe_offset=25] = Pointer(to=f_accum).unsafe_bitcast[Int]()[]
    t[unsafe_offset=26] = Pointer(to=f_empty).unsafe_bitcast[Int]()[]
    t[unsafe_offset=27] = Pointer(to=f_info).unsafe_bitcast[Int]()[]
    t[unsafe_offset=28] = Pointer(to=f_props).unsafe_bitcast[Int]()[]
    t[unsafe_offset=29] = Pointer(to=f_host_alloc).unsafe_bitcast[Int]()[]
    t[unsafe_offset=30] = Pointer(to=f_host_free).unsafe_bitcast[Int]()[]
    t[unsafe_offset=31] = Pointer(to=f_pinned).unsafe_bitcast[Int]()[]
    return t


# --- what ops need ------------------------------------------------------------


def current_device() -> Int:
    return Int(external_call["tmb_current_device", Int32]())


def current_stream(device: Int) -> Int:
    return Int(external_call["tmb_current_stream", Int64](Int32(device)))


def ctx_for(device: Int) raises -> DeviceContext:
    """The context view an op launches on: the device's current stream."""
    return stream_ctx(device, current_stream(device))


def ctx_ptr(ctx: DeviceContext) -> Int:
    """The C++ DeviceContext handle kernels rebuild a `DeviceContext` from
    (TensorSpec.ctx_ptr / the trailing ctx slot): the same pointer
    `Device._device_context_ptr()` handed the old Python path."""
    return Int(ctx._handle.value())


def record_stream(handle: Int, device: Int, stream: Int) raises:
    """Op-side twin of h_record_stream."""
    var box = BufP(unsafe_from_address=handle)
    if device != box[].device:
        raise Error("record_stream: stream of another device")
    if stream == box[].stream:
        return
    for i in range(len(box[].users)):
        if box[].users[i] == stream:
            return
    box[].users.append(stream)
