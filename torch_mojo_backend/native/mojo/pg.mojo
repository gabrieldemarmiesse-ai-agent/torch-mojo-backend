"""Process-group core: NCCL / RCCL / mojoccl collectives on a per-device comm
stream.

torch's `MojoProcessGroup` (distributed/process_group.py) is a thin adapter:
it resolves data pointers, stages non-contiguous tensors with torch ops and
wraps a device-typed torch Future whose completion events are recorded on the
comm stream, so consumers on any stream wait exactly as with ProcessGroupNCCL.
Communicators, the comm stream and the library calls live here. The three
libraries share the NCCL C ABI; mojoccl passes the 128-byte unique id the
way the SysV ABI does for a by-value struct: 16 words after the register
arguments (see mojoccl.mojo's ncclCommInitRank).
"""
from std.collections import Dict
from std.ffi import OwnedDLHandle, c_char
from std.memory.alloc import unsafe_alloc

from device import _add_stream, dev, stream_ctx, current_stream, set_error

comptime UID_BYTES = 128


@fieldwise_init
struct Comm(Copyable, Movable):
    var handle: Int64
    var device: Int
    var stream: Int  # comm stream id on the device
    var raw: Int  # its vendor stream handle (what the library enqueues on)


struct PG(Movable):
    var lib: OwnedDLHandle
    var rank: Int
    var world: Int
    var comms: Dict[Int, Comm]

    def __init__(out self, path: String, rank: Int, world: Int) raises:
        self.lib = OwnedDLHandle(path)
        self.rank = rank
        self.world = world
        self.comms = Dict[Int, Comm]()

    def error_string(self, rc: Int32) -> String:
        try:
            var p = self.lib.get_function[Pointer[UInt8, MutUntrackedOrigin]](
                "ncclGetErrorString"
            )(rc)
            return String(unsafe_from_utf8_ptr=p)
        except e:
            return "error " + String(rc)

    def check(self, rc: Int32, what: StaticString) raises:
        if rc != 0:
            raise Error(what, ": ", self.error_string(rc))

    def comm(self, device: Int) raises -> Comm:
        var c = self.comms.find(device)
        if not c:
            raise Error(
                "no communicator on mojo:", device, " (init_device first)"
            )
        return c.value().copy()

    def sync_in(self, c: Comm) raises:
        """The comm stream waits for everything the caller's current stream has
        enqueued so far (inputs are ready, outputs are no longer being read)."""
        var cur = current_stream(c.device)
        if cur != c.stream:
            stream_ctx(c.device, c.stream).enqueue_wait_for(
                stream_ctx(c.device, cur)
            )


comptime PGP = Pointer[PG, MutUntrackedOrigin]


@always_inline
def _pg(p: Int) -> PGP:
    return PGP(unsafe_from_address=p)


def tmb_pg_create(
    path: Pointer[c_char, MutUntrackedOrigin], rank: Int32, world: Int32
) abi("C") -> Int:
    try:
        var box = unsafe_alloc[PG](1)
        box.unsafe_write(
            PG(
                String(unsafe_from_utf8_ptr=path.unsafe_bitcast[UInt8]()),
                Int(rank),
                Int(world),
            )
        )
        return Int(box)
    except e:
        set_error(String(e))
        return 0


def tmb_pg_destroy(p: Int) abi("C"):
    if p == 0:
        return
    var box = _pg(p)
    try:
        for item in box[].comms.items():
            _ = box[].lib.get_function[Int32]("ncclCommDestroy")(
                item.value.handle
            )
    except e:
        set_error(String(e))
    var moved = box.unsafe_take_pointee()
    box.unsafe_free()
    _ = moved^


def tmb_pg_version(p: Int) abi("C") -> Int32:
    try:
        var v: Int32 = 0
        _pg(p)[].check(
            _pg(p)[].lib.get_function[Int32]("ncclGetVersion")(Pointer(to=v)),
            "ncclGetVersion",
        )
        return v
    except e:
        set_error(String(e))
        return -1


def tmb_pg_unique_id(
    p: Int, buf: Pointer[UInt8, MutUntrackedOrigin]
) abi("C") -> Int32:
    try:
        _pg(p)[].check(
            _pg(p)[].lib.get_function[Int32]("ncclGetUniqueId")(buf),
            "ncclGetUniqueId",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_init_device(
    p: Int, device: Int32, uid: Pointer[UInt8, MutUntrackedOrigin]
) abi("C") -> Int32:
    """Create the communicator of this rank on mojo:`device` (one per device)
    and its dedicated comm stream."""
    try:
        var pg = _pg(p)
        var d = dev(Int(device))
        if d[].is_cpu:
            raise Error("the mojo process group needs an accelerator device")
        var sid = _add_stream(d, 0)
        var raw = d[].raw[sid]
        if raw == 0:
            raise Error(
                "no vendor stream handle for the comm stream (NCCL needs one)"
            )
        var w = uid.unsafe_bitcast[UInt64]()
        var handle: Int64 = 0
        var rc: Int32
        with d[].ctx.push_context():  # ncclCommInitRank binds the communicator to the current device
            rc = pg[].lib.get_function[Int32]("ncclCommInitRank")(
                Pointer(to=handle),
                Int32(pg[].world),
                Int32(pg[].rank),
                Int64(0),
                Int64(0),
                Int64(0),
                w[unsafe_offset=0],
                w[unsafe_offset=1],
                w[unsafe_offset=2],
                w[unsafe_offset=3],
                w[unsafe_offset=4],
                w[unsafe_offset=5],
                w[unsafe_offset=6],
                w[unsafe_offset=7],
                w[unsafe_offset=8],
                w[unsafe_offset=9],
                w[unsafe_offset=10],
                w[unsafe_offset=11],
                w[unsafe_offset=12],
                w[unsafe_offset=13],
                w[unsafe_offset=14],
                w[unsafe_offset=15],
            )
        pg[].check(rc, "ncclCommInitRank")
        pg[].comms[Int(device)] = Comm(handle, Int(device), sid, raw)
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_comm_stream(p: Int, device: Int32) abi("C") -> Int64:
    try:
        return Int64(_pg(p)[].comm(Int(device)).stream)
    except e:
        set_error(String(e))
        return -1


def tmb_pg_allreduce(
    p: Int, device: Int32, ptr: Int, count: Int, dtype: Int32, op: Int32
) abi("C") -> Int32:
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        pg[].sync_in(c)
        pg[].check(
            pg[].lib.get_function[Int32]("ncclAllReduce")(
                ptr, ptr, count, dtype, op, c.handle, c.raw
            ),
            "ncclAllReduce",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_broadcast(
    p: Int, device: Int32, ptr: Int, count: Int, dtype: Int32, root: Int32
) abi("C") -> Int32:
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        pg[].sync_in(c)
        pg[].check(
            pg[].lib.get_function[Int32]("ncclBroadcast")(
                ptr, ptr, count, dtype, root, c.handle, c.raw
            ),
            "ncclBroadcast",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_reduce(
    p: Int,
    device: Int32,
    ptr: Int,
    count: Int,
    dtype: Int32,
    op: Int32,
    root: Int32,
) abi("C") -> Int32:
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        pg[].sync_in(c)
        pg[].check(
            pg[].lib.get_function[Int32]("ncclReduce")(
                ptr, ptr, count, dtype, op, root, c.handle, c.raw
            ),
            "ncclReduce",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_allgather(
    p: Int, device: Int32, send: Int, recv: Int, count: Int, dtype: Int32
) abi("C") -> Int32:
    """recv[world * count] <- every rank's send[count]."""
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        pg[].sync_in(c)
        pg[].check(
            pg[].lib.get_function[Int32]("ncclAllGather")(
                send, recv, count, dtype, c.handle, c.raw
            ),
            "ncclAllGather",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_reduce_scatter(
    p: Int,
    device: Int32,
    send: Int,
    recv: Int,
    count: Int,
    dtype: Int32,
    op: Int32,
) abi("C") -> Int32:
    """recv[count] <- reduce of send[world * count] chunk `rank`."""
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        pg[].sync_in(c)
        pg[].check(
            pg[].lib.get_function[Int32]("ncclReduceScatter")(
                send, recv, count, dtype, op, c.handle, c.raw
            ),
            "ncclReduceScatter",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_send(
    p: Int, device: Int32, ptr: Int, count: Int, dtype: Int32, peer: Int32
) abi("C") -> Int32:
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        pg[].sync_in(c)
        pg[].check(
            pg[].lib.get_function[Int32]("ncclSend")(
                ptr, count, dtype, peer, c.handle, c.raw
            ),
            "ncclSend",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_recv(
    p: Int, device: Int32, ptr: Int, count: Int, dtype: Int32, peer: Int32
) abi("C") -> Int32:
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        pg[].sync_in(c)
        pg[].check(
            pg[].lib.get_function[Int32]("ncclRecv")(
                ptr, count, dtype, peer, c.handle, c.raw
            ),
            "ncclRecv",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_group_start(p: Int) abi("C") -> Int32:
    try:
        _pg(p)[].check(
            _pg(p)[].lib.get_function[Int32]("ncclGroupStart")(),
            "ncclGroupStart",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_group_end(p: Int) abi("C") -> Int32:
    try:
        _pg(p)[].check(
            _pg(p)[].lib.get_function[Int32]("ncclGroupEnd")(), "ncclGroupEnd"
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_async_error(p: Int, device: Int32) abi("C") -> Int32:
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        var err: Int32 = 0
        pg[].check(
            pg[].lib.get_function[Int32]("ncclCommGetAsyncError")(
                c.handle, Pointer(to=err)
            ),
            "ncclCommGetAsyncError",
        )
        return err
    except e:
        set_error(String(e))
        return -1


def tmb_pg_abort(p: Int, device: Int32) abi("C") -> Int32:
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        pg[].check(
            pg[].lib.get_function[Int32]("ncclCommAbort")(c.handle),
            "ncclCommAbort",
        )
        return 0
    except e:
        set_error(String(e))
        return 1


def tmb_pg_synchronize_comm(p: Int, device: Int32) abi("C") -> Int32:
    """Host-wait for every collective issued so far on the device's comm stream.
    """
    try:
        var pg = _pg(p)
        var c = pg[].comm(Int(device))
        stream_ctx(c.device, c.stream).synchronize()
        return 0
    except e:
        set_error(String(e))
        return 1


comptime PG_VTABLE_SLOTS = 17


def pg_vtable() -> Pointer[Int, MutUntrackedOrigin]:
    """Function addresses for the Python adapter, in this fixed order:
    0 create, 1 destroy, 2 version, 3 unique_id, 4 init_device, 5 comm_stream,
    6 allreduce, 7 broadcast, 8 reduce, 9 allgather, 10 reduce_scatter,
    11 send, 12 recv, 13 group_start, 14 group_end, 15 async_error, 16 abort,
    and synchronize_comm last."""
    var t = unsafe_alloc[Int](PG_VTABLE_SLOTS + 1)
    var f0: def(Pointer[c_char, MutUntrackedOrigin], Int32, Int32) thin abi(
        "C"
    ) -> Int = tmb_pg_create
    var f1: def(Int) thin abi("C") -> None = tmb_pg_destroy
    var f2: def(Int) thin abi("C") -> Int32 = tmb_pg_version
    var f3: def(Int, Pointer[UInt8, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> Int32 = tmb_pg_unique_id
    var f4: def(Int, Int32, Pointer[UInt8, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> Int32 = tmb_pg_init_device
    var f5: def(Int, Int32) thin abi("C") -> Int64 = tmb_pg_comm_stream
    var f6: def(Int, Int32, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_allreduce
    var f7: def(Int, Int32, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_broadcast
    var f8: def(Int, Int32, Int, Int, Int32, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_reduce
    var f9: def(Int, Int32, Int, Int, Int, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_allgather
    var f10: def(Int, Int32, Int, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_reduce_scatter
    var f11: def(Int, Int32, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_send
    var f12: def(Int, Int32, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_recv
    var f13: def(Int) thin abi("C") -> Int32 = tmb_pg_group_start
    var f14: def(Int) thin abi("C") -> Int32 = tmb_pg_group_end
    var f15: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_async_error
    var f16: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_abort
    var f17: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_synchronize_comm
    t[unsafe_offset=0] = Pointer(to=f0).unsafe_bitcast[Int]()[]
    t[unsafe_offset=1] = Pointer(to=f1).unsafe_bitcast[Int]()[]
    t[unsafe_offset=2] = Pointer(to=f2).unsafe_bitcast[Int]()[]
    t[unsafe_offset=3] = Pointer(to=f3).unsafe_bitcast[Int]()[]
    t[unsafe_offset=4] = Pointer(to=f4).unsafe_bitcast[Int]()[]
    t[unsafe_offset=5] = Pointer(to=f5).unsafe_bitcast[Int]()[]
    t[unsafe_offset=6] = Pointer(to=f6).unsafe_bitcast[Int]()[]
    t[unsafe_offset=7] = Pointer(to=f7).unsafe_bitcast[Int]()[]
    t[unsafe_offset=8] = Pointer(to=f8).unsafe_bitcast[Int]()[]
    t[unsafe_offset=9] = Pointer(to=f9).unsafe_bitcast[Int]()[]
    t[unsafe_offset=10] = Pointer(to=f10).unsafe_bitcast[Int]()[]
    t[unsafe_offset=11] = Pointer(to=f11).unsafe_bitcast[Int]()[]
    t[unsafe_offset=12] = Pointer(to=f12).unsafe_bitcast[Int]()[]
    t[unsafe_offset=13] = Pointer(to=f13).unsafe_bitcast[Int]()[]
    t[unsafe_offset=14] = Pointer(to=f14).unsafe_bitcast[Int]()[]
    t[unsafe_offset=15] = Pointer(to=f15).unsafe_bitcast[Int]()[]
    t[unsafe_offset=16] = Pointer(to=f16).unsafe_bitcast[Int]()[]
    t[unsafe_offset=17] = Pointer(to=f17).unsafe_bitcast[Int]()[]
    return t
