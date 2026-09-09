# /dev/shm rendezvous bootstrap for mojoccl's ncclCommInitRank.
#
# Real NCCL exchanges per-rank IPC handles over its own bootstrap socket,
# whose address is encoded in the 128-byte ncclUniqueId. This library has no
# socket bootstrap yet, so it stands in with a directory: the rank that calls
# ncclGetUniqueId creates a fresh dir under /dev/shm and encodes its path in
# the id (that id then travels through the caller's OWN rendezvous -- for
# this repo, the c10d store in distributed/process_group.py, unchanged).
# Every rank's ncclCommInitRank decodes the path back out, atomically writes
# its own `rank<r>.handle` file there, and polls for the rest.
#
# This is intra-node only. The design leaves room for a multi-node TCP
# bootstrap later (encode "host:port" instead of a path, listen instead of
# mkdir) without changing ncclCommInitRank's call shape.

from std.ffi import OwnedDLHandle, CStringSlice
from std.memory.alloc import unsafe_alloc
from std.os import remove, rmdir
from std.pathlib import Path
from std.tempfile import mkdtemp
from std.time import sleep, perf_counter_ns
from std.collections.string import chr

comptime UID_BYTES = 128
comptime HANDLE_BYTES = 64


def create_rendezvous_dir() raises -> String:
    """Called once, by the rank that generates the unique id."""
    return mkdtemp(prefix="mojoccl-", dir="/dev/shm")


def encode_dir(dir: String, out_bytes: Pointer[UInt8, MutAnyOrigin]) raises:
    """Writes `dir`'s UTF-8 bytes, NUL-padded, into the 128-byte unique id."""
    for i in range(UID_BYTES):
        out_bytes[unsafe_offset=i] = 0
    var db = dir.as_bytes()
    if len(db) > UID_BYTES - 2:
        raise Error(
            "mojoccl: rendezvous directory path too long for the 128-byte"
            " unique id"
        )
    for i in range(len(db)):
        out_bytes[unsafe_offset=i] = db[i]


def decode_dir(in_bytes: Pointer[UInt8, MutAnyOrigin]) raises -> String:
    """The inverse of `encode_dir`, reading up to the first NUL byte."""
    var n = 0
    while n < UID_BYTES and Int(in_bytes[unsafe_offset=n]) != 0:
        n += 1
    if n == 0:
        raise Error("mojoccl: empty rendezvous directory in the unique id")
    var s = String(capacity=n)
    for i in range(n):
        s += chr(Int(in_bytes[unsafe_offset=i]))
    return s^


def _to_cstr(s: String) -> Pointer[Int8, ImmutAnyOrigin]:
    """A heap copy of `s`'s bytes plus a trailing NUL.

    `CStringSlice`'s constructors only VALIDATE an existing terminator
    (`std.ffi.cstring._validate_bytes`); neither copies. A plain `String`'s
    backing storage carries no such terminator (measured: both the
    `StringSlice` and `Span[Byte]` constructors raise "CStringSlice is not
    nul-terminated" on one), so this builds the terminated buffer by hand.
    """
    var src = s.as_bytes()
    var n = len(src)
    var buf = unsafe_alloc[UInt8](n + 1)
    for i in range(n):
        buf[unsafe_offset=i] = src[i]
    buf[unsafe_offset=n] = 0
    return Pointer[Int8, ImmutAnyOrigin](unsafe_from_address=Int(buf))


def _rename(old: String, new: String) raises:
    """POSIX rename(2) via libc -- the atomic-publish half of the handle
    write. Not in std.os yet, so called directly (the same OwnedDLHandle +
    get_function shape as every other FFI call in this library)."""
    var libc = OwnedDLHandle("libc.so.6")
    var old_c = CStringSlice(unsafe_from_ptr=_to_cstr(old))
    var new_c = CStringSlice(unsafe_from_ptr=_to_cstr(new))
    var rc = libc.get_function[Int32]("rename")(
        old_c.unsafe_ptr(), new_c.unsafe_ptr()
    )
    if rc != 0:
        raise Error("mojoccl: rename to " + new + " failed, rc=" + String(rc))


def _atomic_write(final_path: String, tmp_path: String, content: String) raises:
    """write+rename: readers never observe a partially written file."""
    with open(tmp_path, "w") as f:
        f.write(content)
    _rename(tmp_path, final_path)


def write_rank_handle(
    dir: String, rank: Int, ordinal: Int, handle: Pointer[UInt8, MutAnyOrigin]
) raises:
    """Atomically publishes this rank's (device ordinal, IPC handle) pair.

    Written as space-separated decimal integers (matches
    proto/ipc_probe.mojo's handle-file convention) so there is no binary
    layout to keep in sync between writer and reader.
    """
    var s = String(ordinal)
    for i in range(HANDLE_BYTES):
        s += " " + String(Int(handle[unsafe_offset=i]))
    _atomic_write(
        dir + "/rank" + String(rank) + ".handle",
        dir + "/.rank" + String(rank) + ".handle.tmp",
        s,
    )


def write_rank_done(dir: String, rank: Int) raises:
    """Atomically publishes that this rank has opened every peer's IPC
    handle. Rank 0 waits for every rank's marker (`wait_for_done`) before
    removing `dir` (`remove_rendezvous_dir`) -- see ncclCommInitRank."""
    _atomic_write(
        dir + "/rank" + String(rank) + ".done",
        dir + "/.rank" + String(rank) + ".done.tmp",
        "1",
    )


def read_rank_handle(
    dir: String, rank: Int, out_handle: Pointer[UInt8, MutAnyOrigin]
) raises -> Int:
    """Reads a published handle file; returns the peer's device ordinal and
    fills `out_handle[0:64]`."""
    var path = dir + "/rank" + String(rank) + ".handle"
    var s: String
    with open(path, "r") as f:
        s = f.read()
    var k = 0
    var ordinal = -1
    for part in s.split(" "):
        if part.byte_length() == 0:
            continue
        if ordinal == -1:
            ordinal = Int(part)
        else:
            out_handle[unsafe_offset=k] = UInt8(Int(part))
            k += 1
    if k != HANDLE_BYTES or ordinal == -1:
        raise Error("mojoccl: malformed handle file " + path)
    return ordinal


def _wait_for_file(
    path: String, rank: Int, timeout_s: Float64, dir: String
) raises:
    var t0 = perf_counter_ns()
    var deadline_ns = Int(timeout_s * 1.0e9)
    while not Path(path).exists():
        if perf_counter_ns() - t0 > deadline_ns:
            raise Error(
                "mojoccl: timeout after "
                + String(timeout_s)
                + "s waiting for rank "
                + String(rank)
                + " at "
                + dir
            )
        sleep(0.02)


def wait_for_rank(dir: String, rank: Int, timeout_s: Float64) raises:
    """Blocks until `rank`'s handle file exists, or raises past the timeout."""
    _wait_for_file(
        dir + "/rank" + String(rank) + ".handle", rank, timeout_s, dir
    )


def wait_for_done(dir: String, rank: Int, timeout_s: Float64) raises:
    """Blocks until `rank`'s done marker (`write_rank_done`) exists, or
    raises past the timeout."""
    _wait_for_file(dir + "/rank" + String(rank) + ".done", rank, timeout_s, dir)


def remove_rendezvous_dir(dir: String, nranks: Int) raises:
    """Removes every rank's handle/done file, then the directory itself.

    Called once, by rank 0, once every rank's done marker has landed
    (normal completion) or on an init failure rank 0 hit after creating
    `dir` (`ncclGetUniqueId`). A file a peer never got around to writing is
    not an error here and is skipped; any other failure propagates, and so
    does the final `rmdir`: a non-empty directory after every known file
    was cleared means something unexpected is in there, worth surfacing.
    """
    for r in range(nranks):
        var handle = dir + "/rank" + String(r) + ".handle"
        if Path(handle).exists():
            remove(handle)
        var done = dir + "/rank" + String(r) + ".done"
        if Path(done).exists():
            remove(done)
    rmdir(dir)
