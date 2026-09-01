"""Side device streams for the mojo device.

The eager backend runs compute, transfers, and stream-ordered frees on ONE
stream per device — the MAX DeviceContext's default stream (see
docs/kernel_call_queue.md). This module adds *extra* streams for work that
should overlap it, e.g. NCCL collectives (torch_mojo_backend/distributed).

A Stream wraps a `max.driver.DeviceStream`; everything goes through MAX's
Python driver API plus two Mojo functions in the `tensor_holder` extension
for one primitive Python lacks: a fence event recorded at a POINT IN TIME on
the foreign stream, waited on the owning stream ARBITRARILY LATER at
destructor time. `DeviceStream.wait_for(stream)` only waits on the other
stream's TAIL at call time, and the Python `DeviceEvent` has no wait-on-event
at all — neither fits a destructor-time wait, hence
`tensor_holder.fence_event_record` / `fence_event_wait`.
`Stream.record_event`/`is_ready`/`synchronize` are for host-side completion
polling instead (see the process group's completion worker).

Streams are addressed by their per-stream `DeviceContext` pointer, the same
handle the kernel extensions take for a device context. `Stream.handle` (the
native `CUstream`/`hipStream_t`) is kept only for native libraries that need
one (NCCL) — ordering never uses it.

Thread-safety: the completion worker polls events from its own thread while
holder destructors run on whatever thread drops the last reference; MAX/
AsyncRT's thread-safety is undocumented and this module does not serialize
around it.

AMD is expected to work through these same abstractions but is unverified —
no AMD hardware was available when this was written.

Correctness rules for stream users:

1. Drain the kernel-call queue before ordering a side stream against the
   default stream — a queued-but-unlaunched kernel is invisible to any
   stream (`deferred_compile.drain()`).
2. Hold a Python reference to every tensor a side stream reads or writes
   until its work is known complete. Frees are stream-ordered on the
   DEFAULT stream, so a buffer freed while a side stream still uses it is a
   use-after-free.
3. Anything the default stream must observe from a side stream needs an
   explicit fence (event or host wait); nothing is implicit.
"""

import threading
from types import ModuleType

import max.driver

# The Mojo extension carrying the fence-event functions, resolved on first
# use so that importing this module never triggers a Mojo kernel build.
_tensor_holder: ModuleType | None = None


def _holder_mod() -> ModuleType:
    global _tensor_holder
    if _tensor_holder is None:
        from torch_mojo_backend import eager_kernels

        _tensor_holder = eager_kernels.tensor_holder
    return _tensor_holder


def default_stream_ctx_ptr(device: max.driver.Device) -> int:
    """DeviceContext pointer of `device`'s default stream — the owning stream
    a free fence must enqueue its waits onto.

    Read from the device itself (not `device.default_stream`): that is
    exactly the context pointer the kernel extensions allocate/free through,
    and a test asserts the two stay in sync.
    """
    from torch_mojo_backend import eager_kernels

    return eager_kernels._ctx_ptr(device)


class Stream:
    """One extra device stream, orderable against the device's default."""

    def __init__(
        self, device: max.driver.Device, name: str, wrap_default: bool = False
    ) -> None:
        if device.api == "cpu":
            raise NotImplementedError(
                "side streams require an accelerator; the CPU device runs "
                "everything on one stream already"
            )
        self.device = device
        self.name = name
        self.is_default = wrap_default
        if wrap_default:
            self._stream = device.default_stream
            self.ctx_ptr = default_stream_ctx_ptr(device)
        else:
            self._stream = max.driver.DeviceStream(device)  # MAX owns; keep alive
            self.ctx_ptr = self._stream._device_context_ptr()
        # Native CUstream / hipStream_t, for native libraries that take one
        # (NCCL). Ordering never uses it — that goes through ctx_ptr.
        self.handle = self._stream.native_stream_handle

    def wait_default_stream(self) -> None:
        """Order this stream after the default stream (drain the kernel-call
        queue first — see rule 1 above)."""
        if self.is_default:
            return
        self._stream.wait_for(self.device)

    def make_default_stream_wait(self) -> None:
        """Order subsequent default-stream work after this stream's work."""
        if self.is_default:
            return
        self.device.default_stream.wait_for(self._stream)

    # `object` rather than a real type on the two fence-event methods: the
    # class (`tensor_holder.FenceEvent`) only exists once the Mojo extension
    # has been built and imported, so there is nothing to annotate against.
    def wait_fence_event(self, event: object) -> None:
        _holder_mod().fence_event_wait(self.ctx_ptr, event)

    def record_fence_event(self) -> object:
        """Waitable by another stream at any later time — see module docstring."""
        return _holder_mod().fence_event_record(self.ctx_ptr)

    def record_event(self) -> max.driver.DeviceEvent:
        return self._stream.record_event()

    def query(self) -> bool:
        """True once this stream's enqueued work has completed.

        MAX has no non-blocking stream query, so this records a fresh event
        at the tail and asks whether it already fired. Not a hot path.
        """
        return self._stream.record_event().is_ready()

    def synchronize(self) -> None:
        self._stream.synchronize()


_STREAMS: dict[tuple[int, str], Stream] = {}
_STREAMS_LOCK = threading.Lock()


def get_stream(device: max.driver.Device, name: str) -> Stream:
    """The (device, name) stream, created on first use. Thread-safe.

    The reserved name "default" wraps the device's default stream instead of
    creating a new one.
    """
    key = (device.id, name)
    with _STREAMS_LOCK:
        stream = _STREAMS.get(key)
        if stream is None:
            stream = Stream(device, name, wrap_default=(name == "default"))
            _STREAMS[key] = stream
        return stream


def default_stream(device: max.driver.Device) -> Stream:
    """The stream view of the device's default stream."""
    return get_stream(device, "default")


def record_use(
    holder: object, stream: Stream, owner_ctx_ptr: int | None = None
) -> None:
    """Order `holder`'s eventual free after `stream` work enqueued so far.

    The record_stream analog for this backend. MAX frees a buffer
    stream-ordered on its OWNING stream only — a reader on another stream
    races the pool's reuse (see the free-race stress test) — so any tensor a
    side stream reads or writes must be recorded here before its last
    reference may drop.

    `owner_ctx_ptr` defaults to the device's default stream, where this
    backend allocates and frees everything.
    """
    if owner_ctx_ptr is None:
        owner_ctx_ptr = default_stream_ctx_ptr(stream.device)
    if stream.ctx_ptr == owner_ctx_ptr:
        return  # the free is already ordered on this stream
    holder.record_foreign_use(
        stream.ctx_ptr, stream.record_fence_event(), owner_ctx_ptr
    )


def record_use_on_stream_ctx(
    holder: object, stream_ctx_ptr: int, owner_ctx_ptr: int
) -> None:
    """``record_use`` for a bare stream context pointer instead of a Stream.

    ``aten::record_stream`` hands an opaque ``(device, stream_id)`` pair, and
    mojo streams use their per-stream ``DeviceContext`` pointer as their
    ``stream_id`` — so this fences a foreign stream MAX knows nothing else
    about, without needing a Stream object. Id 0 means the device's default
    stream in c10's encoding, which for this backend is the owning stream
    too.
    """
    if stream_ctx_ptr == 0 or stream_ctx_ptr == owner_ctx_ptr:
        return
    holder.record_foreign_use(
        stream_ctx_ptr, _holder_mod().fence_event_record(stream_ctx_ptr), owner_ctx_ptr
    )
