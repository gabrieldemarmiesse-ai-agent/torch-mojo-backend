"""Side device streams for the mojo device.

All compute, transfers, and stream-ordered frees ride ONE stream per device
(the MAX default stream); a `Stream` here is an extra stream for work that
overlaps it, e.g. NCCL collectives. Everything uses MAX's Python driver API
except the one primitive it lacks — waiting on an event recorded at a point
in time, enqueued arbitrarily later (at destructor time) — carried by
`tensor_holder.fence_event_record`/`fence_event_wait` in Mojo. Streams are
addressed by their per-stream `DeviceContext` pointer; `Stream.handle` (the
native CUstream) exists only for native libraries like NCCL. MAX/AsyncRT
thread-safety is undocumented; AMD should work through the same abstractions
but is unverified.

Rules for stream users: drain the kernel-call queue before any cross-stream
ordering (a queued launch is invisible to every stream); hold references to
tensors a side stream touches until its work completes (frees are
default-stream-ordered); nothing is fenced implicitly.
"""

import threading
from typing import TYPE_CHECKING, Protocol

import max.driver

from torch_mojo_backend import eager_kernels

if TYPE_CHECKING:
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import _TensorHolderModule


def _holder_mod() -> "_TensorHolderModule":
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (  # noqa: PLC0415 -- cycle: torch_mojo_tensor reaches this module (for `Stream`) before it defines this name
        _holder_mod as holder_mod,
    )

    return holder_mod()


class _ForeignUseRecorder(Protocol):
    """`torch_mojo_tensor._HolderOwner`, structurally (importing it here would
    be circular)."""

    def record_foreign_use(
        self, stream_ctx_ptr: int, event: object, owner_ctx_ptr: int
    ): ...


def default_stream_ctx_ptr(device: max.driver.Device) -> int:
    """DeviceContext pointer of `device`'s default stream — the owning stream
    a free fence must enqueue its waits onto.

    Read from the device itself (not `device.default_stream`): that is
    exactly the context pointer the kernel extensions allocate/free through,
    and a test asserts the two stay in sync.
    """
    return eager_kernels._ctx_ptr(device)


class Stream:
    """One extra device stream, orderable against the device's default.

    Infra-level, like c10::Stream: no current-stream state, so no
    ``with stream:`` — the torch-facing ``torch.mojo.Stream`` wrapper adds
    that (follow-up streams PR), the way torch.cuda.Stream does over c10.
    """

    def __init__(
        self, device: max.driver.Device, name: str, wrap_default: bool = False
    ):
        if device.api == "cpu" and not wrap_default:
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

    def wait_default_stream(self):
        """Order this stream after the default stream (drain the kernel-call
        queue first — see rule 1 above)."""
        if self.is_default:
            return
        self._stream.wait_for(self.device)

    def wait_stream(self, other: "Stream"):
        """Order later work here after everything `other` has right now.

        Tail-at-call-time semantics — the free fence wants the different
        thing (a point-in-time wait), and uses `record_fence_event` instead.
        """
        if other is not self:
            self._stream.wait_for(other._stream)

    def make_default_stream_wait(self):
        """Order subsequent default-stream work after this stream's work."""
        if self.is_default:
            return
        self.device.default_stream.wait_for(self._stream)

    # `object` rather than a real type on the two fence-event methods: the
    # class (`tensor_holder.FenceEvent`) only exists once the Mojo extension
    # has been built and imported, so there is nothing to annotate against.
    def wait_fence_event(self, event: object):
        _holder_mod().fence_event_wait(self.ctx_ptr, event)

    def record_fence_event(self) -> object:
        """Waitable by another stream at any later time — see module docstring."""
        return _holder_mod().fence_event_record(self.ctx_ptr)

    def record_event(
        self, event: max.driver.DeviceEvent | None = None
    ) -> max.driver.DeviceEvent:
        """Capture everything enqueued here so far. Pass an existing event
        to re-record it (e.g. a timing event) instead of minting a new one.
        """
        if event is None:
            return self._stream.record_event()
        self._stream.record_event(event)
        return event

    def query(self) -> bool:
        """True once this stream's enqueued work has completed.

        MAX has no non-blocking stream query, so this records a fresh event
        at the tail and asks whether it already fired. Not a hot path.
        """
        return self._stream.record_event().is_ready()

    def synchronize(self):
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
    holder: _ForeignUseRecorder, stream: Stream, owner_ctx_ptr: int | None = None
):
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
    holder: _ForeignUseRecorder, stream_ctx_ptr: int, owner_ctx_ptr: int
):
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
