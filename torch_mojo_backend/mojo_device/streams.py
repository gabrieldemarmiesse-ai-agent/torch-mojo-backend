"""torch.Stream / torch.Event support for the mojo device.

PyTorch's generic torch.Stream/torch.Event route through a C++ device guard
that is a stub for Python-backed PrivateUse1 devices: stream construction
always reports stream id 0 and record/wait/query are silent no-ops.
register_mojo_devices() dispatches mojo devices to the classes here instead
— the same remedy the backend already applies to
torch.accelerator.synchronize. User streams are
mojo_device.device_streams.Stream instances; see that module's docstring
for the completion-event vs fence-event split.

Stream derives from the real torch._C.Stream so C++ argument parsing
(THPStream_Check) accepts it wherever a schema declares a Stream — notably
aten::record_stream. Its stream_id is the device stream's per-stream MAX
DeviceContext pointer, an opaque int to torch and the token
aten::record_stream uses to find the stream to fence a buffer's free
against (mojo_device/aten_ops/streams.py). The native
CUstream/hipStream_t stays available as Stream.native_handle.

Execution semantics: eager mojo kernels always EXECUTE on the device's
default stream today. `with stream:` only tracks the current stream per
thread (current_stream()/event/wait act on it); it does not redirect
kernel launches, so device-agnostic pipelining code gets CUDA-single-
stream semantics — correct, without the extra compute concurrency.
Redirecting launches through per-stream MAX DeviceContexts is the known
follow-up.
"""

import itertools
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias

import max.driver
import torch
from torch._C._autograd import DeviceType

from torch_mojo_backend.mojo_device import device_streams

_DeviceLike: TypeAlias = torch.device | str | int | None

_PRIVATEUSE1_DEVICE_TYPE = DeviceType.PrivateUse1.value
_user_stream_ids = itertools.count()
_current_stacks = threading.local()  # {device_index: [Stream, ...]} per thread
_default_streams: dict[int, "Stream"] = {}
_default_streams_lock = threading.Lock()


def _current_device_index() -> int:
    from torch_mojo_backend.mojo_device import (  # noqa: PLC0415 -- cycle: torch_mojo_device_module imports this module (see the E402 note at its end)
        torch_mojo_device_module,
    )

    return torch_mojo_device_module.current_device()


def _resolve_index(device: _DeviceLike) -> int:
    if device is None:
        return _current_device_index()
    if isinstance(device, int):
        return device
    resolved = torch.device(device)
    if resolved.type != "mojo":
        raise ValueError(f"expected a mojo device, got {resolved}")
    return resolved.index if resolved.index is not None else _current_device_index()


def _max_device_of(index: int) -> max.driver.Device:
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (  # noqa: PLC0415 -- same cycle, one module further: torch_mojo_tensor imports torch_mojo_device_module
        find_equivalent_max_device,
    )

    return find_equivalent_max_device(torch.device("mojo", index))


class Event:
    """Mirror of torch.Event for mojo devices, backed by a driver event.

    Both backing events are created lazily at first ``record``, on the
    device of the stream being recorded. ``blocking`` is accepted and
    ignored (host waits always yield); ``interprocess`` is unsupported.
    """

    def __init__(
        self,
        device: _DeviceLike = None,
        *,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ):
        if interprocess:
            raise NotImplementedError(
                "interprocess events are not supported on the mojo device"
            )
        self.enable_timing = enable_timing
        # driver event: host-side query/synchronize/timing. fence event:
        # what another stream can be made to wait on (device_streams.py).
        self._event: max.driver.DeviceEvent | None = None
        self._fence: object | None = None
        self._device_index: int | None = (
            None if device is None else _resolve_index(device)
        )

    @property
    def device(self) -> torch.device | None:
        if self._device_index is None:
            return None
        return torch.device("mojo", self._device_index)

    def record(self, stream: "Stream | None" = None):
        if stream is None:
            stream = current_stream(self._device_index)
        device_stream = stream._device_stream
        if self._event is None:
            self._event = max.driver.DeviceEvent(
                device_stream.device, enable_timing=self.enable_timing
            )
        # Back to back on one stream: both mark the same point in it.
        device_stream.record_event(self._event)
        self._fence = device_stream.record_fence_event()
        self._device_index = stream.device_index

    def wait(self, stream: "Stream | None" = None):
        if self._fence is None:
            raise RuntimeError("Event must be recorded before it can be waited on")
        if stream is None:
            stream = current_stream(self._device_index)
        stream._device_stream.wait_fence_event(self._fence)

    def query(self) -> bool:
        return True if self._event is None else self._event.is_ready()

    def synchronize(self):
        if self._event is not None:
            self._event.synchronize()

    def elapsed_time(self, end_event: "Event") -> float:
        if not (self.enable_timing and end_event.enable_timing):
            raise RuntimeError("both events must be created with enable_timing=True")
        if self._event is None or end_event._event is None:
            raise RuntimeError("both events must be recorded before elapsed_time")
        return self._event.elapsed_time(end_event._event)

    @classmethod
    def from_ipc_handle(cls, device: _DeviceLike, handle: bytes) -> "Event":
        raise NotImplementedError(
            "interprocess events are not supported on the mojo device"
        )

    def __repr__(self) -> str:
        return f"torch.mojo.Event(device={self.device}, recorded={self._event is not None})"


class Stream(torch._C.Stream):
    """Mirror of torch.Stream for mojo devices, backed by a device stream.

    Derived from the real ``torch._C.Stream`` (captured here rather than via
    ``torch.Stream``, which ``register_mojo_devices()`` later replaces with
    a dispatching wrapper) so C++ argument parsing (``THPStream_Check``)
    accepts it wherever a schema declares a ``Stream`` — notably
    ``aten::record_stream``. Every ordering method below overrides the
    base's stub PrivateUse1 device-guard no-op.
    """

    _device_stream: device_streams.Stream
    _index: int
    priority: int

    def __new__(
        cls,
        device: _DeviceLike = None,
        priority: int = 0,
        *,
        _device_stream: device_streams.Stream | None = None,
    ) -> "Stream":
        index = _resolve_index(device)
        if _device_stream is None:
            _device_stream = device_streams.Stream(
                _max_device_of(index), f"user-{next(_user_stream_ids)}"
            )
        self = super().__new__(
            cls, _device_stream.ctx_ptr, index, _PRIVATEUSE1_DEVICE_TYPE
        )
        self._device_stream = _device_stream
        self._index = index
        # MAX supports stream priorities but max.driver.DeviceStream's
        # constructor doesn't take one yet; accepted and reported as 0.
        self.priority = 0
        return self

    def __init__(self, *args: object, **kwargs: object):
        # Everything happens in __new__ (the base type is a C type whose
        # tp_new takes the three identity fields); accept and ignore the
        # constructor arguments so `Stream(device, priority=...)` works.
        pass

    @property
    def device(self) -> torch.device:
        return torch.device("mojo", self._index)

    @property
    def device_index(self) -> int:
        return self._index

    @property
    def device_type(self) -> str:
        return "mojo"

    @property
    def stream_id(self) -> int:
        return self._device_stream.ctx_ptr

    @property
    def native_handle(self) -> int:
        return self._device_stream.handle

    def query(self) -> bool:
        self._fence_pending_comm_work()
        return self._device_stream.query()

    def synchronize(self):
        self._fence_pending_comm_work()
        self._device_stream.synchronize()

    def _fence_pending_comm_work(self):
        """Both observations above answer "is my work done?" for this stream.

        On the default stream that has to include a collective still in
        flight on the comm stream, which is ordered onto it lazily
        (mojo_device/comm_fence.py) — so fence before observing, once.
        """
        if self._device_stream.is_default:
            from torch_mojo_backend.mojo_device import (  # noqa: PLC0415 -- same cycle: comm_fence imports torch_mojo_tensor
                comm_fence,
            )

            comm_fence.fence_device(self._index)

    def wait_event(self, event: Event):
        event.wait(self)

    def wait_stream(self, other: "Stream | torch.Stream"):
        """Order this stream after everything enqueued on `other` so far."""
        self._device_stream.wait_stream(_stream_of(other, self._index))

    def record_event(self, event: Event | None = None) -> Event:
        if event is None:
            event = Event(device=self._index)
        event.record(self)
        return event

    def is_capturing(self) -> bool:
        return False  # no graph capture on the mojo device

    def __enter__(self) -> "Stream":
        stacks = getattr(_current_stacks, "stacks", None)
        if stacks is None:
            stacks = _current_stacks.stacks = {}
        stacks.setdefault(self._index, []).append(self)
        return self

    def __exit__(self, *exc_info: object):
        _current_stacks.stacks[self._index].pop()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Stream)
            and other._device_stream.ctx_ptr == self._device_stream.ctx_ptr
        )

    def __hash__(self) -> int:
        return hash(self._device_stream.ctx_ptr)

    def __repr__(self) -> str:
        return (
            f"torch.mojo.Stream(device=mojo:{self._index}, "
            f"native_handle=0x{self._device_stream.handle:x}"
            f"{', default' if self._device_stream.is_default else ''})"
        )


def _stream_of(stream: torch.Stream, index: int) -> device_streams.Stream:
    """The device stream of ours, or of a stub torch.Stream (= the default
    stream: the stub PrivateUse1 guard can only ever describe that one)."""
    if isinstance(stream, Stream):
        return stream._device_stream
    return default_stream(index)._device_stream


def default_stream(device: _DeviceLike = None) -> Stream:
    index = _resolve_index(device)
    with _default_streams_lock:
        stream = _default_streams.get(index)
        if stream is None:
            stream = Stream(
                device=index,
                _device_stream=device_streams.default_stream(_max_device_of(index)),
            )
            _default_streams[index] = stream
        return stream


def current_stream(device: _DeviceLike = None) -> Stream:
    index = _resolve_index(device)
    stacks = getattr(_current_stacks, "stacks", None)
    if stacks and stacks.get(index):
        return stacks[index][-1]
    return default_stream(index)


def set_stream(stream: Stream):
    """Make `stream` this thread's ambient current stream (no nesting)."""
    stacks = getattr(_current_stacks, "stacks", None)
    if stacks is None:
        stacks = _current_stacks.stacks = {}
    stacks[stream.device_index] = [stream]


@contextmanager
def stream(stream: "Stream | None") -> Iterator[None]:
    """Context-manager helper mirroring torch.cuda.stream(s)."""
    if stream is None:
        yield
        return
    with stream:
        yield
