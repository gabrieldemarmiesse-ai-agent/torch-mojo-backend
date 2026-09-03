"""Lazy default-stream fencing for collectives that ran on the comm stream.

A collective enqueued on the "nccl" side stream completes its Work future at
enqueue time, so ``wait()`` never blocks the host (see
``distributed/process_group.py`` for why that matters). The device ordering
that future no longer carries is carried by the tensors instead: every buffer
a collective reads or writes is recorded here as *pending*, and the first
default-stream consumer of one makes the default stream wait on the comm
stream before it runs. A stream is FIFO, so that single wait orders the
default stream after every collective enqueued so far — hence a per-buffer
record but a per-device fence.

Buffers are keyed by ``id(tensor._holder)``: views share their base's holder,
so a write into a bucket view fences on the whole bucket. Identity keys can
be recycled after a free, which can only add a fence, never drop one.

``PENDING`` is the fast path — empty means no collective is in flight, the
only state the per-op hook in ``register.py`` has to distinguish. That hook
captures the dict by reference, so it is mutated in place, never rebound.
"""

import threading

from torch_mojo_backend.mojo_device import device_streams
from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

PENDING: dict[int, int] = {}  # id(holder) -> mojo device index
_COMM_STREAMS: dict[int, device_streams.Stream] = {}
# Collectives are issued from the autograd thread while the main thread runs
# the optimizer; the truthiness poll on the hot path stays lock-free.
# Reentrant, and every fence below idempotent, because the fence drains the
# kernel-call queue while holding it and a launch may fence again.
_LOCK = threading.RLock()


# A record only leaves PENDING when something consumes its buffer, so a
# program that collectives onto fresh tensors and never reads them would grow
# it without bound. Fence at this many (DDP reuses one buffer per bucket and
# sits around 75) rather than carry a leak.
_MAX_PENDING = 4096


def mark_pending(
    index: int, stream: device_streams.Stream, tensors: tuple[TorchMojoTensor, ...]
):
    """Record every buffer a collective just enqueued on `stream` touches."""
    with _LOCK:
        _COMM_STREAMS[index] = stream
        if len(PENDING) >= _MAX_PENDING:
            fence_all()  # reentrant lock
        for tensor in tensors:
            PENDING[id(tensor._holder)] = index


def _keys_locked(index: int) -> list[int]:
    return [key for key, value in PENDING.items() if value == index]


def _fence_locked(index: int):
    keys = _keys_locked(index)
    stream = _COMM_STREAMS.get(index)
    if not keys or stream is None:
        return
    # Queued launches were issued before this fence and must stay before it
    # on the default stream (rule 1 in device_streams.py).
    from torch_mojo_backend.mojo_device import deferred_compile

    deferred_compile.drain()
    stream.make_default_stream_wait()
    for key in keys:  # only once the wait is on the stream
        PENDING.pop(key, None)


def fence_device(index: int):
    """Order the default stream after every collective pending on `index`."""
    if not PENDING:
        return
    with _LOCK:
        _fence_locked(index)


def fence_all():
    """``fence_device`` for every device with a collective in flight."""
    if not PENDING:
        return
    with _LOCK:
        for index in sorted(set(PENDING.values())):
            _fence_locked(index)


def fence_tensor(tensor: TorchMojoTensor):
    """Fence before a host read of `tensor` that no eager op mediates."""
    if not PENDING:
        return
    index = PENDING.get(id(tensor._holder))
    if index is not None:
        fence_device(index)


def discard(index: int):
    """Forget `index`'s pending buffers without fencing — for callers that
    already ordered or completed the comm stream themselves."""
    if not PENDING:
        return
    with _LOCK:
        for key in _keys_locked(index):
            PENDING.pop(key, None)


def _pending_index(value: object) -> int | None:
    if isinstance(value, TorchMojoTensor):
        return PENDING.get(id(value._holder))
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, TorchMojoTensor):
                index = PENDING.get(id(item._holder))
                if index is not None:
                    return index
    return None


def fence_pending_args(args: tuple[object, ...], kwargs: dict[str, object]):
    """Fence if any argument of an eager op reads or writes a pending buffer.

    Reached only while ``PENDING`` is non-empty. Lists are scanned one level
    deep, which is where the foreach and concat ops keep their tensors.
    """
    for value in args:
        index = _pending_index(value)
        if index is not None:
            fence_device(index)
            return
    for value in kwargs.values():
        index = _pending_index(value)
        if index is not None:
            fence_device(index)
            return
