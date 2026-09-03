import functools
import math
import sys
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Protocol, cast, runtime_checkable

import max.driver
import torch
from max.driver import CPU
from max.dtype import DType
from max.experimental.torch import max_dtype_to_torch

from torch_mojo_backend import eager_kernels
from torch_mojo_backend.eager_kernels.data_movement_ops import DataMovementExtension
from torch_mojo_backend.eager_kernels.output_specs import (
    _submit_prepared_into,
    _TensorOutputSpec,
)
from torch_mojo_backend.mojo_device import objc_autorelease, torch_mojo_device_module


class _MojoTensorHolder(Protocol):
    """The Mojo `TensorHolder` owning one device allocation (no Python stubs)."""

    def data_ptr(self) -> int: ...
    def get_nbytes(self) -> int: ...


@runtime_checkable
class _TensorHolderModule(Protocol):
    """The subset of the JIT-compiled `tensor_holder` Mojo extension this
    file calls directly. The module itself has no stubs (its source is
    Mojo, not Python); every member here is a raw PythonObject-in,
    PythonObject-out native call, so `object` is the honest signature."""

    def alloc(self, ctx_ptr: int, nbytes: int) -> tuple[_MojoTensorHolder, int]: ...
    def alloc_from_host(
        self, ctx_ptr: int, data_ptr: int, nbytes: int
    ) -> tuple[_MojoTensorHolder, int, object]: ...
    def copy_to_host(
        self, ctx_ptr: int, src_ptr: int, dst_ptr: int, nbytes: int
    ) -> object: ...
    def copy_to_pinned_host(
        self, ctx_ptr: int, src_ptr: int, nbytes: int
    ) -> tuple[object, int]: ...
    def CopyStrided(self, *args: object) -> object: ...
    def copy_d2d(self, ctx_ptr: int, dst_ptr: int, src_ptr: int, nbytes: int): ...
    def fence_event_record(self, ctx_ptr: int) -> object: ...
    def fence_event_wait(self, ctx_ptr: int, event: object): ...


# The Mojo extension module (torch_mojo_backend.eager_kernels.tensor_holder),
# resolved lazily so that importing torch_mojo_backend never triggers a Mojo
# kernel compile.
_tensor_holder: _TensorHolderModule | None = None


def _holder_mod() -> _TensorHolderModule:
    global _tensor_holder
    holder = _tensor_holder
    if holder is None:
        holder = cast(_TensorHolderModule, eager_kernels.tensor_holder)
        _tensor_holder = holder
    return holder


class _ReadyEvent(Protocol):
    """The one thing the pending-transfer queues ask of a recorded event
    (`max.driver.DeviceEvent`; tests inject stand-ins)."""

    def is_ready(self) -> bool: ...


# GPU H2D copies consume a MAX-owned pinned staging allocation asynchronously.
# Keep that transfer owner alive until an event recorded behind its DMA
# completes. This mirrors the lifetime tracking performed by CUDA's pinned
# memory allocator without depending on torch-cuda.
_PENDING_H2D: dict[max.driver.Device, deque[tuple[_ReadyEvent, object]]] = {}
_PENDING_H2D_LOCK = threading.Lock()

# A non-blocking D2H returns a CPU tensor that aliases a MAX-owned pinned host
# allocation. DLPack ties that owner to the returned tensor, while this queue
# also retains it until the DMA event completes if the tensor dies early.
_PENDING_D2H: dict[max.driver.Device, deque[tuple[_ReadyEvent, object]]] = {}
_PENDING_D2H_LOCK = threading.Lock()

# A stream/event failure is already a fatal device condition, but raw-pointer
# lifetime must remain safe while the exception propagates. If both event
# recording and recovery synchronization fail, retain that transfer owner for
# the process lifetime rather than risk freeing memory still used by DMA.
_FAILED_TRANSFER_OWNERS: dict[max.driver.Device, list[tuple[object, object]]] = {}
_FAILED_TRANSFER_OWNERS_LOCK = threading.Lock()

# PyTorch's Python PrivateUse1 guard currently advertises one C++ autograd
# device queue, at index zero. Keep the storage-less wrapper TensorImpl on that
# bookkeeping device, as ``_acc.create_empty_tensor`` did. ``_torch_device`` and
# the public ``device`` property continue to carry the real Mojo device.
_WRAPPER_TENSORIMPL_DEVICE = torch.device("privateuseone:0")

# Captured at import: `sys` module globals may already be torn down when a
# late __del__ runs.
_is_finalizing = sys.is_finalizing


class _HolderOwner:
    """Shared owner of one device allocation, fencing its stream-ordered free.

    MAX frees a buffer stream-ordered on its OWNING stream only, so a reader
    on another stream races the pool's reuse (measured — see
    device_streams.record_use). ``_wait`` caches
    ``tensor_holder.fence_event_wait`` at record time because ``__del__`` can
    run during interpreter shutdown, when module globals are already cleared.
    """

    __slots__ = ("_holder", "_events", "_owner_ctx", "_wait")

    # A compile-backend output is owned by its MAX Buffer instead (compiler.py).
    _holder: _MojoTensorHolder | max.driver.Buffer
    _events: dict[int, object] | None  # {foreign stream ctx ptr: newest FenceEvent}
    _owner_ctx: int | None
    _wait: Callable[[int, object], None] | None

    def __init__(self, holder: "_MojoTensorHolder | max.driver.Buffer"):
        self._holder = holder
        self._events = None
        self._owner_ctx = None
        self._wait = None

    def data_ptr(self) -> int:
        holder = self._holder
        if isinstance(holder, max.driver.Buffer):
            return holder._data_ptr()
        return holder.data_ptr()

    def get_nbytes(self) -> int:
        holder = self._holder
        if isinstance(holder, max.driver.Buffer):
            return holder.num_elements * holder.dtype.size_in_bytes
        return holder.get_nbytes()

    def record_foreign_use(
        self, stream_ctx_ptr: int, event: object, owner_ctx_ptr: int
    ):
        if self._events is None:
            self._events = {}
            self._wait = _holder_mod().fence_event_wait
        self._owner_ctx = owner_ctx_ptr
        # A later event on the same stream dominates the earlier one; the
        # replaced event is released when this rebinding drops it.
        self._events[stream_ctx_ptr] = event

    def __del__(self):
        events, wait, owner_ctx = self._events, self._wait, self._owner_ctx
        if not events or wait is None or owner_ctx is None or _is_finalizing():
            return
        for event in events.values():
            wait(owner_ctx, event)


def _ctx_ptr(device: max.driver.Device) -> int:
    # Rebinds this module-level name to the real (cached) implementation on
    # first use, so the lazy import costs one call, not one per call.
    global _ctx_ptr
    from torch_mojo_backend.eager_kernels import _ctx_ptr as real_ctx_ptr

    # Self-rebind via `global`: ty compares the two same-shaped `def`s
    # nominally, not structurally, and treats reassigning a function to
    # (what is, at runtime,) itself as unsound.
    _ctx_ptr = real_ctx_ptr  # ty: ignore[invalid-assignment]
    return real_ctx_ptr(device)


def _retain_failed_transfer_owner(device: max.driver.Device, owner: object) -> object:
    token = object()
    with _FAILED_TRANSFER_OWNERS_LOCK:
        _FAILED_TRANSFER_OWNERS.setdefault(device, []).append((token, owner))
    return token


def _forget_failed_transfer_owner(device: max.driver.Device, token: object):
    with _FAILED_TRANSFER_OWNERS_LOCK:
        retained = _FAILED_TRANSFER_OWNERS.get(device)
        if retained is None:
            return
        retained[:] = [entry for entry in retained if entry[0] is not token]
        if not retained:
            _FAILED_TRANSFER_OWNERS.pop(device, None)


def _record_h2d_source(device: max.driver.Device, source: object, non_blocking: bool):
    """Retain a pinned transfer owner until its default-stream H2D ends."""
    # MAX's CPU device uses a worker pool whose copies are not stream-ordered
    # with kernels. The Mojo helper drains it before returning; keep the Python
    # side blocking as well rather than advertising unsupported async behavior.
    if device == CPU():
        non_blocking = False

    try:
        event = device.default_stream.record_event()
    except Exception:
        # Stash before recovery: if synchronization also fails, retaining for
        # process lifetime is safer than freeing a raw DMA source prematurely.
        token = _retain_failed_transfer_owner(device, source)
        device.default_stream.synchronize()
        _forget_failed_transfer_owner(device, token)
        _release_synchronized_h2d_sources(device)
        raise

    if not non_blocking:
        event.synchronize()
        _release_synchronized_h2d_sources(device)
        return

    with _PENDING_H2D_LOCK:
        pending = _PENDING_H2D.setdefault(device, deque())
        # Retain the current DMA owner before querying older events. Event
        # queries can fail; unwinding must not free the just-enqueued source.
        pending.append((event, source))
        while pending and pending[0][0].is_ready():
            pending.popleft()
        # Do not impose a count-based wait here: a burst can legitimately have
        # many copies behind long-running GPU work. The FIFO is reaped on every
        # transfer and after explicit/blocking synchronization, while each
        # event keeps its exact source alive until DMA completion.


def _release_synchronized_h2d_sources(device: max.driver.Device):
    """Drop ready sources after the caller synchronized ``device``'s stream.

    Another thread may enqueue a transfer between the stream synchronization
    and this cleanup. Checking each event under the queue lock preserves that
    post-sync source until its own DMA completes.
    """
    with _PENDING_H2D_LOCK:
        pending = _PENDING_H2D.get(device)
        while pending and pending[0][0].is_ready():
            pending.popleft()
        if not pending:
            _PENDING_H2D.pop(device, None)


def _record_d2h_owner(device: max.driver.Device, owner: object):
    """Retain a pinned D2H allocation until its default-stream DMA ends."""
    try:
        event = device.default_stream.record_event()
    except Exception:
        token = _retain_failed_transfer_owner(device, owner)
        device.default_stream.synchronize()
        _forget_failed_transfer_owner(device, token)
        _release_synchronized_h2d_sources(device)
        _release_synchronized_d2h_owners(device)
        raise

    with _PENDING_D2H_LOCK:
        pending = _PENDING_D2H.setdefault(device, deque())
        # Retain the current source/destination before querying older events;
        # an event-query exception must not drop memory still used by DMA.
        pending.append((event, owner))
        while pending and pending[0][0].is_ready():
            pending.popleft()


def _release_synchronized_d2h_owners(device: max.driver.Device):
    """Drop pinned D2H owners whose stream events have completed."""
    with _PENDING_D2H_LOCK:
        pending = _PENDING_D2H.get(device)
        while pending and pending[0][0].is_ready():
            pending.popleft()
        if not pending:
            _PENDING_D2H.pop(device, None)


@runtime_checkable
class MojoTensorLike(Protocol):
    """Anything carrying the core Mojo eager payload metadata.

    Payload-level helpers only read these attributes, and host-contract
    tests exercise them with lightweight stand-ins, so their signatures
    declare this structural contract rather than the concrete wrapper.
    """

    _shape: tuple[int, ...]
    # Namespaced on purpose. PyTorch library code assigns its own bookkeeping
    # onto parameter objects, and because a mojo Parameter *is* the wrapper
    # instance (nn.Parameter.__new__ takes the custom-tensor path and returns
    # `data.detach()`), such a write lands directly in this payload. FSDP1's
    # `FlatParameter._init_metadata` sets `_strides` to the per-parameter
    # stride list, which silently replaced the layout strides here and made
    # every later view op read a list of tuples. Keep payload names that torch
    # could plausibly reuse under a `_mojo_` prefix.
    _mojo_strides: tuple[int, ...]
    _dtype: DType
    _device: object


def _device_of(tensor: MojoTensorLike) -> max.driver.Device:
    """MojoTensorLike._device is `object` for host-contract test doubles
    (checked invariantly, so even TorchMojoTensor's own concrete Device
    can't narrow the Protocol member); real operands are always a
    TorchMojoTensor with a genuine Device."""
    return cast(max.driver.Device, tensor._device)


def _dispatch_entry(
    func: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    """``deferred_compile.dispatch``, resolved on first use.

    ``deferred_compile`` imports the call queue this module feeds, so it
    cannot be imported at module scope. Rebinding the global on the first
    dispatch keeps that direction intact and leaves the steady state at one
    plain global lookup instead of a per-op ``sys.modules`` hit (the same
    trick as ``output_specs._alloc``).
    """
    global _dispatch_entry
    from torch_mojo_backend.mojo_device import deferred_compile

    # Same nominal-vs-structural self-rebind quirk as _ctx_ptr above, even
    # though both sides print identically.
    _dispatch_entry = deferred_compile.dispatch  # ty: ignore[invalid-assignment]
    return _dispatch_entry(func, args, kwargs)


@runtime_checkable
class _SynchronizableStream(Protocol):
    def synchronize(self): ...


@runtime_checkable
class _SynchronizableDevice(Protocol):
    """What allocation recovery needs from a device: a default stream it can
    synchronize. ``max.driver.Device`` satisfies it; host-contract tests use
    lightweight stand-ins.

    ``default_stream`` is a ``@property`` (not a plain data attribute) so it
    is checked covariantly: a data member would be checked invariantly and
    ``max.driver.Device``'s own concrete ``default_stream`` -> ``DeviceStream``
    property could never satisfy it.
    """

    @property
    def default_stream(self) -> _SynchronizableStream: ...


def _alloc_with_recovery(
    device: _SynchronizableDevice, nbytes: int
) -> tuple[_MojoTensorHolder, int]:
    """One device allocation, with the reactive last resort under the budget.

    When the allocator refuses, the largest reclaimable set is whatever the
    kernel-call queue still retains: drain it (launching every pending item,
    waiting builds out), synchronize the device so the stream-ordered frees
    actually land, and retry exactly once. This cannot defeat size-class
    carve-up — the arena never splits or merges, which is what the proactive
    run-ahead budget prevents — but it converts pressure the budget cannot
    see (a model simply large for the card, external processes) into a
    stall instead of a failure. Non-OOM errors and a second failure
    propagate untouched.
    """
    holder_mod = _holder_mod()
    # _ctx_ptr wants a real max.driver.Device; production always passes one
    # here (only its default_stream is used above), and host-contract tests
    # that pass a lighter _SynchronizableDevice stand-in monkeypatch
    # `_ctx_ptr` itself rather than calling into the real implementation.
    ctx_ptr = _ctx_ptr(cast(max.driver.Device, device))
    try:
        return holder_mod.alloc(ctx_ptr, nbytes)
    except Exception as exc:
        from torch_mojo_backend.eager_kernels import call_queue, is_device_oom

        if not is_device_oom(exc):
            raise
        sys.stderr.write(
            f"torch-mojo-backend: device allocation of {nbytes} bytes failed; "
            f"draining {len(call_queue._QUEUE)} queued launch(es), "
            "synchronizing, and retrying once...\n"
        )
        call_queue.drain()
        device.default_stream.synchronize()
        return holder_mod.alloc(ctx_ptr, nbytes)


def _row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _compute_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """torch's relaxed contiguity: size-1 dims never break contiguity."""
    expected = 1
    for size, stride in zip(reversed(shape), reversed(strides)):
        if size == 1:
            continue
        if stride != expected:
            return False
        expected *= size
    return True


# max DType -> torch dtype, cached as a plain dict: max_dtype_to_torch is
# called once per tensor wrapper created (~600/decode step).
_TORCH_DTYPE_OF: dict[DType, torch.dtype] = {}


def _torch_dtype_of(dtype: DType) -> torch.dtype:
    td = _TORCH_DTYPE_OF.get(dtype)
    if td is None:
        td = _TORCH_DTYPE_OF[dtype] = max_dtype_to_torch(dtype)
    return td


# max.driver.Device -> torch.device, cached like _TORCH_DTYPE_OF. Computed
# once per wrapper created so the `device` property is a plain attribute
# read — which also lets dynamo trace `x.device` inside compiled functions
# (the property body must not construct max.driver objects).
_TORCH_DEVICE_OF: dict[max.driver.Device, torch.device] = {}


def _torch_device_of(device: max.driver.Device) -> torch.device:
    td = _TORCH_DEVICE_OF.get(device)
    if td is None:
        if device == CPU():
            td = torch_mojo_device_module.cpu()
        else:
            td = torch.device(f"mojo:{device.id}")
        _TORCH_DEVICE_OF[device] = td
    return td


# Strided kernels take shapes/strides padded to rank 8 with LEADING entries.
MAX_RANK = 8


def _pad8(values: Sequence[int], fill: int) -> tuple[int, ...]:
    values = tuple(values)
    if len(values) > MAX_RANK:
        raise NotImplementedError(
            f"mojo tensors support at most rank {MAX_RANK}, got {len(values)}"
        )
    return (fill,) * (MAX_RANK - len(values)) + values


class TorchMojoTensor(torch.Tensor):
    """Eager mojo tensor.

    A storage-less ``PrivateUse1`` wrapper subclass whose payload is:

    - `_holder`: a Mojo `TensorHolder` owning the device allocation. Views
      share the *same* holder object; CPython's refcount on it is the
      ownership mechanism, and the last drop enqueues the stream-ordered
      free (see docs/strided_owning_tensors_design.md).
    - Layout metadata as plain Python attributes (`_ptr`, `_shape`,
      `_mojo_strides` in elements, `_offset` in elements from the allocation
      start, `_dtype` as a max DType, `_numel`, `_itemsize`, `_device`,
      `_is_contiguous`).

    The wrapper's Python dispatch key only redispatches to the numerical
    ``PrivateUse1`` kernels. Autograd, autocast, and ADInplaceOrView remain
    PyTorch-owned layers above it. In particular, PyTorch can reconstruct a
    detached wrapper through ``__torch_dispatch__`` when it saves an operator
    output for backward, preserving the Python-side allocation payload.
    """

    # Declared here (not just assigned in `_make`) so the type checker
    # resolves reads across the codebase; matches `_PAYLOAD_ATTRIBUTES` below.
    _holder: _HolderOwner
    _ptr: int
    _shape: tuple[int, ...]
    _mojo_strides: tuple[int, ...]
    _offset: int
    _dtype: DType
    _itemsize: int
    _numel: int
    _device: max.driver.Device
    _torch_device: torch.device
    _is_contiguous: bool

    @classmethod
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: Sequence[type],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        """Redispatch wrapper operations to the existing Mojo backend kernels."""
        # Give higher-priority wrappers such as FakeTensor and
        # FunctionalTensor their opportunity to handle mixed-subclass calls.
        if not all(issubclass(cls, tensor_type) for tensor_type in types):
            return NotImplemented

        # The MAX Metal runtime autoreleases ObjC objects per kernel launch;
        # without a periodically drained pool they leak until macOS SIGKILLs
        # the process (see objc_autorelease.py). No-op off macOS.
        objc_autorelease.note_op_dispatched()

        # The deferred-compile layer executes ops while kernel variants are
        # still building in the background (and is a plain pass-through to
        # the ordinary PrivateUse1 path when no compile is in flight).
        return _dispatch_entry(func, args, kwargs or {})

    @classmethod
    def _make(
        cls,
        holder: "_HolderOwner | _MojoTensorHolder | max.driver.Buffer",
        ptr: int,
        shape: Sequence[int],
        strides: Sequence[int],
        offset: int,
        dtype: DType,
        device: max.driver.Device,
        contiguous: bool | None = None,
    ) -> "TorchMojoTensor":
        if not isinstance(holder, _HolderOwner):
            holder = _HolderOwner(holder)
        shape = tuple(shape)
        strides = tuple(strides)
        res = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            storage_offset=offset,
            dtype=_torch_dtype_of(dtype),
            layout=torch.strided,
            device=_WRAPPER_TENSORIMPL_DEVICE,
            requires_grad=False,
        )
        res._holder = holder
        res._ptr = ptr
        res._shape = shape
        res._mojo_strides = strides
        res._offset = offset
        res._dtype = dtype
        res._itemsize = dtype.size_in_bytes
        res._numel = math.prod(shape)
        res._device = device
        res._torch_device = _torch_device_of(device)
        res._is_contiguous = (
            _compute_contiguous(shape, strides) if contiguous is None else contiguous
        )
        return res

    @classmethod
    def _alloc(
        cls, shape: Sequence[int], dtype: DType, device: max.driver.Device
    ) -> "TorchMojoTensor":
        """A new contiguous uninitialized tensor (one device allocation)."""
        shape = tuple(shape)
        numel = math.prod(shape)
        holder, ptr = _alloc_with_recovery(device, numel * dtype.size_in_bytes)
        result = cls._make(
            holder,
            ptr,
            shape,
            _row_major_strides(shape),
            0,
            dtype,
            device,
            contiguous=True,
        )
        return result

    @classmethod
    def _view_of(
        cls,
        base: "TorchMojoTensor",
        shape: Sequence[int],
        strides: Sequence[int],
        offset: int,
        contiguous: bool | None = None,
    ) -> "TorchMojoTensor":
        """A zero-copy view: shares base's holder, new layout metadata.

        `offset` is absolute, in elements from the allocation start.
        `contiguous` skips the contiguity rescan when the caller knows it.
        """
        ptr = base._ptr + (offset - base._offset) * base._itemsize
        return cls._make(
            base._holder,
            ptr,
            shape,
            strides,
            offset,
            base._dtype,
            base._device,
            contiguous=contiguous,
        )

    @classmethod
    def _from_cpu(
        cls,
        cpu_tensor: torch.Tensor,
        device: max.driver.Device,
        *,
        non_blocking: bool = False,
    ) -> "TorchMojoTensor":
        """H2D: allocate and enqueue a copy from a CPU torch tensor."""
        from max.experimental.torch.torch import torch_dtype_to_max

        t = cpu_tensor.detach()
        if t.device.type != "cpu":
            # alloc_from_host below dereferences data_ptr() as HOST memory, so
            # handing it a device pointer faults inside the runtime instead of
            # raising anything Python can catch.  Callers must bounce through
            # the host; this turns a future mistake into a message.
            raise RuntimeError(
                "TorchMojoTensor._from_cpu requires a CPU source tensor, got "
                f"{t.device}; copy it to the host before uploading."
            )
        if not t.is_contiguous():
            t = t.contiguous()
        dtype = torch_dtype_to_max(t.dtype)
        nbytes = t.numel() * t.element_size()
        if nbytes == 0:
            # Nothing to transfer; skip alloc_from_host's full queue drain.
            return cls._alloc(tuple(t.shape), dtype, device)
        holder, ptr, transfer_owner = _holder_mod().alloc_from_host(
            _ctx_ptr(device), t.data_ptr(), nbytes
        )
        _record_h2d_source(device, transfer_owner, non_blocking)
        return cls._make(
            holder,
            ptr,
            tuple(t.shape),
            _row_major_strides(t.shape),
            0,
            dtype,
            device,
            contiguous=True,
        )

    def _to_cpu_tensor(self, *, non_blocking: bool = False) -> torch.Tensor:
        """D2H into MAX-owned pinned storage exposed as a CPU tensor.

        Host read: queued kernel launches must land first (call queue).

        With ``non_blocking=True`` on a GPU, the returned tensor aliases the
        pinned destination immediately and the caller must synchronize before
        consuming it, matching PyTorch's asynchronous accelerator-to-CPU
        contract. Blocking and CPU-device copies are ready on return.
        """
        from torch_mojo_backend.mojo_device import comm_fence, deferred_compile

        # A collective that wrote these bytes may still be flying on the comm
        # stream; the default stream this transfer rides is ordered after it
        # lazily (mojo_device/comm_fence.py), and this is a first use.
        comm_fence.fence_tensor(self)
        src = self if self._is_contiguous else self._materialize_contiguous()
        # Reading device bytes is a host read: every queued launch must have
        # executed before the transfer is enqueued -- INCLUDING the strided
        # materialization just queued above, whose output is what this reads.
        deferred_compile.drain()
        if src._numel == 0:
            return torch.empty(self._shape, dtype=max_dtype_to_torch(self._dtype))

        nbytes = src._numel * src._itemsize
        if not non_blocking or src._device == CPU():
            out = torch.empty(src._shape, dtype=max_dtype_to_torch(src._dtype))
            _holder_mod().copy_to_host(
                _ctx_ptr(src._device), src._ptr, out.data_ptr(), nbytes
            )
            _release_synchronized_h2d_sources(src._device)
            _release_synchronized_d2h_owners(src._device)
            return out

        owner, ptr = _holder_mod().copy_to_pinned_host(
            _ctx_ptr(src._device), src._ptr, nbytes
        )
        # Record and retain both ends before DLPack adoption, which can raise.
        # A non-contiguous input may make ``src`` a temporary whose holder must
        # remain alive until the non-owning DeviceBuffer copy has completed.
        _record_d2h_owner(src._device, (owner, src._holder))
        try:
            from torch_mojo_backend.mojo_device import dlpack

            return torch.from_dlpack(
                dlpack.make_capsule(owner, ptr, src._shape, src._dtype, CPU())
            )
        except Exception:
            src._device.default_stream.synchronize()
            _release_synchronized_h2d_sources(src._device)
            _release_synchronized_d2h_owners(src._device)
            raise

    def _materialize_contiguous(self) -> "TorchMojoTensor":
        """A new contiguous tensor with this tensor's (strided) contents."""
        rank = len(self._shape)
        if self._numel > 0 and rank <= 4:
            # Hot path (attention q/k/v transposes, expand): the rank-4
            # PermuteCopy gathers a strided source into a contiguous
            # destination with no destination index math and half the
            # coordinate math of the generic rank-8 CopyStrided.
            return _submit_prepared_into(_PermuteCopyExtension.prepare(self))
        out = TorchMojoTensor._alloc(self._shape, self._dtype, self._device)
        if self._numel > 0:
            _copy_strided_into(out, self)
        return out

    def _contig(self) -> "TorchMojoTensor":
        """self if already contiguous, else a materialized copy."""
        return self if self._is_contiguous else self._materialize_contiguous()

    def __dlpack__(self, *, stream: int | None = None, **_unused: object) -> object:
        """Export the device allocation as a "dltensor" capsule.

        torch's inherited `__dlpack__` would export the zero-byte meta
        storage; this override exports the real allocation described by the
        Python-side metadata. Non-contiguous tensors are materialized first,
        and the capsule pins the (materialized) tensor's holder. `stream` is
        ignored: producers and consumers share the device's default stream
        (the same assumption the eager kernels make).

        Publishing a raw pointer is a payload read from outside
        `__torch_dispatch__`, so it drains the call queue: the consumer must
        not see a buffer whose producing launches -- including the copy a
        strided export just queued -- are still waiting on a compile.
        """
        from torch_mojo_backend.mojo_device import comm_fence, deferred_compile, dlpack

        comm_fence.fence_tensor(self)  # same reason as _to_cpu_tensor's
        src = self._contig()
        deferred_compile.drain()
        return dlpack.make_capsule(
            src._holder, src._ptr, src._shape, src._dtype, src._device
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        from torch_mojo_backend.mojo_device import dlpack

        return dlpack.dlpack_device(self._device)

    def __coerce_same_metadata_as_tangent__(
        self, expected_meta: object, expected_type: type | None = None
    ) -> "TorchMojoTensor | None":
        """Accept mojo tensors as backward tangents under torch.compile.

        AOTAutograd guesses tangent types from fake tensors, which are plain
        `torch.Tensor`s, and rejects runtime tangents of unexpected types
        unless this hook coerces them. A mojo tensor behaves exactly like a
        plain tensor for dispatch purposes, so no conversion is needed.
        """
        if expected_type not in (None, torch.Tensor):
            return None
        return self

    def __reduce_ex__(self, protocol: int) -> object:
        """Pickle as a portable plain CPU tensor.

        torch.Tensor's reduce would pickle this subclass's `__dict__`, which
        holds the unpicklable Mojo `TensorHolder`. Checkpoints written from
        this device therefore serialize tensor values on the CPU and can be
        loaded without this backend being installed. Moving them back is the
        caller's `.to('mojo')` or `load_state_dict` onto a Mojo model.
        """
        if hasattr(self, "_holder"):
            return self._to_cpu_tensor().__reduce_ex__(protocol)
        return super().__reduce_ex__(protocol)

    def __repr__(self) -> str:
        if hasattr(self, "_holder"):
            return f"TorchMojoTensor({self._to_cpu_tensor()!r}, device='{self.device}')"
        return super().__repr__()

    # NOTE: shape/ndim/dim/size/stride/is_contiguous/numel/storage_offset are
    # deliberately NOT overridden. ``_make_wrapper_subclass`` records the same
    # sizes, strides and storage offset on the TensorImpl that ``_make`` stores
    # in the Python payload, and ``_rebind_payload`` resizes the TensorImpl
    # alongside a payload swap, so the C++ accessors are already authoritative
    # (they were only shadowed while the wrapper was a contiguous-only
    # ``_acc.create_empty_tensor``). Re-adding a Python override would break
    # torch.compile: dynamo models a tensor subclass through
    # ``TensorWithTFOverrideVariable``, which INLINES Python-level metadata
    # overrides -- so ``x.shape[0]`` would be traced as a read of the
    # ``_shape`` tuple, specializing the shape as a constant and, once dynamic,
    # minting a symbol unrelated to the tensor's size symbol (a graph input the
    # MAX backend cannot supply). It is also slower than the C++ accessor.

    @property
    def device(self) -> torch.device:
        # A plain attribute read so dynamo can trace `x.device` in compiled
        # functions (e.g. `torch.arange(T, device=idx.device)`).
        if hasattr(self, "_torch_device"):
            return self._torch_device
        return super().device

    def _set_data(self, value: torch.Tensor):
        """``tensor.data = other`` for a wrapper whose payload is in Python.

        The C++ setter reaches ``Variable::set_data``, which shallow-copies
        the *TensorImpl* metadata and stops there. For an ordinary backend
        that moves the storage pointer too and is the whole story; here the
        allocation lives in this object's ``_holder``/``_ptr``, so the C++
        path left every kernel reading the old buffer — silently, with no
        error and wrong numbers. (FSDP1 swaps a flat parameter between its
        sharded and unsharded buffers with exactly this assignment on every
        unshard and reshard.) Move the payload as well.
        """
        if (
            isinstance(value, TorchMojoTensor)
            and hasattr(self, "_holder")
            and hasattr(value, "_holder")
        ):
            if value._device != self._device:
                raise RuntimeError(
                    "tensor.data = other requires both tensors on the same mojo "
                    f"device (got {self.device} and {value.device})"
                )
            _rebind_payload_exact(self, value)
            return
        # torch's stub types `TensorBase.data` as a plain Tensor (the
        # instance-level property return), not the getset_descriptor it
        # actually is when accessed on the class -- real at runtime, just
        # unmodeled by the stub.
        torch._C.TensorBase.data.__set__(self, value)  # ty: ignore[unresolved-attribute]

    # Only the setter is overridden; reads keep going straight to the C++
    # getset descriptor so nothing about `x.data` changes for dynamo.
    data = property(
        torch._C.TensorBase.data.__get__,  # ty: ignore[unresolved-attribute]
        _set_data,
    )

    __torch_function__ = torch._C._disabled_torch_function_impl


class _PermuteCopyExtension(
    eager_kernels.MojoExtension[_TensorOutputSpec, "TorchMojoTensor"]
):
    """Stateless descriptor for rank-at-most-four strided materialization.

    It borrows data_movement_ops' source but not `MojoFileExtension`'s ABI:
    that base takes the specialization apart from `(op, arg_dtypes, flags)`
    call arguments, while this descriptor names its own defines. Deriving
    from it would shadow `make_defines` with the base's
    `make_canonical_defines` and never call the override.
    """

    MOJO_FILE: ClassVar[Path] = DataMovementExtension.MOJO_FILE

    @classmethod
    def make_defines(cls, tensor: MojoTensorLike) -> dict[str, bool | int | str]:
        return {
            "OP": "PermuteCopy",
            "DTYPE_ARG_0": tensor._dtype.name,
            "DTYPE_OUT": tensor._dtype.name,
        }

    @classmethod
    def expected_output_specs(cls, tensor: MojoTensorLike) -> _TensorOutputSpec:
        return _TensorOutputSpec(
            tuple(tensor._shape), tensor._dtype, _device_of(tensor)
        )

    @classmethod
    def extension_args(
        cls, out: TorchMojoTensor, tensor: MojoTensorLike
    ) -> tuple[object, ...]:
        rank = len(tensor._shape)
        pad = 4 - rank
        # extension_args is only ever called with the concrete wrapper (the
        # MojoTensorLike param type is for host-contract stand-ins in
        # make_defines/expected_output_specs, which don't need `_ptr`).
        mojo_tensor = cast(TorchMojoTensor, tensor)
        return (
            out._ptr,
            mojo_tensor._ptr,
            (1,) * pad + tuple(tensor._shape),
            (0,) * pad + tuple(tensor._mojo_strides),
            tensor._dtype.size_in_bytes,
            _ctx_ptr(_device_of(tensor)),
        )


_CPU_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)

_PAYLOAD_ATTRIBUTES = (
    "_holder",
    "_ptr",
    "_shape",
    "_mojo_strides",
    "_offset",
    "_dtype",
    "_itemsize",
    "_numel",
    "_device",
    "_torch_device",
    "_is_contiguous",
)


def _resize_tensorimpl(dst: TorchMojoTensor, shape: Sequence[int]):
    """Set dst's TensorImpl sizes (and grow its placeholder storage)."""
    torch.ops.aten.resize_.default.redispatch(
        _CPU_KEYSET, dst, shape, memory_format=None
    )


def _as_strided_tensorimpl(
    dst: TorchMojoTensor, shape: Sequence[int], strides: Sequence[int], offset: int
):
    """State dst's TensorImpl layout exactly. Metadata only, no data moved."""
    torch.ops.aten.as_strided_.default.redispatch(
        _CPU_KEYSET, dst, shape, strides, offset
    )


def _move_payload_attributes(dst: TorchMojoTensor, src: TorchMojoTensor):
    for name in _PAYLOAD_ATTRIBUTES:
        setattr(dst, name, getattr(src, name))
    # Any cached spec describes the old allocation or layout. Rebuild it on
    # the next spec operation instead of retaining stale pointer metadata.
    dst.__dict__.pop("_spec", None)


def _storage_extent(tensor: TorchMojoTensor) -> int:
    """Elements from the allocation start that `tensor`'s layout reaches."""
    if not all(tensor._shape):
        return tensor._offset
    return (
        tensor._offset
        + 1
        + sum(
            (size - 1) * stride
            for size, stride in zip(tensor._shape, tensor._mojo_strides)
        )
    )


def _rebind_payload(dst: TorchMojoTensor, src: TorchMojoTensor):
    """Move ``src``'s eager payload into ``dst`` without changing identity.

    Both the Python payload and the real TensorImpl must move together. A
    manual ``__dict__`` rebind makes direct properties look right while APIs
    such as ``torch.numel``, ``mT`` and ``flatten`` continue reading the stale
    TensorImpl.

    Swapping TensorImpl pointers is not valid inside an out kernel: the boxed
    dispatcher retains the original TensorImpl to enforce the schema's alias
    return, so a swap would make the call return the discarded wrapper. The
    CPU ``resize_`` kernel only updates TensorImpl/storage metadata here (the
    wrapper's dummy storage uses the Meta allocator); redispatch it explicitly
    to retain the original TensorImpl, then move the authoritative Mojo
    payload. No CPU tensor data is allocated or read.
    """
    _resize_tensorimpl(dst, src._shape)
    _move_payload_attributes(dst, src)


def _rebind_payload_exact(dst: TorchMojoTensor, src: TorchMojoTensor):
    """``_rebind_payload`` that also reproduces src's strides and offset.

    ``_rebind_payload`` moves the payload and resizes the TensorImpl, but
    ``resize_`` can only give contiguous strides at the offset the
    destination already had. That is enough for an ``out=`` kernel, and not
    enough for ``set_`` / ``tensor.data = view``, where the source is
    routinely a strided view at a non-zero offset (FSDP1 hands every
    original parameter a slice of the flat parameter this way). ``as_strided_``
    is a metadata-only composite kernel — it calls ``setStrided`` and touches
    no data — so redispatching it below this backend's Python kernels states
    the layout exactly.

    ``setStrided`` does bounds-check the layout against the wrapper's
    placeholder storage, which was sized for whatever the destination used to
    hold, so the resize that precedes it asks for the source's full extent
    rather than its element count. Nothing is allocated either way: the
    placeholder storage is on the Meta allocator and only its recorded byte
    count moves.
    """
    if src._mojo_strides == _row_major_strides(src._shape) and src._offset == 0:
        _rebind_payload(dst, src)
        return
    # Rewind to offset zero before growing: `resize_` sizes the storage from
    # the offset the destination already has, which after an earlier exact
    # rebind need not be zero and need not be large enough.
    _as_strided_tensorimpl(dst, (0,), (1,), 0)
    _resize_tensorimpl(dst, (_storage_extent(src),))
    _move_payload_attributes(dst, src)
    _as_strided_tensorimpl(dst, src._shape, src._mojo_strides, src._offset)


def _resize_payload(dst: TorchMojoTensor, shape: Sequence[int]):
    """Resize an eager out tensor and keep aliases when storage is sufficient.

    PyTorch resets a resized view to contiguous strides at its existing
    storage offset. Reuse that same allocation when the requested logical
    bytes fit; this preserves writes observed through another view such as
    ``base[:0]``. Otherwise use an ordinary context allocation. The final
    swap synchronizes Python metadata and TensorImpl metadata without changing
    ``dst``'s Python identity.
    """
    shape = tuple(shape)
    if shape == tuple(dst._shape):
        return

    required_bytes = math.prod(shape) * dst._itemsize
    allocation_bytes = int(dst._holder.get_nbytes())
    available_bytes = allocation_bytes - dst._offset * dst._itemsize
    if available_bytes < 0:
        available_bytes = 0
    if required_bytes <= available_bytes:
        replacement = TorchMojoTensor._make(
            dst._holder,
            dst._ptr,
            shape,
            _row_major_strides(shape),
            dst._offset,
            dst._dtype,
            dst._device,
            contiguous=True,
        )
    else:
        replacement = TorchMojoTensor._alloc(shape, dst._dtype, dst._device)
    _rebind_payload(dst, replacement)


def _copy_strided_enqueue(dst: TorchMojoTensor, src: TorchMojoTensor):
    """Queue the strided copy as an external call (tensor_holder is always
    loaded, so the item is always launch-ready; it only holds FIFO order).
    The queue holds raw pointers, so both tensors are handed over as the
    item's keep-alive: their buffers must outlive the launch."""
    from torch_mojo_backend.eager_kernels import call_queue as _cq

    holder = _holder_mod()
    args = (
        dst._ptr,
        src._ptr,
        _pad8(dst._shape, 1),
        _pad8(dst._mojo_strides, 0),
        _pad8(src._mojo_strides, 0),
        dst._itemsize,
        _ctx_ptr(dst._device),
    )
    _cq.external_call(holder.CopyStrided, args, keepalive=(dst, src))


def _copy_strided_into(dst: TorchMojoTensor, src: TorchMojoTensor):
    """dst[coords] = src[coords]; same shape and dtype, any strides.

    The shared materialize/copy primitive: powers .contiguous(), copy_ into
    views, and expand materialization (src strides may contain 0s).
    """
    from torch_mojo_backend.eager_kernels import call_queue as _cq

    if _cq.enabled() and _cq.active():
        # Hold FIFO position behind queued producers of src/dst.
        _copy_strided_enqueue(dst, src)
        return
    _holder_mod().CopyStrided(
        dst._ptr,
        src._ptr,
        _pad8(dst._shape, 1),
        _pad8(dst._mojo_strides, 0),
        _pad8(src._mojo_strides, 0),
        dst._itemsize,
        _ctx_ptr(dst._device),
    )


@functools.cache
def get_ordered_accelerators() -> list[max.driver.Device]:
    """Get accelerators ordered with GPUs first, then CPU last"""
    from torch_mojo_backend.torch_compile_backend.compiler import get_accelerators

    accelerators = list(get_accelerators())

    # Separate GPU and CPU accelerators
    gpu_accelerators = [acc for acc in accelerators if acc.label == "gpu"]
    cpu_accelerators = [acc for acc in accelerators if acc.label == "cpu"]

    # Order: GPUs first, then CPU last
    return gpu_accelerators + cpu_accelerators


def find_equivalent_max_device(device: torch.device) -> max.driver.Device:
    """Find the equivalent MAX device for a given torch device

    Device mapping:
    - mojo (no index) -> torch.mojo.current_device()
    - mojo:0 -> First GPU (or CPU if no GPUs)
    - mojo:1, mojo:2, ... -> Additional GPUs
    - mojo:<last_index> -> CPU device
    """
    ordered_accelerators = get_ordered_accelerators()

    if device.type == "mojo":
        # mojo with specific index
        if device.index is None:
            # Match PyTorch device semantics: an indexless backend device means
            # the backend's current device, not permanently device zero.
            return ordered_accelerators[torch_mojo_device_module.current_device()]
        else:
            if device.index < len(ordered_accelerators):
                return ordered_accelerators[device.index]
            else:
                raise ValueError(f"Invalid mojo index {device.index}")
    elif device.type == "cpu":
        # Find CPU accelerator (should be last in ordered list)
        for acc in reversed(ordered_accelerators):  # Check from the end
            if acc.label == "cpu":
                return acc
        # If no CPU found, return last accelerator as fallback
        return ordered_accelerators[-1]
    elif device.type in ("cuda", "hip"):
        # Find GPU accelerator (should be first in ordered list)
        # TODO: allow setting the default device index globally like with cuda
        gpu_index = device.index if device.index is not None else 0
        gpu_accelerators = [acc for acc in ordered_accelerators if acc.label == "gpu"]
        if gpu_index < len(gpu_accelerators):
            return gpu_accelerators[gpu_index]
        raise RuntimeError(f"GPU index {gpu_index} not available in MAX")
    else:
        raise NotImplementedError(f"Cannot convert {device.type} to MAX device")
