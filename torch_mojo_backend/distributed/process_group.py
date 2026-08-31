"""A c10d ProcessGroup for the mojo eager device, backed by NCCL.

Design (see docs/distributed.md for the full story):

- Pure-Python ``torch.distributed.ProcessGroup`` subclass. In torch 2.11 a
  Python PG *replaces* the whole process group (distributed_c10d.py:2193-2198),
  so no gloo backend can be composed in by ``init_process_group`` — object
  collectives and ``barrier()`` therefore hand us **CPU** tensors
  (``_get_object_coll_device`` falls back to "cpu" when ``_device_types`` is
  empty). Every override dispatches on the tensor's device type and delegates
  CPU tensors to a private ``ProcessGroupGloo``.

- NCCL runs on the SAME CUDA stream as all mojo compute (the MAX device's
  default stream, obtained via ``DeviceStream.native_stream_handle``). That
  makes ordering free: mojo kernels, NCCL kernels, and the stream-ordered
  frees of MAX buffers all ride one FIFO stream, so no events, no host syncs,
  and no keep-alives are needed for correctness. The one obligation is to
  **drain the host-side kernel-call queue first**: a queued-but-unlaunched
  producer kernel is invisible to the stream (docs/kernel_call_queue.md).

- Work objects wrap an already-completed ``torch.futures.Future`` holding the
  output tensors. Never pass ``devices=`` to that Future: with a device list
  it routes through the stub PythonDeviceGuard whose ``deviceCount() == 1``
  and performs an out-of-bounds write for device indices >= 1. A plain CPU
  future does zero device bookkeeping, which is exactly right here because
  stream order already guarantees device-side completion for any consumer on
  the same stream. ``wait()`` on it is a host-side no-op by design.

- The DDP Reducer calls ``allreduce`` through the C++ trampoline and then
  ``Work.get_future()`` (default_comm_hooks.cpp) — both supported by
  ``_create_work_from_future``. Gradient scaling (/world_size) happens inside
  the Reducer before the collective, so ``allreduce`` here is a plain SUM.

One process drives exactly one GPU (torchrun layout). Set
``CUDA_VISIBLE_DEVICES`` per rank *before* MAX enumerates devices — see
``torch_mojo_backend.distributed.use_local_rank_gpu()``.
"""

import datetime
import os
import sys
import threading
import traceback
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    _create_work_from_future,
)
from torch.distributed import PrefixStore, Store, Work

from torch_mojo_backend.distributed import nccl

_NCCL_DTYPE_OF: dict[torch.dtype, int] = {
    torch.int8: nccl.NCCL_INT8,
    torch.uint8: nccl.NCCL_UINT8,
    torch.bool: nccl.NCCL_UINT8,
    torch.int32: nccl.NCCL_INT32,
    torch.uint32: nccl.NCCL_UINT32,
    torch.int64: nccl.NCCL_INT64,
    torch.uint64: nccl.NCCL_UINT64,
    torch.float16: nccl.NCCL_FLOAT16,
    torch.float32: nccl.NCCL_FLOAT32,
    torch.float64: nccl.NCCL_FLOAT64,
    torch.bfloat16: nccl.NCCL_BFLOAT16,
}

# torch.bool reduces as uint8: SUM/MAX behave as logical OR, MIN as AND —
# the same convention ProcessGroupNCCL uses.


def _nccl_dtype(dtype: torch.dtype) -> int:
    try:
        return _NCCL_DTYPE_OF[dtype]
    except KeyError:
        raise TypeError(f"dtype {dtype} is not supported by the mojo NCCL backend")


def _nccl_red_op(op: ReduceOp) -> int:
    if op == ReduceOp.SUM:
        return nccl.NCCL_SUM
    if op == ReduceOp.PRODUCT:
        return nccl.NCCL_PROD
    if op == ReduceOp.MAX:
        return nccl.NCCL_MAX
    if op == ReduceOp.MIN:
        return nccl.NCCL_MIN
    if op == ReduceOp.AVG:
        return nccl.NCCL_AVG
    raise NotImplementedError(
        f"ReduceOp {op} is not supported by the mojo NCCL backend"
    )


def _completed_work(result: list[torch.Tensor]) -> Work:
    future = torch.futures.Future()  # no devices= — see module docstring
    future.set_result(result)
    return _create_work_from_future(future)


def _loud(fn: Callable) -> Callable:
    """Print the traceback before propagating.

    An exception that escapes into the autograd engine through a C++ backward
    hook on this backend can terminate the process without any Python
    traceback (see mojo_device/aten_ops/autograd_preflight.py). Printing here
    guarantees the root cause is visible even in that worst case.
    """

    def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            print(
                f"[torch-mojo-backend] rank {self.rank()}: error in "
                f"MojoProcessGroup.{fn.__name__}:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            sys.stderr.flush()
            raise

    wrapper.__name__ = fn.__name__
    return wrapper


class MojoProcessGroup(dist.ProcessGroup):
    """NCCL-backed process group for ``mojo`` tensors, gloo for CPU tensors."""

    def __init__(
        self, store: Store, rank: int, world_size: int, timeout: datetime.timedelta
    ) -> None:
        # The 2-arg base init: the 3-arg (store, rank, size) overload is a
        # pybind FACTORY init, and factory inits cannot construct the
        # PyProcessGroup trampoline alias a Python subclass needs — it fails
        # with "returned holder-wrapped instance is not an alias instance".
        # Every in-tree Python PG (test_c10d_pypg, multi_threaded_pg) does
        # this too; the C++-side store stays null and we keep it here instead.
        super().__init__(rank, world_size)
        self._store = store
        self._timeout = timeout
        self._group_name = ""
        # NCCL communicators keyed by mojo device index, created lazily at the
        # first device collective (a collective, blocking rendezvous — every
        # rank reaches it in the same order because collectives are SPMD).
        self._comms: dict[int, nccl.NcclComm] = {}
        self._streams: dict[int, int] = {}
        self._comm_seq = 0
        self._comm_lock = threading.Lock()
        self._device_current = threading.local()
        # Private CPU backend: torch will not compose gloo around a Python PG
        # (see module docstring), so CPU tensors are our job too.
        self._gloo = dist.ProcessGroupGloo(
            PrefixStore("mojo-cpu-gloo", store), rank, world_size, timeout
        )

    # -- plumbing ------------------------------------------------------------

    def getBackendName(self) -> str:
        return "mojo"

    def getGroupName(self) -> str:
        # The C++ accessor raises while no C++ backend is registered, so keep
        # the name Python-side (same workaround as test_c10d_pypg.py).
        return self._group_name

    def setGroupName(self, name: str) -> None:
        self._group_name = name

    def _set_group_name(self, name: str) -> None:
        self._group_name = name

    def shutdown(self) -> None:
        for comm in self._comms.values():
            comm.destroy()
        self._comms.clear()

    def abort(self) -> None:
        for comm in self._comms.values():
            comm.abort()
        self._comms.clear()

    def _ensure_device_current(self, ordinal: int) -> None:
        # NCCL resolves the target GPU from the thread-current CUDA context,
        # and DDP calls us from the autograd thread while init usually runs on
        # the main thread — so re-assert per thread, once.
        if getattr(self._device_current, "ordinal", None) != ordinal:
            nccl.set_current_cuda_device(ordinal)
            self._device_current.ordinal = ordinal

    def _device_state(self, tensor: torch.Tensor) -> tuple[nccl.NcclComm, int]:
        """The (communicator, raw CUstream) serving this mojo tensor's GPU."""
        from torch_mojo_backend.mojo_device import cuda_peer
        from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
            find_equivalent_max_device,
        )

        index = tensor.device.index
        if index is None:
            index = torch.mojo.current_device()
        ordinal = cuda_peer.device_ordinal(tensor._ptr)
        if ordinal is None:
            raise RuntimeError(
                "could not resolve the CUDA ordinal of a mojo tensor; the mojo "
                "NCCL backend currently supports NVIDIA GPUs only"
            )
        self._ensure_device_current(ordinal)
        with self._comm_lock:
            comm = self._comms.get(index)
            if comm is None:
                comm = self._init_comm(index, ordinal)
                self._comms[index] = comm
                max_device = find_equivalent_max_device(torch.device("mojo", index))
                self._streams[index] = max_device.default_stream.native_stream_handle
        return comm, self._streams[index]

    def _init_comm(self, index: int, ordinal: int) -> nccl.NcclComm:
        key = f"nccl-uid-{self._comm_seq}"
        self._comm_seq += 1
        if self.rank() == 0:
            unique_id = nccl.get_unique_id()
            self._store.set(key, unique_id)
        else:
            unique_id = bytes(self._store.get(key))
            if len(unique_id) != nccl.NCCL_UNIQUE_ID_BYTES:
                raise RuntimeError(f"bad NCCL unique id from store key {key}")
        return nccl.NcclComm.init_rank(self.size(), unique_id, self.rank())

    def _drained(self) -> None:
        """Launch every queued mojo kernel so the stream sees all producers."""
        from torch_mojo_backend.mojo_device import deferred_compile

        deferred_compile.drain()

    def _dense(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.is_contiguous() else tensor.contiguous()

    def _is_cpu(self, tensor: torch.Tensor) -> bool:
        if tensor.device.type == "cpu":
            return True
        if tensor.device.type != "mojo":
            raise RuntimeError(
                f"the mojo process group cannot handle tensors on {tensor.device}"
            )
        if getattr(tensor, "_device", None) is not None and tensor._device.api not in (
            "cuda",
            "hip",
        ):
            # mojo:<last> is MAX's host device (device_count counts it too);
            # NCCL cannot touch host memory and gloo cannot touch mojo
            # wrappers, so refuse loudly rather than guess.
            raise NotImplementedError(
                "collectives on the mojo host device are not supported; use a "
                "GPU mojo device or a plain CPU tensor"
            )
        return False

    # -- collectives ---------------------------------------------------------

    @_loud
    def allreduce(
        self, tensors: list[torch.Tensor], opts: AllreduceOptions = AllreduceOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.allreduce(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        op = _nccl_red_op(opts.reduceOp)
        comm, stream = self._device_state(tensor)
        staged = self._dense(tensor)
        self._drained()
        comm.all_reduce(
            staged._ptr,
            staged._ptr,
            staged.numel(),
            _nccl_dtype(staged.dtype),
            op,
            stream,
        )
        if staged is not tensor:
            tensor.copy_(staged)
        return _completed_work(tensors)

    @_loud
    def allreduce_coalesced(
        self,
        tensors: list[torch.Tensor],
        opts: AllreduceCoalescedOptions = AllreduceCoalescedOptions(),
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.allreduce_coalesced(tensors, opts)
        op = _nccl_red_op(opts.reduceOp)
        comm, stream = self._device_state(tensors[0])
        staged = [self._dense(t) for t in tensors]
        self._drained()
        nccl.group_start()
        for s in staged:
            comm.all_reduce(s._ptr, s._ptr, s.numel(), _nccl_dtype(s.dtype), op, stream)
        nccl.group_end()
        for original, s in zip(tensors, staged):
            if s is not original:
                original.copy_(s)
        return _completed_work(tensors)

    @_loud
    def broadcast(
        self, tensors: list[torch.Tensor], opts: BroadcastOptions = BroadcastOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.broadcast(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        comm, stream = self._device_state(tensor)
        staged = self._dense(tensor)
        self._drained()
        comm.broadcast(
            staged._ptr,
            staged._ptr,
            staged.numel(),
            _nccl_dtype(staged.dtype),
            int(opts.rootRank),
            stream,
        )
        if staged is not tensor:
            tensor.copy_(staged)
        return _completed_work(tensors)

    @_loud
    def reduce(
        self, tensors: list[torch.Tensor], opts: ReduceOptions = ReduceOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.reduce(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        comm, stream = self._device_state(tensor)
        staged = self._dense(tensor)
        self._drained()
        comm.reduce(
            staged._ptr,
            staged._ptr,
            staged.numel(),
            _nccl_dtype(staged.dtype),
            _nccl_red_op(opts.reduceOp),
            int(opts.rootRank),
            stream,
        )
        if staged is not tensor:
            tensor.copy_(staged)
        return _completed_work(tensors)

    @_loud
    def allgather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.allgather(output_tensors, input_tensors, opts)
        self._one_per_rank(input_tensors)
        source = self._dense(input_tensors[0])
        outputs = output_tensors[0]
        flat = torch.empty(
            self.size() * source.numel(), dtype=source.dtype, device=source.device
        )
        comm, stream = self._device_state(source)
        self._drained()
        comm.all_gather(
            source._ptr, flat._ptr, source.numel(), _nccl_dtype(source.dtype), stream
        )
        for peer, out in enumerate(outputs):
            chunk = flat.narrow(0, peer * source.numel(), source.numel())
            out.copy_(chunk.view(out.shape))
        return _completed_work(outputs)

    @_loud
    def _allgather_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensor):
            return self._gloo._allgather_base(output_tensor, input_tensor, opts)
        source = self._dense(input_tensor)
        if output_tensor.numel() != source.numel() * self.size():
            raise ValueError("all_gather_into_tensor output has the wrong size")
        dest = self._dense(output_tensor)
        comm, stream = self._device_state(source)
        self._drained()
        comm.all_gather(
            source._ptr, dest._ptr, source.numel(), _nccl_dtype(source.dtype), stream
        )
        if dest is not output_tensor:
            output_tensor.copy_(dest)
        return _completed_work([output_tensor])

    @_loud
    def allgather_into_tensor_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.allgather_into_tensor_coalesced(
                output_tensors, input_tensors, opts
            )
        comm, stream = self._device_state(input_tensors[0])
        staged_in = [self._dense(t) for t in input_tensors]
        staged_out = [self._dense(t) for t in output_tensors]
        self._drained()
        nccl.group_start()
        for source, dest in zip(staged_in, staged_out):
            comm.all_gather(
                source._ptr,
                dest._ptr,
                source.numel(),
                _nccl_dtype(source.dtype),
                stream,
            )
        nccl.group_end()
        for original, s in zip(output_tensors, staged_out):
            if s is not original:
                original.copy_(s)
        return _completed_work(output_tensors)

    @_loud
    def reduce_scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.reduce_scatter(output_tensors, input_tensors, opts)
        output = output_tensors[0]
        inputs = input_tensors[0]
        if len(inputs) != self.size():
            raise ValueError("reduce_scatter expects world_size input tensors")
        count = output.numel()
        flat = torch.empty(
            self.size() * count, dtype=output.dtype, device=output.device
        )
        for peer, chunk in enumerate(inputs):
            flat.narrow(0, peer * count, count).copy_(chunk.reshape(-1))
        dest = self._dense(output)
        comm, stream = self._device_state(dest)
        self._drained()
        comm.reduce_scatter(
            flat._ptr,
            dest._ptr,
            count,
            _nccl_dtype(dest.dtype),
            _nccl_red_op(opts.reduceOp),
            stream,
        )
        if dest is not output:
            output.copy_(dest)
        return _completed_work(output_tensors)

    @_loud
    def _reduce_scatter_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensor):
            return self._gloo._reduce_scatter_base(output_tensor, input_tensor, opts)
        source = self._dense(input_tensor)
        dest = self._dense(output_tensor)
        if source.numel() != dest.numel() * self.size():
            raise ValueError("reduce_scatter_tensor input has the wrong size")
        comm, stream = self._device_state(dest)
        self._drained()
        comm.reduce_scatter(
            source._ptr,
            dest._ptr,
            dest.numel(),
            _nccl_dtype(dest.dtype),
            _nccl_red_op(opts.reduceOp),
            stream,
        )
        if dest is not output_tensor:
            output_tensor.copy_(dest)
        return _completed_work([output_tensor])

    @_loud
    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.reduce_scatter_tensor_coalesced(
                output_tensors, input_tensors, opts
            )
        comm, stream = self._device_state(output_tensors[0])
        op = _nccl_red_op(opts.reduceOp)
        staged_in = [self._dense(t) for t in input_tensors]
        staged_out = [self._dense(t) for t in output_tensors]
        self._drained()
        nccl.group_start()
        for source, dest in zip(staged_in, staged_out):
            comm.reduce_scatter(
                source._ptr,
                dest._ptr,
                dest.numel(),
                _nccl_dtype(dest.dtype),
                op,
                stream,
            )
        nccl.group_end()
        for original, s in zip(output_tensors, staged_out):
            if s is not original:
                original.copy_(s)
        return _completed_work(output_tensors)

    @_loud
    def alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work:
        if self._is_cpu(output_tensor):
            return self._gloo.alltoall_base(
                output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts
            )
        world = self.size()
        source = self._dense(input_tensor)
        dest = self._dense(output_tensor)
        row = source.numel() // max(source.shape[0], 1) if source.dim() else 1
        if not input_split_sizes:
            input_split_sizes = [source.shape[0] // world] * world
        if not output_split_sizes:
            output_split_sizes = [dest.shape[0] // world] * world
        dtype = _nccl_dtype(source.dtype)
        comm, stream = self._device_state(source)
        self._drained()
        nccl.group_start()
        send_offset = 0
        recv_offset = 0
        itemsize = source.element_size()
        for peer in range(world):
            send_count = input_split_sizes[peer] * row
            recv_count = output_split_sizes[peer] * row
            comm.send(
                source._ptr + send_offset * itemsize, send_count, dtype, peer, stream
            )
            comm.recv(
                dest._ptr + recv_offset * itemsize, recv_count, dtype, peer, stream
            )
            send_offset += send_count
            recv_offset += recv_count
        nccl.group_end()
        if dest is not output_tensor:
            output_tensor.copy_(dest)
        return _completed_work([output_tensor])

    @_loud
    def alltoall(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.alltoall(output_tensors, input_tensors, opts)
        comm, stream = self._device_state(input_tensors[0])
        staged_in = [self._dense(t) for t in input_tensors]
        staged_out = [self._dense(t) for t in output_tensors]
        self._drained()
        nccl.group_start()
        for peer in range(self.size()):
            source = staged_in[peer]
            dest = staged_out[peer]
            comm.send(
                source._ptr, source.numel(), _nccl_dtype(source.dtype), peer, stream
            )
            comm.recv(dest._ptr, dest.numel(), _nccl_dtype(dest.dtype), peer, stream)
        nccl.group_end()
        for original, s in zip(output_tensors, staged_out):
            if s is not original:
                original.copy_(s)
        return _completed_work(output_tensors)

    @_loud
    def gather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: GatherOptions = GatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.gather(output_tensors, input_tensors, opts)
        root = int(opts.rootRank)
        source = self._dense(input_tensors[0])
        dtype = _nccl_dtype(source.dtype)
        comm, stream = self._device_state(source)
        result: list[torch.Tensor] = []
        if self.rank() == root:
            outputs = output_tensors[0]
            # The root's own contribution is a local device copy, not a
            # self-send: same convention as ProcessGroupNCCL.
            outputs[root].copy_(source.view(outputs[root].shape))
            staged_out = [self._dense(t) for t in outputs]
            self._drained()
            nccl.group_start()
            for peer, dest in enumerate(staged_out):
                if peer != root:
                    comm.recv(dest._ptr, dest.numel(), dtype, peer, stream)
            nccl.group_end()
            for original, s in zip(outputs, staged_out):
                if s is not original:
                    original.copy_(s)
            result = outputs
        else:
            self._drained()
            nccl.group_start()
            comm.send(source._ptr, source.numel(), dtype, root, stream)
            nccl.group_end()
        return _completed_work(result)

    @_loud
    def scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ScatterOptions = ScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.scatter(output_tensors, input_tensors, opts)
        root = int(opts.rootRank)
        dest = self._dense(output_tensors[0])
        dtype = _nccl_dtype(dest.dtype)
        comm, stream = self._device_state(dest)
        if self.rank() == root:
            sources = [self._dense(t) for t in input_tensors[0]]
            output_tensors[0].copy_(sources[root].view(output_tensors[0].shape))
            self._drained()
            nccl.group_start()
            for peer, source in enumerate(sources):
                if peer != root:
                    comm.send(source._ptr, source.numel(), dtype, peer, stream)
            nccl.group_end()
        else:
            self._drained()
            nccl.group_start()
            comm.recv(dest._ptr, dest.numel(), dtype, root, stream)
            nccl.group_end()
            if dest is not output_tensors[0]:
                output_tensors[0].copy_(dest)
        return _completed_work([output_tensors[0]])

    @_loud
    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.send(tensors, dst_rank, tag)
        self._one_per_rank(tensors)
        source = self._dense(tensors[0])
        comm, stream = self._device_state(source)
        self._drained()
        comm.send(
            source._ptr, source.numel(), _nccl_dtype(source.dtype), dst_rank, stream
        )
        return _completed_work(tensors)

    @_loud
    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.recv(tensors, src_rank, tag)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        dest = self._dense(tensor)
        comm, stream = self._device_state(dest)
        self._drained()
        comm.recv(dest._ptr, dest.numel(), _nccl_dtype(dest.dtype), src_rank, stream)
        if dest is not tensor:
            tensor.copy_(dest)
        return _completed_work(tensors)

    @_loud
    def barrier(self, opts: BarrierOptions = BarrierOptions()) -> Work:
        # Complete this rank's device work first, then rendezvous over gloo.
        # This gives barrier() the "everything before me is done everywhere"
        # meaning users expect from the CUDA backend.
        if self._comms:
            torch.mojo.synchronize()
        return self._gloo.barrier(opts)

    def _one_per_rank(self, tensors: list[torch.Tensor]) -> None:
        if len(tensors) != 1:
            raise NotImplementedError(
                "the mojo backend runs one process per GPU; multi-device-per-rank "
                f"collectives are not supported (got {len(tensors)} tensors)"
            )


def create_mojo_process_group(
    store: Store, rank: int, world_size: int, timeout: datetime.timedelta
) -> MojoProcessGroup:
    """The creator function registered with torch.distributed.Backend."""
    return MojoProcessGroup(store, rank, world_size, timeout)


def use_local_rank_gpu() -> None:
    """Pin this torchrun worker to its GPU via CUDA_VISIBLE_DEVICES.

    Call as early as possible — before any mojo tensor is created and before
    MAX enumerates devices (enumeration is cached per process). With exactly
    one visible GPU per rank, ``mojo``/``mojo:0`` is always the right device,
    the phantom ``privateuseone:0`` TensorImpl index is always truthful, and
    each process binds a single CUDA context.

    SLURM (and other launchers) often pre-set CUDA_VISIBLE_DEVICES to the
    whole allocation ("0,1,...,7"); in that case each rank keeps only its
    LOCAL_RANK-th entry. A single already-pinned entry is left alone.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    rank_index = int(local_rank)
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        visible = os.environ.get(var)
        if visible is None:
            os.environ[var] = local_rank
            continue
        entries = [entry for entry in visible.split(",") if entry]
        if len(entries) <= 1:
            continue  # already pinned (or nothing to slice)
        if rank_index >= len(entries):
            raise RuntimeError(
                f"LOCAL_RANK={rank_index} but {var}={visible!r} lists only "
                f"{len(entries)} devices"
            )
        os.environ[var] = entries[rank_index]
