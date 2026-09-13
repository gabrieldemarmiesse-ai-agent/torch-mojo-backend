"""The `mojo` torch.distributed backend: a thin adapter over the Mojo
process-group core (native/mojo/pg.mojo).

Every collective runs on a per-device comm stream: Mojo makes that stream
wait for the caller's current stream, issues the NCCL / RCCL / mojoccl call
on it, and the adapter records the touched buffers on it (their release is
then fenced) and wraps a device-typed torch Future whose completion events
are recorded on the comm stream, so whoever waits on the returned Work (the
DDP reducer, a user) orders their own stream after the collective without
blocking the host — the contract ProcessGroupNCCL implements. CPU tensors
go to a private gloo group.
"""

from __future__ import annotations

import ctypes
import datetime
import functools
import os
import sys
import traceback
from collections.abc import Callable
from typing import cast

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
    PrefixStore,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    Work,
    _create_work_from_future,  # ty: ignore[unresolved-import] -- in torch 2.11's C module, absent from its stub
)

from torch_mojo_backend import native
from torch_mojo_backend.distributed import nccl
from torch_mojo_backend.native import device_module

_PRIVATEUSE1 = 20  # c10::DeviceType::PrivateUse1

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
        raise TypeError(
            f"dtype {dtype} is not supported by the mojo NCCL/RCCL backend"
        ) from None


def _nccl_red_op(op: ReduceOp | ReduceOp.RedOpType) -> int:
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
        f"ReduceOp {op} is not supported by the mojo NCCL/RCCL backend"
    )


def _loud(method: Callable[..., object]) -> Callable[..., object]:
    """torch swallows exceptions raised inside a Python ProcessGroup method
    called from the autograd thread; print the traceback before re-raising."""

    @functools.wraps(method)
    def wrapper(*args: object, **kwargs: object) -> object:
        try:
            return method(*args, **kwargs)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            raise

    return wrapper


class _Core:
    """The tmb_pg_* entries of the Mojo backend, taken from its vtable
    (functions of an imported Mojo module are not exported symbols)."""

    def __init__(self):
        lib = native.backend_lib()
        lib.tmb_pg_vtable.restype = ctypes.c_void_p
        table = ctypes.cast(lib.tmb_pg_vtable(), ctypes.POINTER(ctypes.c_void_p))
        i32, i64, vp, sz = (
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        # the order of pg.mojo's pg_vtable
        self.create = ctypes.CFUNCTYPE(vp, ctypes.c_char_p, i32, i32)(table[0])
        self.destroy = ctypes.CFUNCTYPE(None, vp)(table[1])
        self.version = ctypes.CFUNCTYPE(i32, vp)(table[2])
        self.unique_id = ctypes.CFUNCTYPE(i32, vp, vp)(table[3])
        self.init_device = ctypes.CFUNCTYPE(i32, vp, i32, vp)(table[4])
        self.comm_stream = ctypes.CFUNCTYPE(i64, vp, i32)(table[5])
        self.allreduce = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32)(table[6])
        self.broadcast = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32)(table[7])
        self.reduce = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32, i32)(table[8])
        self.allgather = ctypes.CFUNCTYPE(i32, vp, i32, vp, vp, sz, i32)(table[9])
        self.reduce_scatter = ctypes.CFUNCTYPE(i32, vp, i32, vp, vp, sz, i32, i32)(
            table[10]
        )
        self.send = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32)(table[11])
        self.recv = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32)(table[12])
        self.group_start = ctypes.CFUNCTYPE(i32, vp)(table[13])
        self.group_end = ctypes.CFUNCTYPE(i32, vp)(table[14])
        self.async_error = ctypes.CFUNCTYPE(i32, vp, i32)(table[15])
        self.abort = ctypes.CFUNCTYPE(i32, vp, i32)(table[16])
        self.synchronize_comm = ctypes.CFUNCTYPE(i32, vp, i32)(table[17])

    def check(self, rc: int, what: str):
        if rc != 0:
            raise RuntimeError(f"{what} failed: {native.last_error()}")


class MojoProcessGroup(dist.ProcessGroup):
    def __init__(
        self, store: Store, rank: int, world_size: int, timeout: datetime.timedelta
    ):
        super().__init__(rank, world_size)  # ty: ignore[missing-argument, invalid-argument-type] -- the 2-arg base ctor is the one a Python subclass can use
        self._store = store
        self._timeout = timeout
        # CPU tensors (object collectives, barriers on CPU groups) go to gloo.
        self._gloo = cast(
            dist.ProcessGroup,  # the gloo stub declares none of the collectives; the base class does
            dist.ProcessGroupGloo(
                PrefixStore("mojo-cpu-gloo", store), rank, world_size, timeout
            ),
        )
        self._core = _Core()
        self._path = nccl.library_path()
        self._handle = self._core.create(str(self._path).encode(), rank, world_size)
        if not self._handle:
            raise RuntimeError(
                f"could not load the collectives library: {native.last_error()}"
            )
        version = self._core.version(self._handle)
        name = "mojoccl" if nccl.uses_mojoccl() else nccl.vendor_name()
        if os.environ.get("TORCH_MOJO_BACKEND_TRACE", "1") != "0":
            print(
                f"[TRACE] collectives via {self._path} ({name} version {version})",
                file=sys.stderr,
                flush=True,
            )
        self._ready: dict[int, torch.Stream] = {}
        self._seq = 0
        self._group_name = ""

    # ---- plumbing ------------------------------------------------------------

    def getBackendName(self) -> str:
        return "mojo"

    def _set_group_name(self, name: str):
        self._group_name = name

    def getGroupName(self) -> str:
        return self._group_name

    def setGroupName(self, name: str):
        self._group_name = name

    def shutdown(self):
        if self._handle:
            self._core.destroy(self._handle)
            self._handle = None

    def abort(self):
        for index in list(self._ready):
            self._core.abort(self._handle, index)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    @staticmethod
    def _is_cpu(tensor: torch.Tensor) -> bool:
        if tensor.device.type == "cpu":
            return True
        if tensor.device.type != "mojo":
            raise ValueError(
                f"the mojo distributed backend handles mojo and cpu tensors, got {tensor.device}"
            )
        return tensor.device == device_module.cpu()

    @staticmethod
    def _index(tensor: torch.Tensor) -> int:
        return (
            tensor.device.index
            if tensor.device.index is not None
            else device_module.current_device()
        )

    def _ensure(self, index: int) -> torch.Stream:
        """One communicator per device, bootstrapped through the c10d store."""
        stream = self._ready.get(index)
        if stream is not None:
            return stream
        key = f"mojo-ccl-unique-id-{self._seq}-{index}"
        self._seq += 1
        if self.rank() == 0:
            buf = ctypes.create_string_buffer(nccl.NCCL_UNIQUE_ID_BYTES)
            self._core.check(self._core.unique_id(self._handle, buf), "ncclGetUniqueId")
            unique_id = buf.raw
            self._store.set(key, unique_id)  # ty: ignore[invalid-argument-type] -- the binding takes bytes too
        else:
            unique_id = bytes(self._store.get(key))
            if len(unique_id) != nccl.NCCL_UNIQUE_ID_BYTES:
                raise RuntimeError(f"bad unique id from store key {key}")
        buf = ctypes.create_string_buffer(unique_id, nccl.NCCL_UNIQUE_ID_BYTES)
        self._core.check(
            self._core.init_device(self._handle, index, buf), "ncclCommInitRank"
        )
        stream_id = self._core.comm_stream(self._handle, index)
        stream = torch.Stream(
            stream_id=stream_id, device_index=index, device_type=_PRIVATEUSE1
        )
        self._ready[index] = stream
        return stream

    def _work(self, index: int, result: list[torch.Tensor]) -> Work:
        """A Work whose wait() makes the waiter's stream follow the comm stream."""
        future: torch.futures.Future[list[torch.Tensor]] = torch.futures.Future(
            devices=[torch.device("mojo", index)]
        )
        with device_module.stream(self._ready[index]):
            future.set_result(result)
        return _create_work_from_future(future)

    def _staged(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.is_contiguous() else tensor.contiguous()

    def _finish(self, index: int, *pairs: tuple[torch.Tensor, torch.Tensor]):
        """After the collective: copy staged results back on the comm stream
        and record every touched buffer on it so its release is fenced."""
        comm = self._ready[index]
        with device_module.stream(comm):
            for tensor, staged in pairs:
                if staged is not tensor:
                    tensor.copy_(staged)
                    staged.record_stream(comm)
                tensor.record_stream(comm)

    @staticmethod
    def _one_per_rank(tensors: list[torch.Tensor]):
        if len(tensors) != 1:
            raise ValueError(
                "the mojo backend takes exactly one tensor per rank per call"
            )

    def _group(self):
        self._core.check(self._core.group_start(self._handle), "ncclGroupStart")

    def _ungroup(self):
        self._core.check(self._core.group_end(self._handle), "ncclGroupEnd")

    # ---- collectives -----------------------------------------------------------

    @_loud
    def allreduce(
        self, tensors: list[torch.Tensor], opts: AllreduceOptions = AllreduceOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.allreduce(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        index = self._index(tensor)
        self._ensure(index)
        staged = self._staged(tensor)
        self._core.check(
            self._core.allreduce(
                self._handle,
                index,
                staged.data_ptr(),
                staged.numel(),
                _nccl_dtype(staged.dtype),
                _nccl_red_op(opts.reduceOp),
            ),
            "ncclAllReduce",
        )
        self._finish(index, (tensor, staged))
        return self._work(index, tensors)

    @_loud
    def allreduce_coalesced(
        self,
        tensors: list[torch.Tensor],
        opts: AllreduceCoalescedOptions = AllreduceCoalescedOptions(),
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.allreduce_coalesced(tensors, opts)
        index = self._index(tensors[0])
        self._ensure(index)
        staged = [self._staged(t) for t in tensors]
        self._group()
        for s in staged:
            self._core.check(
                self._core.allreduce(
                    self._handle,
                    index,
                    s.data_ptr(),
                    s.numel(),
                    _nccl_dtype(s.dtype),
                    _nccl_red_op(opts.reduceOp),
                ),
                "ncclAllReduce",
            )
        self._ungroup()
        self._finish(index, *zip(tensors, staged))
        return self._work(index, tensors)

    @_loud
    def broadcast(
        self, tensors: list[torch.Tensor], opts: BroadcastOptions = BroadcastOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.broadcast(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        index = self._index(tensor)
        self._ensure(index)
        staged = self._staged(tensor)
        root = opts.rootRank * len(tensors) + opts.rootTensor
        self._core.check(
            self._core.broadcast(
                self._handle,
                index,
                staged.data_ptr(),
                staged.numel(),
                _nccl_dtype(staged.dtype),
                root,
            ),
            "ncclBroadcast",
        )
        self._finish(index, (tensor, staged))
        return self._work(index, tensors)

    @_loud
    def reduce(
        self, tensors: list[torch.Tensor], opts: ReduceOptions = ReduceOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.reduce(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        index = self._index(tensor)
        self._ensure(index)
        staged = self._staged(tensor)
        self._core.check(
            self._core.reduce(
                self._handle,
                index,
                staged.data_ptr(),
                staged.numel(),
                _nccl_dtype(staged.dtype),
                _nccl_red_op(opts.reduceOp),
                opts.rootRank,
            ),
            "ncclReduce",
        )
        self._finish(index, (tensor, staged))
        return self._work(index, tensors)

    def _allgather_flat(
        self, index: int, output: torch.Tensor, input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Base all-gather: output (world * input.numel()) <- every rank's input."""
        if output.numel() != input.numel() * self.size():
            raise ValueError(
                "all_gather output must hold world_size copies of the input"
            )
        if output.dtype != input.dtype:
            raise ValueError("all_gather output dtype must match the input")
        staged_in = self._staged(input)
        staged_out = self._staged(output)
        self._core.check(
            self._core.allgather(
                self._handle,
                index,
                staged_in.data_ptr(),
                staged_out.data_ptr(),
                staged_in.numel(),
                _nccl_dtype(input.dtype),
            ),
            "ncclAllGather",
        )
        return staged_in, staged_out

    @_loud
    def _allgather_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensor):
            return self._gloo._allgather_base(output_tensor, input_tensor, opts)
        index = self._index(input_tensor)
        self._ensure(index)
        staged_in, staged_out = self._allgather_flat(index, output_tensor, input_tensor)
        self._finish(index, (input_tensor, staged_in), (output_tensor, staged_out))
        return self._work(index, [output_tensor])

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
        input = input_tensors[0]
        outputs = output_tensors[0]
        if len(outputs) != self.size():
            raise ValueError("all_gather needs world_size output tensors")
        index = self._index(input)
        comm = self._ensure(index)
        flat = torch.empty(
            self.size() * input.numel(), dtype=input.dtype, device=input.device
        )
        staged_in, _ = self._allgather_flat(index, flat, input)
        with device_module.stream(comm):
            n = input.numel()
            for r, out in enumerate(outputs):
                out.copy_(flat[r * n : (r + 1) * n].view(out.shape))
                out.record_stream(comm)
            flat.record_stream(comm)
        self._finish(index, (input, staged_in))
        return self._work(index, outputs)

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
        index = self._index(input_tensors[0])
        self._ensure(index)
        self._group()
        pairs = [
            self._allgather_flat(index, o, i)
            for o, i in zip(output_tensors, input_tensors)
        ]
        self._ungroup()
        self._finish(
            index,
            *[(i, si) for i, (si, _) in zip(input_tensors, pairs)],
            *[(o, so) for o, (_, so) in zip(output_tensors, pairs)],
        )
        return self._work(index, output_tensors)

    def _reduce_scatter_flat(
        self, index: int, output: torch.Tensor, input: torch.Tensor, op: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input.numel() != output.numel() * self.size():
            raise ValueError(
                "reduce_scatter input must hold world_size chunks of the output"
            )
        if output.dtype != input.dtype:
            raise ValueError("reduce_scatter output dtype must match the input")
        staged_in = self._staged(input)
        staged_out = self._staged(output)
        self._core.check(
            self._core.reduce_scatter(
                self._handle,
                index,
                staged_in.data_ptr(),
                staged_out.data_ptr(),
                staged_out.numel(),
                _nccl_dtype(input.dtype),
                op,
            ),
            "ncclReduceScatter",
        )
        return staged_in, staged_out

    @_loud
    def _reduce_scatter_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(input_tensor):
            return self._gloo._reduce_scatter_base(output_tensor, input_tensor, opts)
        index = self._index(input_tensor)
        self._ensure(index)
        staged_in, staged_out = self._reduce_scatter_flat(
            index, output_tensor, input_tensor, _nccl_red_op(opts.reduceOp)
        )
        self._finish(index, (input_tensor, staged_in), (output_tensor, staged_out))
        return self._work(index, [output_tensor])

    @_loud
    def reduce_scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.reduce_scatter(output_tensors, input_tensors, opts)
        self._one_per_rank(output_tensors)
        output = output_tensors[0]
        inputs = input_tensors[0]
        if len(inputs) != self.size():
            raise ValueError("reduce_scatter needs world_size input tensors")
        index = self._index(output)
        comm = self._ensure(index)
        flat = torch.cat([t.reshape(-1) for t in inputs])
        _, staged_out = self._reduce_scatter_flat(
            index, output, flat, _nccl_red_op(opts.reduceOp)
        )
        with device_module.stream(comm):
            flat.record_stream(comm)
        self._finish(index, (output, staged_out))
        return self._work(index, [output])

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
        index = self._index(output_tensors[0])
        self._ensure(index)
        op = _nccl_red_op(opts.reduceOp)
        self._group()
        pairs = [
            self._reduce_scatter_flat(index, o, i, op)
            for o, i in zip(output_tensors, input_tensors)
        ]
        self._ungroup()
        self._finish(
            index,
            *[(i, si) for i, (si, _) in zip(input_tensors, pairs)],
            *[(o, so) for o, (_, so) in zip(output_tensors, pairs)],
        )
        return self._work(index, output_tensors)

    def _sendrecv_chunks(
        self,
        index: int,
        output: torch.Tensor,
        input: torch.Tensor,
        out_sizes: list[int],
        in_sizes: list[int],
    ):
        dtype = _nccl_dtype(input.dtype)
        itemsize = input.element_size()
        self._group()
        in_off = out_off = 0
        for peer in range(self.size()):
            n_in, n_out = in_sizes[peer], out_sizes[peer]
            if n_in:
                self._core.check(
                    self._core.send(
                        self._handle,
                        index,
                        input.data_ptr() + in_off * itemsize,
                        n_in,
                        dtype,
                        peer,
                    ),
                    "ncclSend",
                )
            if n_out:
                self._core.check(
                    self._core.recv(
                        self._handle,
                        index,
                        output.data_ptr() + out_off * itemsize,
                        n_out,
                        dtype,
                        peer,
                    ),
                    "ncclRecv",
                )
            in_off += n_in
            out_off += n_out
        self._ungroup()

    @_loud
    def alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work:
        if self._is_cpu(input_tensor):
            return self._gloo.alltoall_base(
                output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts
            )
        index = self._index(input_tensor)
        self._ensure(index)
        world = self.size()
        if not input_split_sizes:
            input_split_sizes = [input_tensor.shape[0] // world] * world
        if not output_split_sizes:
            output_split_sizes = [output_tensor.shape[0] // world] * world
        row_in = input_tensor[0].numel() if input_tensor.dim() > 0 else 1
        row_out = output_tensor[0].numel() if output_tensor.dim() > 0 else 1
        staged_in = self._staged(input_tensor)
        staged_out = self._staged(output_tensor)
        self._sendrecv_chunks(
            index,
            staged_out,
            staged_in,
            [s * row_out for s in output_split_sizes],
            [s * row_in for s in input_split_sizes],
        )
        self._finish(index, (input_tensor, staged_in), (output_tensor, staged_out))
        return self._work(index, [output_tensor])

    @_loud
    def alltoall(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.alltoall(output_tensors, input_tensors, opts)
        index = self._index(input_tensors[0])
        self._ensure(index)
        staged_in = [self._staged(t) for t in input_tensors]
        staged_out = [self._staged(t) for t in output_tensors]
        self._group()
        for peer, (i, o) in enumerate(zip(staged_in, staged_out)):
            if i.numel():
                self._core.check(
                    self._core.send(
                        self._handle,
                        index,
                        i.data_ptr(),
                        i.numel(),
                        _nccl_dtype(i.dtype),
                        peer,
                    ),
                    "ncclSend",
                )
            if o.numel():
                self._core.check(
                    self._core.recv(
                        self._handle,
                        index,
                        o.data_ptr(),
                        o.numel(),
                        _nccl_dtype(o.dtype),
                        peer,
                    ),
                    "ncclRecv",
                )
        self._ungroup()
        self._finish(
            index, *zip(input_tensors, staged_in), *zip(output_tensors, staged_out)
        )
        return self._work(index, output_tensors)

    @_loud
    def gather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: GatherOptions = GatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.gather(output_tensors, input_tensors, opts)
        self._one_per_rank(input_tensors)
        input = input_tensors[0]
        index = self._index(input)
        comm = self._ensure(index)
        staged_in = self._staged(input)
        dtype = _nccl_dtype(input.dtype)
        self._group()
        if self.rank() == opts.rootRank:
            outputs = output_tensors[0]
            staged_out = [self._staged(t) for t in outputs]
            for peer, o in enumerate(staged_out):
                if peer == self.rank():
                    continue
                self._core.check(
                    self._core.recv(
                        self._handle, index, o.data_ptr(), o.numel(), dtype, peer
                    ),
                    "ncclRecv",
                )
            self._ungroup()
            with device_module.stream(comm):
                staged_out[self.rank()].copy_(staged_in)
            self._finish(index, (input, staged_in), *zip(outputs, staged_out))
            return self._work(index, outputs)
        self._core.check(
            self._core.send(
                self._handle,
                index,
                staged_in.data_ptr(),
                staged_in.numel(),
                dtype,
                opts.rootRank,
            ),
            "ncclSend",
        )
        self._ungroup()
        self._finish(index, (input, staged_in))
        return self._work(index, input_tensors)

    @_loud
    def scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ScatterOptions = ScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.scatter(output_tensors, input_tensors, opts)
        self._one_per_rank(output_tensors)
        output = output_tensors[0]
        index = self._index(output)
        comm = self._ensure(index)
        staged_out = self._staged(output)
        dtype = _nccl_dtype(output.dtype)
        self._group()
        if self.rank() == opts.rootRank:
            inputs = input_tensors[0]
            staged_in = [self._staged(t) for t in inputs]
            for peer, i in enumerate(staged_in):
                if peer == self.rank():
                    continue
                self._core.check(
                    self._core.send(
                        self._handle, index, i.data_ptr(), i.numel(), dtype, peer
                    ),
                    "ncclSend",
                )
            self._ungroup()
            with device_module.stream(comm):
                staged_out.copy_(staged_in[self.rank()])
            self._finish(index, (output, staged_out), *zip(inputs, staged_in))
            return self._work(index, output_tensors)
        self._core.check(
            self._core.recv(
                self._handle,
                index,
                staged_out.data_ptr(),
                staged_out.numel(),
                dtype,
                opts.rootRank,
            ),
            "ncclRecv",
        )
        self._ungroup()
        self._finish(index, (output, staged_out))
        return self._work(index, output_tensors)

    @_loud
    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.send(tensors, dst_rank, tag)
        index = self._index(tensors[0])
        self._ensure(index)
        staged = [self._staged(t) for t in tensors]
        self._group()
        for s in staged:
            self._core.check(
                self._core.send(
                    self._handle,
                    index,
                    s.data_ptr(),
                    s.numel(),
                    _nccl_dtype(s.dtype),
                    dst_rank,
                ),
                "ncclSend",
            )
        self._ungroup()
        self._finish(index, *zip(tensors, staged))
        return self._work(index, tensors)

    @_loud
    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.recv(tensors, src_rank, tag)
        index = self._index(tensors[0])
        self._ensure(index)
        staged = [self._staged(t) for t in tensors]
        self._group()
        for s in staged:
            self._core.check(
                self._core.recv(
                    self._handle,
                    index,
                    s.data_ptr(),
                    s.numel(),
                    _nccl_dtype(s.dtype),
                    src_rank,
                ),
                "ncclRecv",
            )
        self._ungroup()
        self._finish(index, *zip(tensors, staged))
        return self._work(index, tensors)

    @_loud
    def barrier(self, opts: BarrierOptions = BarrierOptions()) -> Work:
        """Every rank finishes its device work (all streams, comm included),
        then a host rendezvous over gloo."""
        for index in self._ready:
            torch.accelerator.synchronize(index)
            self._core.check(
                self._core.synchronize_comm(self._handle, index),
                "comm stream synchronize",
            )
        return self._gloo.barrier(opts)


def create_mojo_process_group(
    store: Store, rank: int, world_size: int, timeout: datetime.timedelta
) -> MojoProcessGroup:
    """The factory `torch.distributed.Backend.register_backend` stores."""
    return MojoProcessGroup(store, rank, world_size, timeout)


_HSA_LEVEL = "ROCR_VISIBLE_DEVICES"
_RUNTIME_LEVEL = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def _visible_entries(var: str) -> list[str] | None:
    """The comma list in `var`, or None when it is unset or empty."""
    visible = os.environ.get(var)
    if visible is None:
        return None
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    return entries or None


def _keep_local_rank_entry(var: str, entries: list[str], rank_index: int):
    if len(entries) <= 1:
        return  # already pinned, e.g. one srun task per GPU
    if rank_index >= len(entries):
        raise RuntimeError(
            f"LOCAL_RANK={rank_index} but {var}={os.environ[var]!r} lists only "
            f"{len(entries)} devices"
        )
    os.environ[var] = entries[rank_index]


def use_local_rank_gpu():
    """Pin this torchrun worker to its GPU via the vendor's visibility variable.

    Call as early as possible — before any mojo tensor is created and before
    MAX enumerates devices (enumeration is cached per process). With exactly
    one visible GPU per rank, ``mojo``/``mojo:0`` is always the right device,
    the phantom ``privateuseone:0`` TensorImpl index is always truthful, and
    each process binds a single CUDA context / HIP device.

    SLURM (and other launchers) often pre-set the visibility variable to the
    whole allocation — ``CUDA_VISIBLE_DEVICES=0,...,7`` on NVIDIA,
    ``ROCR_VISIBLE_DEVICES=0,...,3`` on AMD; in that case each rank keeps
    only its LOCAL_RANK-th entry. A single already-pinned entry is left
    alone (one srun task per GPU pins that way). When no variable is set at
    all, both the CUDA and HIP ones are set to LOCAL_RANK, whichever runtime
    turns out to be present.

    Only one level is ever narrowed. When ``ROCR_VISIBLE_DEVICES`` is
    present it is the level that gets the rank's entry, and any runtime-level
    list (SLURM's gres plugin exports ``CUDA_VISIBLE_DEVICES`` next to it by
    default) is rewritten to ``"0"``, the only index that exists in a
    one-entry HSA set — narrowing both levels by rank would compose to no
    visible GPU at all for every rank but 0.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        rank_index = int(local_rank)
    except ValueError:
        raise RuntimeError(f"LOCAL_RANK={local_rank!r} is not an integer") from None
    hsa = _visible_entries(_HSA_LEVEL)
    if hsa is not None:
        _keep_local_rank_entry(_HSA_LEVEL, hsa, rank_index)
        for var in _RUNTIME_LEVEL:
            if _visible_entries(var) is not None:
                os.environ[var] = "0"
        return
    pinned = False
    for var in _RUNTIME_LEVEL:
        entries = _visible_entries(var)
        if entries is None:
            continue
        _keep_local_rank_entry(var, entries, rank_index)
        pinned = True
    if not pinned:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank_index)
        os.environ["HIP_VISIBLE_DEVICES"] = str(rank_index)
