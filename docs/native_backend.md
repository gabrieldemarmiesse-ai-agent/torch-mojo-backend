# The native mojo device

The `mojo` torch device is a real PrivateUse1 backend: tensors are ordinary
`torch.Tensor`s whose storage is MAX device memory, and every aten op the
device supports is a Mojo function that torch's C++ dispatcher calls
directly. No Python runs on the op path.

```
torch.add(a, b)  ->  dispatcher  ->  MojoBoxedKernel (C++, native/csrc)
                                         |  TmbValue records
                                         v
                                  op_add_tensor (Mojo, native/mojo/ops_*.mojo)
                                         |  KernelCall
                                         v
                                  logic_ops.so::tmb_call  (built on first use)
```

## The two shims

Both are compiled once per torch/toolchain version, at the first
`register_mojo_devices()`, into `eager_kernels/__mojocache__/native/`
(`TORCH_MOJO_BACKEND_CACHE_DIR` moves the cache; it is contents-addressed so
several checkouts can share it).

**C++ shim (`native/csrc/`)** — the c10 objects torch only accepts as C++
classes, each forwarding to a Mojo function pointer:

| file | what |
|---|---|
| `shim_dispatch.cpp` | `tmb_library_impl`: registers a Mojo function as a boxed kernel. `MojoBoxedKernel` converts the IValue stack to `TmbValue` records (Scalar included, no heap boxing) and back. `tmb_call_op` calls any aten op from Mojo. |
| `shim_runtime.cpp` | allocator (`c10::Allocator` over Mojo alloc/free), `PrivateUse1HooksInterface`, the device guard (devices/streams/events), the Philox generator, `ProfilerStubs`, the tensor C API (`tmb_tensor_*`, `tmb_empty_strided`, `tmb_as_strided`). The current device and per-device current stream are C++ thread-locals (`tmb_current_device/stream`). |
| `shim_autocast.cpp` | `AutocastPrivateUse1` as one boxed fallback with a policy table filled from torch's own CUDA op lists. |

Three translation units compile in parallel: about 7 s wall cold.

**Mojo backend (`native/mojo/`)** — `mojo build backend.mojo --emit shared-lib`:

| file | what |
|---|---|
| `backend.mojo` | `tmb_native_init`: hooks table + one `tmb_library_impl` per op |
| `abi.mojo` | `Value` records, tag constants, `T` (tensor view), result setters, `new_tensor` / `view_strided`, `unsupported()` |
| `device.mojo` | `Dev` per mojo index (accelerators, then the MAX CPU device), stream views, events (MAX events for ordering, vendor driver for query/timing), memory (`Buf` boxes behind DataPtr, `record_stream` fences), transfers |
| `vendor.mojo` | CUDA / HIP driver calls on MAX's raw streams |
| `loader.mojo` | on-demand family builds: closure hash, cache lookup, `mojo build` in a subprocess under a flock, dlopen |
| `kernels.mojo` | `KernelCall`: defines + slots + owned specs for one kernel invocation |
| `ops_*.mojo` | the aten ops |

## Kernel families: the C entry

Every family under `eager_kernels/<family>/` exports one C function per
specialization build:

```mojo
@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32
```

`argv` is an array of 64-bit **slots** (`op_utils.Arg`): ints and pointers
are the value, floats their bit pattern (`_raw_f64`), a tuple the address of
an `[len, e0, ...]` Int array, a TensorSpec its address. The `comptime if
_op_on["Name"]()` ladder picks the one op compiled into the build and calls
its `_spec_dispatcherN[go, "Name"]`; a raised `Error` comes back as
`(rc=1, message)`.

## Writing an op

An op is `def op_x(args: Values, n_args: Int, rets: Values, n_rets: Int)
raises`, registered in `backend.mojo` with `_impl(lib, "x.overload",
op_address[op_x]())`. Read arguments with the `v_*` helpers by schema
position, build outputs with `new_tensor` / `new_like` / `view_strided`, set
results with `ret_tensor` (owned output), `ret_ref` (an input handed back:
in-place ops), `ret_tensor_list`, `ret_scalar_*`.

To run a kernel:

```mojo
var ctx = ctx_for(t.device)          # the device's CURRENT stream
var cp = ctx_ptr(ctx)
var call = KernelCall("logic_ops", "AddSpec")
call.arg_dtype(0, a.dtype)
call.arg_dtype(1, b.dtype)
call.out_dtype(dst.dtype)
call.spec(a.spec(cp)); call.spec(b.spec(cp)); call.spec(dst.spec(cp))
call.run()
_ = ctx
```

**Lifetimes.** Mojo destroys a value right after its last use. Never take a
local's address, hand it over as an Int and let the local die before the
call: the slot then reads freed memory (this produced the first segfault of
this backend). `KernelCall` owns specs, tuples and slots until `run()`
returns; keep the `DeviceContext` alive with `_ = ctx` after the call.

**Declining.** `unsupported("why")` raises with a prefix the entry turns into
rc 2 = `NotImplementedError` in Python; any other `Error` is a
`RuntimeError`. A kernel that declines its inputs raises inside the family;
the op decides whether to try another route or propagate.

**Errors from C.** Every `tmb_*` call returning `int32_t` goes through
`check(rc, "what")`, which appends the shim's thread-local message.

**Streams.** Ops launch on the device's current stream (`ctx_for`), so
`with torch.Stream(...)` really moves execution. Memory is allocated on the
current stream; a tensor used by another stream gets `record_stream`ed by
torch (`recordDataPtrOnStream`), which the backend turns into an event the
owner stream waits on before the buffer is released.

**Threads.** The shim's recursive mutex serializes every call into Mojo, so
ops need no locking of their own; the autograd engine's thread and the main
thread interleave at op granularity.

## Test support

`native.op_counting(True)`, `native.op_count("aten::add.Tensor")` count
boxed-kernel calls per op (the `CallChecker` in `torch_mojo_backend/testing.py`
uses them to assert that an op ran natively).

## Streams and events

`torch.Stream(device="mojo")`, `torch.Event`, `torch.accelerator.current_stream
/ set_stream / synchronize` and `torch.mojo.stream(s)` are torch's own generic
objects driven by the shim's device guard. A stream is a MAX stream of the
device's base context (`DeviceContext.create_stream`, then a `select_stream`
view kernels launch on); events pair a MAX event (ordering: record / wait /
synchronize on every backend) with a CUDA / HIP driver event on the same raw
stream for `query()` and `elapsed_time()`. The MAX CPU device has one stream;
its events time with the host clock.

## Distributed

`torch.distributed.init_process_group(backend="mojo")` registers
`MojoProcessGroup` (distributed/process_group.py), a thin adapter over
`native/mojo/pg.mojo`: one communicator and one dedicated comm stream per
device; every collective makes the comm stream wait for the caller's current
stream, issues the NCCL / RCCL / mojoccl call on it (the three share the NCCL
C ABI; `TORCH_MOJO_BACKEND_CCL=mojo` picks the in-repo Mojo collectives), and
the returned Work is a device-typed torch Future completed while the comm
stream is current, so `wait()` orders the waiter's stream after the collective
without blocking the host. Touched buffers are `record_stream`ed on the comm
stream; the allocator fences their release on it. CPU tensors go to a private
gloo group.

## Profiling

The shim registers torch's PrivateUse1 `ProfilerStubs` over the backend's
timed events, so the legacy profiler reports device time per op:

```python
with torch.autograd.profiler.profile(use_device="mojo") as prof:
    ...
prof.key_averages().table(sort_by="self_device_time_total")   # "Self MOJO" columns
```

`torch.profiler.profile(activities=[CPU, PrivateUse1])` records the CPU-side
op timeline and exports Chrome traces; device kernel rows need the Kineto
PrivateUse1 plugin API that only exists in torch >= 2.12 (with a CUDA torch
wheel that matches the driver, CUPTI already captures MAX kernels).
