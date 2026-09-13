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

Nothing above is compiled ahead of time except the runtime: the op body and
the kernel are each one `mojo build` that happens at the op's first call and
is then cached on disk (see "Three builds" below).

## Three builds

Everything is compiled on demand into `eager_kernels/__mojocache__/native/`
(`TORCH_MOJO_BACKEND_CACHE_DIR` moves the cache; it is contents-addressed so
several checkouts can share it) and each build is keyed by the hash of every
source it compiles in, the toolchain versions and its `-D` defines — so
touching `abi.mojo` invalidates every op extension, not just the backend.

| build | when | what it holds |
|---|---|---|
| C++ shim, Mojo backend | first `register_mojo_devices()` | the runtime, both fixed-size |
| one op extension per aten op | that op's first call | that op's body alone |
| one kernel family specialization | that kernel's first call | one (OP, dtypes, flags) kernel |

The backend library therefore does not grow with the number of ops: it holds
devices, streams, events, memory, the loader, the record ABI and the
registration list, and nothing else. Measured on one H100 node, 226 ops:
0.33 MB and 5.5 s of build, against 3.58 MB and 6.2 s when every op body was
linked into it — the size is the number that was growing, and the rest of the
build time is the compiler and the runtime modules, which are fixed.

The price is the first call of each op: one `mojo build` of 6–7 s, once per
op per source revision per machine (0.3–0.5 MB of cache each), and
milliseconds — a `dlopen` — in every later process. Building all 226 takes
about 22 minutes, which is why `prebuild_ops` exists.

`native.prebuild_ops()` compiles every op extension up front instead, for a
test suite or a CI image that would rather not pay a compile inside the first
call of each op (and, on a shared machine, not inside a GPU lock either).

Every `mojo build` subprocess runs with `MODULAR_HOME` on node-local disk
(`native.compiler_env`, `loader.mojo`'s `_compiler_env`; an explicit value
wins). That is where the *compiler* keeps its own module cache: its default
sits in `$HOME`, which on a cluster is one NFS directory shared by every
node, and concurrent compilers evict each other's entries there — "failed to
produce an archive for the module: No such file or directory". Node-local,
the first build on a machine pays about 25 s to fill it and nothing else
touches it.

The two shims of the first row:

**C++ shim (`native/csrc/`)** — the c10 objects torch only accepts as C++
classes, each forwarding to a Mojo function pointer:

| file | what |
|---|---|
| `shim_dispatch.cpp` | `tmb_library_impl` / `tmb_library_impl_lazy`: registers a Mojo function (or a resolver that produces one at the first call) as a boxed kernel. `MojoBoxedKernel` converts the IValue stack to `TmbValue` records (Scalar included, no heap boxing) and back, and caches the resolved kernel pointer. `tmb_call_op` calls any aten op from Mojo. |
| `shim_runtime.cpp` | allocator (`c10::Allocator` over Mojo alloc/free), `PrivateUse1HooksInterface`, the device guard (devices/streams/events), the Philox generator, `ProfilerStubs`, the tensor C API (`tmb_tensor_*`, `tmb_empty_strided`, `tmb_as_strided`). The current device and per-device current stream are C++ thread-locals (`tmb_current_device/stream`). |
| `shim_autocast.cpp` | `AutocastPrivateUse1` as one boxed fallback with a policy table filled from torch's own CUDA op lists. |

Three translation units compile in parallel: about 7 s wall cold.

**Mojo backend (`native/mojo/`)** — `mojo build backend.mojo --emit shared-lib`:

| file | what |
|---|---|
| `backend.mojo` | `tmb_native_init`: hooks table + the registration list (one `_group[register_x]` per ops file) |
| `registry.mojo` | `impl[op, "name"]`: registers the name behind a lazy trampoline in the backend, *or* is the selected op in that op's extension — see below |
| `abi.mojo` | `Value` records, tag constants, `T` (tensor view), result setters, `new_tensor` / `view_strided`, `unsupported()` |
| `device.mojo` | `Dev` per mojo index (accelerators, then the MAX CPU device), stream views, events (MAX events for ordering, vendor driver for query/timing), memory (`Buf` boxes behind DataPtr, `record_stream` fences), transfers |
| `vendor.mojo` | CUDA / HIP driver calls on MAX's raw streams |
| `loader.mojo` | on-demand builds of op extensions and kernel families: closure hash, cache lookup, `mojo build` in a subprocess under a flock, dlopen |
| `kernels.mojo` | `KernelCall`: defines + slots + owned specs for one kernel invocation |
| `ops_*.mojo` | the aten ops, and each file's `register_<group>` list |

## Op extensions: how an op body reaches the dispatcher

`backend.mojo` registers names, not implementations: `tmb_library_impl_lazy`
gives torch one generic trampoline per aten name, carrying the group file the
op lives in. At the op's first call the trampoline asks the loader for

```
mojo build native/mojo/ops_<group>.mojo --emit shared-lib -D TMB_OP=<aten name>
```

dlopens it, calls its `tmb_op_address` for the address of that op's boxed
entry (`abi.op_entry[op]`, exactly what a non-lazy registration would have
passed), and the shim stores it in the kernel object — so every later call is
the same direct call as before, with no added indirection.

One `impl[op, "name"](site)` line does both jobs, and `registry.TARGET_OP`
(the `TMB_OP` define) picks which:

* **backend library**, no define — register the name; `op` is named only in
  the branch the compiler drops, so the body is not elaborated;
* **that op's extension**, `TMB_OP=<name>` — hand back `op_address[op]()`.
  The other ~40 lines of the group's list compile to nothing, which is what
  keeps an extension to one op rather than a whole file.

A failed build is reported as a `RuntimeError` and is *not* remembered: the
next call tries again, so a compiler that died on a full disk is not fatal
for the process. A kernel that declines its inputs still raises
`NotImplementedError`, as before.

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
raises` in one `ops_<group>.mojo`, registered at the bottom of that same file
in `register_<group>` with `impl[op_x, "x.overload"](site)` (the name is a
compile-time parameter: that is what lets the op's extension select it).
A new group file needs three things: the `register_<group>` list, the
`tmb_op_address` export every group file ends with, and one
`_group[register_<group>](lib, "ops_<group>", prebuild)` line in
`backend.mojo`.

Read arguments with the `v_*` helpers by schema position, build outputs with
`new_tensor` / `new_like` / `view_strided`, set results with `ret_tensor`
(owned output), `ret_ref` (an input handed back: in-place ops),
`ret_tensor_list`, `ret_scalar_*`.

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
gloo group. Non-contiguous operands are staged through dense temporaries
allocated on the comm stream (so freeing them never fences the compute
stream); inputs are only recorded, outputs are copied back. The group
reports itself as its own backend (`_get_backend`, `supports_coalescing`)
because torch looks the backend up that way for `batch_isend_irecv` and
`_coalescing_manager`; the coalescing hooks map to one NCCL group so a
bidirectional exchange cannot deadlock. Inside such a group nothing is
submitted before `group_end`, so every copy-back and every `record_stream`
of a coalesced call is deferred to `_end_coalescing` (which also keeps the
staging buffers alive until then); a call that raises inside the block closes
the group and drops that deferred work, since torch's `_coalescing_manager`
has no `finally`. Each pg.mojo entry holds the backend mutex over its whole
body (`with Locked():`), since these calls come from Python outside the boxed
adapter.

Two limitations, both by design of the Future-based Work: `Work.is_completed()`
is true as soon as the collective is enqueued (completion is a stream event,
not a host-visible flag), and an asynchronous NCCL error surfaces at the next
call or `synchronize`, not through `Work.wait()`; `ncclCommGetAsyncError` is
exposed (`tmb_pg_async_error`) for a watchdog but nothing polls it. The
128-byte `ncclUniqueId` is passed by value the way the x86-64 SysV ABI lays it
out (16 words after the register arguments), so other architectures are
refused at construction.

## One base library for every accelerator

The Mojo base library (`libtmb_backend`) contains no device code and makes
no compile-time choice about the accelerator: `init_backend` asks MAX at run
time which api has devices (`cuda`, `hip`, `metal`, else CPU only) and
`vendor.mojo` resolves the CUDA or HIP driver entry points by name from that
answer. So it builds on a machine with no accelerator at all, and one build
serves NVIDIA, AMD and Apple machines; its cache key deliberately excludes
the accelerators (`toolchain_identity`), while kernel specializations, which
carry device code, are keyed with them (`kernel_identity`). Checked by
building the library on the cluster's login node (no GPU) and running the
runtime tests and the two-rank collective check on an H100 with that exact
file.

## Supported torch versions

Checked with the CPU wheels of torch 2.7.1, 2.8.0, 2.9.1, 2.10.0, 2.11.0,
2.12.1, 2.13.0 and 2.14.0 on an H100 (MAX 26.5, Mojo 1.0, Python 3.12):
device registration, ops with autograd against CPU, autocast, streams and
events, the profiler, a fused-AdamW training step, seeded RNG,
torch.compile through the mojo backend, and the two-rank collective check
with NCCL and with the Mojo collectives. Version-specific pieces: the shim
compiles as C++20 from torch 2.14 (its headers require it) and C++17
before; the guard method behind `torch.Stream.native_handle` exists from
2.11, so code that needs a stream's vendor handle uses
`torch.mojo.stream_native_handle` instead. torch 2.6 and older cannot
import the package: MAX 26.5's torch interop (`max.experimental.torch`,
used by the torch.compile backend) references a dtype added in 2.7.

## Triton

Triton kernels run on mojo tensors with only the CPU torch wheel and the
`triton` wheel installed: Triton compiles and launches through its own GPU
backend (bundled `ptxas`, `libcuda` or `libamdhip64` from the display
driver) and asks torch only which device and stream are current, through a
driver object. `torch_mojo_backend/triton_driver.py` provides that driver
for the mojo device (a CUDA and an untested HIP variant, chosen by the
accelerator MAX drives); `register_mojo_devices()` installs it the moment
`triton.runtime.driver` is imported, or at once if it already is
(`TORCH_MOJO_BACKEND_TRITON=0` opts out; `enable_triton()` does it by
hand). Launches go to the mojo current stream's vendor handle
(`torch.mojo.stream_native_handle`), so they are ordered with our kernels;
like `torch.cuda`, a launch targets the *current* device, so select it
(`with torch.mojo.device(i):`) for tensors on another GPU. The autotuner
and `triton.testing.do_bench` time through the mojo device's events. The
CUDA device ordinal is the mojo device index. Checked with the Triton
tutorial add kernel, an autotuned kernel, a second GPU, and Liger-Kernel's
RMSNorm forward/backward (a package written against `torch.cuda`; its own
`device.type == "cuda"` branches fall back to generic paths on "mojo", e.g.
one SM's worth of partial weight gradients, and its autocast decorators
bind to the CPU device type because it infers the device from
`torch.cuda.is_available()`). With a CUDA build of torch, the installed
driver still routes every Triton launch to the mojo current stream.

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
