# Mojo extensions

## Compiled at first call

Eager-mode kernels are Mojo CPython extensions compiled on demand. The first
call into a specialization that is not in `__mojocache__` runs `mojo build`
inline, at the call site, and the call waits for it; every later call — and
every later process, as long as the sources and the toolchain are unchanged —
loads the cached `.so`. `TORCH_MOJO_BACKEND_TRACE` (on by default; `0`
silences it) prints a `[TRACE]` line when each variant build starts and
finishes, with its duration.

An earlier design overlapped those builds with graph discovery: launches were
queued behind a background compile pool so a cold workload compiled many
variants concurrently. It was removed in September 2026 — the FIFO ordering,
keep-alive, run-ahead budget and cross-thread rules it needed were too much
machinery for the cold-cache case — and may come back once the nominal,
warm-cache path is where it should be. The measurements it was built on are
kept in [docs/fast_eager_design.md](fast_eager_design.md) under "Compile
granularity".

## Stateless operation descriptors

Each operation is represented by a stateless `MojoExtension` subclass. Its
Mojo source path is a class attribute; per-invocation values never live on the
class or an instance. The descriptor provides three pieces of operation
semantics:

- `expected_output_specs(...)` validates metadata and describes every output;
- `make_defines(...)` returns the exact compile-time specialization;
- `extension_args(...)` converts inputs and preallocated outputs into the
  runtime arguments accepted by the extension.

Output descriptions include shape, dtype, device, and layout information
needed for allocation. Operations that alias an input, mutate in place, use an
`out=` argument, or return multiple tensors must describe those facts as well.

All state belongs to the shared infrastructure: compiled-module caches and
build locks. This avoids a race where simultaneous calls with different dtypes
overwrite a descriptor's "current" specialization.

## One compiled function per variant

Every specialized `.so` exposes one Python-callable function with a constant
name:

```python
extension.call(runtime_arguments...)
```

A Mojo source file may implement many operations, but its compile-time
definitions select only one of them for a particular `.so`. There is no Python
entry-point name in the cache identity and no attribute lookup for individual
operations. `tensor_holder` remains a special process-wide module because it
owns the shared Python tensor types.

## Compile-time definitions and cache identity

`make_defines(...)` returns a dictionary such as:

```python
{
    "OP": "AddSpec",
    "DTYPE_ARG_0": "float32",
    "DTYPE_ARG_1": "bfloat16",
    "DTYPE_OUT": "float32",
    "INPLACE": False,
}
```

The loader normalizes values and sorts entries by key. The same canonical
ordered representation generates both the compiler arguments and the defines
portion of the cache key, so dictionary insertion order cannot create duplicate
variants. Argument positions remain explicit: exchanging `DTYPE_ARG_0` and
`DTYPE_ARG_1` is a different specialization.

The compiler receives the normalized entries directly:

```text
-D DTYPE_ARG_0=float32
-D DTYPE_ARG_1=bfloat16
-D DTYPE_OUT=float32
-D INPLACE=0
-D OP=AddSpec
```

Every relevant flag is always present with a canonical default. Dtypes,
operation mode, output dtype, and implementation-selecting flags belong in the
definitions. Shapes, strides, pointers, scalar values, and device contexts are
runtime data unless they truly select different generated code. Consequently,
the same operation, dtypes, and flags reuse one `.so` across different shapes.

The full build identity also includes the Mojo source and dependency hashes,
the compiler/toolchain and host ABI identity, and the loader ABI version. The
canonical defines are serialized unambiguously and hashed for filenames; raw
`key=value-key=value` strings are not safe when string values contain
separators.

Each identity is immutable. A cache hit loads that exact `.so`; a miss builds
it. An unsupported dtype or flag reported by Mojo indicates a bug in Python's
definition mapping and is surfaced directly. There is no widening, dtype
escalation, fallback build, or launch-time retry.

One toolchain limit shapes the kernels rather than the build identity. ptxas
caps a kernel's *static* shared memory at 48 KiB (every target through CUDA
12.8; portable targets such as `sm_90` still in 13.x, only `sm_90a`-style
architecture-specific targets are exempt) and fails the whole `.so` on the
first kernel over the line, so every route needing more than that — the
wgmma/TMA GEMM kernels, flash attention, the tf32 and fp32 cores — stages its
tiles in dynamic (`extern`) shared memory, sized per launch by
`shared_mem_bytes` and opted into with
`FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES`. That window has never been
capped that way, so those kernels assemble under any ptxas and any target, and
nothing has to be probed or gated.

The ptxas itself comes from the `nvidia-cuda-nvcc-cu12` wheel (CUDA 12.8)
the package depends on: `torch_mojo_backend/_ptxas.py` sets
`MODULAR_NVPTX_COMPILER_PATH` to it when the variable is unset, before `max`
is imported, so the cubins load on any r570+ driver instead of requiring the
driver that MAX's own bundled ptxas assumes. An explicit setting wins.

Builds are protected by a per-identity file lock (`flock`) and written to a
temporary file before an atomic rename. Concurrent requests for one identity,
in this process or another, compile it once, and an interrupted compiler
cannot leave a partial file that looks valid.

## Into-style ABI

Python knows and allocates the outputs before the extension runs, so
extensions whose output metadata is inferred in Python use an Into-style ABI:

```python
extension.call(input_specs..., output_specs..., runtime_parameters...)
```

The call writes into the supplied outputs and returns no newly allocated tensor.
Multiple-output operations receive all output specs. In-place and `out=`
operations pass the appropriate existing tensors, and zero-sized outputs may
complete without launching a kernel.

The Python descriptor mirrors only the metadata-level eligibility checks needed
to prepare a call; the Mojo implementation still validates the runtime
arguments and raises on disagreement, which the Python caller turns into a
fallback to the generic path.
