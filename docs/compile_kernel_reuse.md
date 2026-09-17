# Reusing native Mojo code in torch.compile

`aten_functions.py` builds MAX graphs and also accepts MAX eager tensors.
The native device dispatches through `native/mojo/ops_*.mojo` to specialized
libraries in `eager_kernels/`. Reuse should happen below these adapters: share
the calculation or kernel launch, while each backend retains its own tensor
allocation, metadata, and stream handling.

## First implementation: acos

Both backends now use `mojo_kernels/unary_math.mojo::acos_value`:

- Native `elementwise_ops` calls the helper from its existing scalar/vector
  loops. Its launch geometry, alignment checks, and dtype dispatch stay intact.
- `mojo_kernels/unary.mojo` registers `tmb_acos` as an `ElementwiseUnaryOp`.
  `custom_mojo_ops.acos` calls it through `F.custom`, which supports both
  `TensorValue` and MAX's eager `Tensor`.
- `aten_acos` becomes a single call instead of duplicating the polynomial in
  Python. Half inputs compute in float32. The helper also restores PyTorch's
  NaN result outside [-1, 1], because Mojo 1.0's float32 primitive clamps them.

The graph adapter exposes SIMD math to MAX's elementwise compiler. This allows
one elementwise kernel for standalone acos and preserves fusion opportunities
with neighboring operations. It does not call the native boxed dispatcher or
load its `tmb_call` library from a graph. A graph expression with many nodes
can already fuse, so fewer nodes alone are not evidence of fewer GPU launches.

The native loader can import the sibling `mojo_kernels` package and includes
its modules in the source hash closure when imported. Editing shared math
therefore invalidates the native specialization, as well as rebuilding the
graph extension. Keep the helper itself independent of graph registration,
tensor ownership, and launch APIs.

## Migration plan

1. **Inventory and measure.** Start with ATen mappings that compose several
   MAX operations. Profile the actual compiled graph: record kernel count and
   device time, both alone and surrounded by typical neighboring operations.
   Compare against the native entry point on the same inputs. Prioritize
   repeated reductions and materialized intermediates.
2. **Elementwise math.** Extract small, dtype/width-generic helpers and expose
   them using MAX's elementwise traits, following acos. Candidates include
   `asinh`, `cosh`, and `sinh`, whose compile mappings currently compose MAX
   operations. Check numerical edge cases before reusing a native formula;
   sharing an approximation also shares its weaknesses. Reconcile the separate
   graph/native GELU-backward implementations in the same way. Simple direct
   MAX mappings such as `exp` need measurements to justify replacing them.
3. **Fused reductions and multi-output ops.** Start with `_log_softmax` and
   its backward, then the generic `native_layer_norm` path, including returned
   statistics. Extract an ordinary Mojo launch function accepting pointers,
   runtime dimensions/strides, and a caller-provided `DeviceContext`. Call it
   from both the native family entry and a MAX custom-op `execute` adapter.
   MAX allocates graph outputs and provides its execution context; the adapter
   must not allocate torch tensors, select a native stream, or synchronize.
   Retain existing fused MAX paths until measurements favor the shared kernel.
4. **Validate one op at a time.** Test `torch.compile(fullgraph=True)`, direct
   MAX eager execution, and public native-device calls. Include supported
   dtypes, domain boundaries, empty/scalar inputs, symbolic sizes, strided
   layouts, and gradients where applicable. Assert the ATen mapping was used.
   Measure device time and launches on real graphs, and check other targets
   with cross-compilation plus a review of host dispatch. Shapes remain runtime
   data, and native kernels retain independence from torch's vendor libraries.

Opaque custom launch ops can prevent fusion across their boundaries. Use them
when sharing a full reduction or attention kernel saves work; prefer the
elementwise traits when sharing pointwise math is sufficient. Existing native
`out=` and in-place handling belongs to the native adapter, not the graph op.

## Validation of the acos migration

With MAX 26.5, Mojo 1.0, and torch 2.11.0+cu128 on an H100, profiling a
357 x 789 float32 input in fresh processes recorded:

| Compiled expression | GPU kernel launches |
| --- | ---: |
| Previous Python polynomial for acos | 1 |
| Shared Mojo acos | 1 |
| Shared Mojo acos followed by adding 0.25 | 1 |

Thus acos removes duplicated code and fixes domain behavior while preserving
fusion; the old graph already fused its polynomial. These are launch-count
checks, not a controlled performance benchmark.

The targeted CPU/H100 suite passed 39 tests, with 2 GPU-float64 cases skipped
and 1 existing native-float64 expected failure. Coverage includes both MAX
value types, native device execution, half dtypes, domain errors, dynamic and
strided shapes, empty/scalar tensors, and the compiled backward pass. The
cache/source checks passed 179 tests; `ty check` and pre-commit passed too.

Cross-compilation to AMD gfx942 succeeded for float32/bfloat16 acos and exp.
The acos assembly changes for the NaN guard; both exp kernels are unchanged.
Native launch geometry is unchanged. AMD runtime performance is unmeasured.
