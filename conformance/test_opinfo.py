"""Every OpInfo in PyTorch's `op_db`, executed on the mojo device and compared
against the same call on the CPU.

What comes from OpInfo rather than from us:

* the sample inputs (`sample_inputs_func`) and, with `--reference-inputs`, the
  larger `reference_inputs_func` corpus of awkward shapes, non-contiguous
  layouts, broadcasts and scalar overloads;
* which of a sample's tensors belong on the device and which ATen requires on
  the CPU, read back from where the sample generator itself puts them when it
  targets a device (`_opinfo_placements`);
* the dtypes each operator is expected to support, per `op.dtypesIfCUDA` —
  the accelerator set, not the CPU one;
* the accuracy bar: `op.precisionOverride` when the operator declares one,
  otherwise `torch.testing.assert_close`'s dtype defaults, applied through
  `TestCase.assertEqual`.  We never invent a tolerance;
* the corner cases: `error_inputs_func`, i.e. the inputs PyTorch asserts must
  RAISE, and the exception type and message it requires.

An operator we have not implemented raises NotImplementedError from the
dispatch layer.  That is reported as a distinct outcome (skip with a
"not implemented" reason), never as a pass and never as a wrong answer,
because the point of this suite is to tell those three apart.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase, run_tests

# The dtypes worth exercising on an accelerator backend.  Deliberately not the
# full OpInfo set: float64 is absent on some GPUs we target and complex is not
# implemented at all, so including them would report a backend-wide gap once
# per operator instead of once.
_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list | tuple):
        return type(value)(_to_cpu(v) for v in value)
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    return value


def _not_implemented(exc: BaseException) -> bool:
    """True when the backend declined the op rather than got it wrong."""
    if isinstance(exc, NotImplementedError):
        return True
    text = str(exc).lower()
    return (
        "not implemented" in text
        or "no fast implementation" in text
        # Some error_inputs_func corpora build a deliberately wrong-dtype
        # tensor (complex, float64) purely to provoke the error under test.
        # MAX doesn't support those dtypes at all, so construction fails
        # before the operator under test is ever reached -- a backend
        # limitation, not a wrong error-handling result.
        or "unsupported torch dtype" in text
    )


def _tensor_specs(sample: Any) -> list[tuple[torch.Size, torch.dtype, torch.device]]:
    """Shape, dtype and device of every tensor in `sample`, in the order
    `SampleInput.transform` visits them.

    `transform` is also what moves the sample to the device below, so walking
    with it is what makes two walks line up position by position.
    """
    specs: list[tuple[torch.Size, torch.dtype, torch.device]] = []

    def record(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            specs.append((value.shape, value.dtype, value.device))
        return value

    sample.transform(record)
    return specs


def _opinfo_placements(
    op: Any, dtype: torch.dtype
) -> list[list[tuple[torch.Size, torch.dtype, torch.device]]] | None:
    """Where OpInfo itself puts each sample-input tensor when it builds this
    operator's samples for a device -- one entry per tensor, per sample.

    Not every tensor in a sample is data.  Some are metadata that ATen
    requires on the CPU whatever device the operand lives on:
    `tensor_split`'s `tensor_indices_or_sections` is checked outright
    (`aten/src/ATen/native/TensorShape.cpp`, `tensor_split`), and `narrow`'s
    `start`, `logsumexp`'s `dim` and advanced indexing's index tensors are
    conventionally left there too.  `sample_inputs_func` already encodes this,
    by building such arguments with a plain `torch.tensor(...)` while the
    operand goes through `make_tensor(device=device)`.  So asking a generator
    to target a device and reading back where each tensor landed IS the
    per-argument device rule, straight from OpInfo -- no table of ours to
    maintain and, crucially, nothing to learn from an exception after the
    fact.  Moving every tensor found in a sample and then recognising ATen's
    complaint by its wording was the previous answer here; it reported a
    harness artifact as an operator skip, and silently became a spurious
    failure the moment upstream reworded the message.

    The device asked for is `meta`: it allocates nothing and runs no kernel,
    so it cannot fail for the reason samples are built on the CPU in the first
    place (a backend missing an op that `make_tensor` reaches), and it does
    not need the backend under test to exist at all.  Only `.device` is read
    from the result.

    None means the placement is not observable and the caller should move
    everything, as this harness always did: the generator raised on meta
    (data-dependent construction -- `nonzero`, `bincount`, `searchsorted` and
    ~100 others), or yielded a different number of samples there (`sort`,
    `argsort` and `to` branch on the device), so position no longer identifies
    an argument.
    """
    rng_state = torch.get_rng_state()
    try:
        samples = list(op.sample_inputs("meta", dtype, requires_grad=False))
    except Exception:  # noqa: BLE001 - any failure here just means "unobservable"
        return None
    finally:
        # Meta construction fills nothing, so today no generator draws from the
        # default RNG here (checked across op_db: sample values are byte-identical
        # with and without this probe).  Rewind anyway: one generator hardcoding a
        # `torch.randn(...)` would otherwise shift every later sample's values and
        # quietly move tolerance-borderline comparisons suite-wide, for a probe
        # whose only output is a device.
        torch.set_rng_state(rng_state)
    return [_tensor_specs(sample) for sample in samples]


def _to_device(
    sample: Any,
    device: str,
    placement: list[tuple[torch.Size, torch.dtype, torch.device]] | None,
) -> Any:
    """`sample`, built on the CPU, with every tensor moved to `device` except
    the ones `placement` says OpInfo keeps on the CPU.

    `placement` is only trusted when its tensors line up with this sample's
    one for one, same shapes and same dtypes; otherwise the two builds
    diverged and position does not identify an argument, so everything moves.
    """
    if placement is not None:
        specs = _tensor_specs(sample)
        if [s[:2] for s in specs] != [p[:2] for p in placement]:
            placement = None
    keep_on_cpu = (
        iter([p[2].type == "cpu" for p in placement]) if placement is not None else None
    )

    def move(value: Any) -> Any:
        # transform() also visits torch.dtype values, which have no device.
        if not isinstance(value, torch.Tensor):
            return value
        if keep_on_cpu is not None and next(keep_on_cpu):
            return value
        return value.to(device)

    return sample.transform(move)


def _cross_device_comparison_skip_reason(op: Any, dtype: torch.dtype) -> str | None:
    """None, or why `test_matches_cpu` should not compare this op's output.

    OpInfo already flags operators whose output is legitimately allowed to
    differ across devices by skipping `TestCommon.test_compare_cpu`,
    upstream's own CPU-vs-other-device consistency test: uninitialized
    memory for the `empty` family, a per-device RNG stream for `dropout`
    and `multi_head_attention_forward`. `native_batch_norm` is a sharper
    case: with `training=False` ATen's CPU kernel returns
    `save_mean`/`save_invstd` as empty `(0,)` tensors
    (`aten/src/ATen/native/Normalization.cpp`, `batch_norm_cpu`), while its
    CUDA kernel always returns them populated with `running_mean` and
    `rsqrt(running_var + eps)` (`aten/src/ATen/native/cuda/Normalization.cu`,
    `batch_norm_cuda_out`) -- required by `native_batch_norm_backward` so a
    frozen (eval-mode) BatchNorm still produces gradients. Upstream has
    known about this CPU/CUDA split since 2022 (pytorch/pytorch#85960) and,
    rather than pick a side, encodes it as an unconditional
    `DecorateInfo(expectedFailure, 'TestCommon', 'test_compare_cpu')` on the
    op's own OpInfo entry: CPU and an accelerator are not required to agree
    here. `test_matches_cpu` is the same comparison upstream's
    `test_compare_cpu` makes, just against a different accelerator, so it
    honors the same verdict instead of hand-maintaining a duplicate list of
    "known-divergent" ops that would drift out of sync with upstream's own.

    Not every `test_compare_cpu` skip upstream carries is this principled --
    e.g. `nn.functional.conv3d` skips it unconditionally with only "break
    slow tests" as a comment, not a documented semantic split. Honoring
    those too trades a small amount of value-comparison coverage for the
    same reason: a hand-picked subset would itself be a drift-prone list,
    and re-deriving "principled vs. merely convenient" from a skip that
    carries no machine-readable reason is not something this function can
    do reliably.
    """
    for skip in op.skips:
        # op.skips is not guaranteed to be homogeneous: some OpInfo entries
        # (e.g. linalg.solve_triangular) mix in a bare decorator function
        # (skipCPUIfNoLapack) alongside proper DecorateInfo instances.
        if getattr(skip, "test_name", None) != "test_compare_cpu":
            continue
        if skip.cls_name not in (None, "TestCommon"):
            continue
        if skip.device_type not in (None, "privateuse1"):
            continue
        if skip.dtypes is not None and dtype not in skip.dtypes:
            continue
        if callable(skip.active_if):
            # DecorateInfo.is_active() would call this with the concrete
            # test's param_kwargs, which we don't have here; treating a
            # callable as unconditionally active (its truthiness, not its
            # result) would be wrong, so decline it rather than guess.
            continue
        if not skip.active_if:
            continue
        return (
            "OpInfo skips TestCommon.test_compare_cpu for this op "
            "(output not expected to match across devices)"
        )
    return None


class TestOpInfoConformance(TestCase):
    """One test per (operator, dtype), driven entirely by OpInfo metadata."""

    @ops(op_db, allowed_dtypes=_DTYPES)
    def test_matches_cpu(self, device: str, dtype: torch.dtype, op: Any) -> None:
        """Same operator, same inputs, mojo vs CPU, at OpInfo's own tolerance.

        Samples are built on the CPU and moved, never built on the device.
        `sample_inputs(device=...)` constructs its tensors THERE, so a backend
        missing any op that `make_tensor` reaches fails during input
        construction and reports as if the operator under test were broken --
        which is a property of the harness, not of the operator.  Moving also
        makes both legs read bit-identical inputs.

        Which tensors to move is OpInfo's call, not ours: `_opinfo_placements`
        reads back where the operator's own sample generator puts each tensor
        when it targets a device, so arguments ATen requires on the CPU stay
        there instead of being moved and rejected.
        """
        skip_reason = _cross_device_comparison_skip_reason(op, dtype)
        if skip_reason is not None:
            self.skipTest(skip_reason)
        placements = _opinfo_placements(op, dtype)
        checked = 0
        for index, sample in enumerate(
            op.sample_inputs("cpu", dtype, requires_grad=False)
        ):
            placement = None
            if placements is not None and index < len(placements):
                placement = placements[index]
            try:
                moved = _to_device(sample, device, placement)
                actual = op(moved.input, *moved.args, **moved.kwargs)
            except Exception as exc:  # noqa: BLE001 - triaging is the point
                if _not_implemented(exc):
                    self.skipTest(f"not implemented on mojo: {exc}")
                raise
            expected = op(sample.input, *sample.args, **sample.kwargs)
            # assertEqual carries the OpInfo precisionOverride for this dtype
            # when the operator declares one; otherwise assert_close defaults.
            self.assertEqual(_to_cpu(actual), expected, exact_dtype=True)
            checked += 1
        if checked == 0:
            self.skipTest("OpInfo produced no sample inputs for this dtype")

    @ops(
        [op for op in op_db if op.error_inputs_func is not None],
        allowed_dtypes=(torch.float32,),
    )
    def test_errors_match(self, device: str, dtype: torch.dtype, op: Any) -> None:
        """The inputs PyTorch says must raise, must raise here too.

        A backend that silently accepts a malformed call is a worse failure
        than one that cannot run it at all, and only OpInfo knows which calls
        those are per operator.
        """
        checked = 0
        error_inputs = op.error_inputs(device)
        while True:
            try:
                error_input = next(error_inputs)
            except StopIteration:
                break
            except Exception as exc:  # noqa: BLE001 - triaging is the point
                # Building the error input itself (not the op call) can hit a
                # backend limitation, e.g. a dtype-mismatch case constructed
                # via a dtype MAX doesn't support at all.
                if _not_implemented(exc):
                    self.skipTest(f"not implemented on mojo: {exc}")
                raise
            sample = error_input.sample_input
            # `self.skipTest(...)` must not be reachable from inside a
            # `with self.assertRaises(error_input.error_type):` block: a
            # sufficiently broad error_type (plain `Exception`, which every
            # `unittest.SkipTest` also is) would catch it as if it were the
            # expected exception, silently turning a real skip into a false
            # pass. So classify the raised/returned outcome first, and only
            # reach for assertRaises afterwards, scoped to the one check
            # that needs its swallow-on-match behavior (see below).
            try:
                out = op(sample.input, *sample.args, **sample.kwargs)
            except Exception as exc:  # noqa: BLE001 - triaging is the point
                if _not_implemented(exc):
                    self.skipTest(f"not implemented on mojo: {exc}")
                if not isinstance(exc, error_input.error_type):
                    raise
            else:
                # Some dunder ops (e.g. __rmod__) called directly, rather
                # than through operator syntax, correctly decline by
                # returning the `NotImplemented` sentinel instead of raising.
                # Upstream's own test_ops.py::test_errors accepts exactly
                # that outcome, via this identical assertFalse nested inside
                # `assertRaises(error_type)`: nesting it only here (not
                # around the skipTest above, which the original bug did)
                # still fails for real on a mismatch or on a call that
                # returned an ordinary value without raising anything.
                with self.assertRaises(error_input.error_type):
                    self.assertFalse(isinstance(out, type(NotImplemented)))
            checked += 1
        if checked == 0:
            self.skipTest("OpInfo declared no error inputs for this operator")


instantiate_device_type_tests(
    TestOpInfoConformance, globals(), only_for=("privateuse1",)
)


if __name__ == "__main__":
    run_tests()
