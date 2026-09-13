"""Native `_foreach_*` / `_fused_adamw_` ops (`native/mojo/ops_foreach.mojo`).

Migrated from `tests/test_eager_optimizer_ops.py` (deleted): only the tests
that exercise ops this group registers. That file also covered `lerp.Scalar`,
`sub.out`, `mul.out`, `linalg_vector_norm.out`, `any.out` and
`isin.Tensor_Tensor_out` directly -- those belong to the binary/reductions/
compare groups' own native test files, not here.

`native.op_count("aten::<op>")` (not `CallChecker`) verifies a call reached
this group's boxed kernel: the in-place foreach overloads (`_foreach_add_.
Scalar`, ...) have no `aten_functions` twin for `CallChecker`'s name-derived
matching to key off (torch.compile always sees the functionalized, non-
underscore form), and `_foreach_norm.Scalar` / `_fused_adamw_` have none
either. `_foreach_sqrt` is the one op here with a matching
`aten_functions.aten__foreach_sqrt` twin, so it uses `call_checker` like the
other groups' tests do.

Some correctness tests below exercise the SEQUENTIAL fallback (mixed dtype,
non-contiguous, non-fp32), which calls the per-tensor op (`add_.Scalar`,
`mul_.Tensor`, `sqrt`, `linalg_vector_norm`, ...) through `tmb_call_op`.
Until the groups that register those land in the integrated tree, those
particular tests raise NotImplementedError from the dispatcher (no
PrivateUse1 kernel and no CompositeExplicitAutograd registration for a bare
`Tensor.mul_(Scalar)` et al.) -- that is an integration gap, not a bug in
this file.
"""

import pytest
import torch

from torch_mojo_backend import aten_functions, native
from torch_mojo_backend.testing import CallChecker

pytestmark = pytest.mark.xdist_group(name="group_native_foreach")


def _watch(op_name: str) -> None:
    """Reset the shim's per-op call counters and start counting."""
    del op_name  # kept as a parameter so call sites read like a doc comment
    native.op_counting(True)
    native.op_counts_reset()


def _assert_ran(op_name: str):
    assert native.op_count(op_name) > 0, f"{op_name} did not run natively"


def _fused_adamw_case(device: str, *, amsgrad: bool):
    """Nonuniform, nonzero AdamW state without shared storage (empty, 0-d,
    1-d, and 2-d shapes; deterministic values, no RNG)."""
    shapes = ((), (0,), (7,), (17, 65))

    def values(shape: tuple[int, ...], *, scale: float, offset: float) -> torch.Tensor:
        numel = torch.empty(shape).numel()
        return (
            torch.arange(numel, dtype=torch.float32)
            .mul(scale)
            .add(offset)
            .reshape(shape)
        )

    parameters = [
        values(shape, scale=0.003, offset=-0.75 + index * 0.1)
        for index, shape in enumerate(shapes)
    ]
    gradients = [
        values(shape, scale=-0.0007, offset=0.3 - index * 0.02)
        for index, shape in enumerate(shapes)
    ]
    exp_avgs = [
        values(shape, scale=0.0002, offset=-0.08 + index * 0.01)
        for index, shape in enumerate(shapes)
    ]
    exp_avg_sqs = [
        values(shape, scale=0.00001, offset=0.01 + index * 0.001)
        for index, shape in enumerate(shapes)
    ]
    max_exp_avg_sqs = [value.mul(1.25) for value in exp_avg_sqs] if amsgrad else []
    state_steps = [
        torch.tensor(float(step), dtype=torch.float32) for step in (3, 5, 11, 17)
    ]
    groups = (
        parameters,
        gradients,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    )
    return tuple([[value.to(device) for value in group] for group in groups])


@pytest.mark.parametrize(
    ("amsgrad", "maximize"), [(False, False), (False, True), (True, False)]
)
def test_fused_adamw_matches_cpu(mojo_gpu: str, amsgrad: bool, maximize: bool):
    _watch("aten::_fused_adamw_")
    cpu_groups = _fused_adamw_case("cpu", amsgrad=amsgrad)
    mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=amsgrad)
    kwargs = {
        "lr": 0.025,
        "beta1": 0.8,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": amsgrad,
        "maximize": maximize,
    }

    assert torch.ops.aten._fused_adamw_.default(*cpu_groups, **kwargs) is None
    assert torch.ops.aten._fused_adamw_.default(*mojo_groups, **kwargs) is None

    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    _assert_ran("aten::_fused_adamw_")


def test_fused_adamw_tensor_lr_matches_cpu(mojo_gpu: str):
    _watch("aten::_fused_adamw_.tensor_lr")
    cpu_groups = _fused_adamw_case("cpu", amsgrad=False)
    mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=False)
    cpu_lr = torch.tensor(0.0125, dtype=torch.float32)
    mojo_lr = cpu_lr.to(mojo_gpu)
    kwargs = {
        "beta1": 0.8,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }

    torch.ops.aten._fused_adamw_.tensor_lr(*cpu_groups, lr=cpu_lr, **kwargs)
    torch.ops.aten._fused_adamw_.tensor_lr(*mojo_groups, lr=mojo_lr, **kwargs)

    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    _assert_ran("aten::_fused_adamw_.tensor_lr")


def test_fused_adamw_grad_scale_and_found_inf(mojo_gpu: str):
    """found_inf=1 must skip every write; found_inf=0 with grad_scale halves
    the (unwritten) gradient's effective magnitude but otherwise runs."""
    _watch("aten::_fused_adamw_")
    for found_inf_value in (0.0, 1.0):
        cpu_groups = _fused_adamw_case("cpu", amsgrad=True)
        mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=True)
        before = [[t.cpu().clone() for t in group] for group in mojo_groups]
        cpu_grad_scale = torch.tensor(2.0, dtype=torch.float32)
        mojo_grad_scale = cpu_grad_scale.to(mojo_gpu)
        cpu_found_inf = torch.tensor(found_inf_value, dtype=torch.float32)
        mojo_found_inf = cpu_found_inf.to(mojo_gpu)
        kwargs = {
            "lr": 0.025,
            "beta1": 0.8,
            "beta2": 0.95,
            "weight_decay": 0.1,
            "eps": 1e-8,
            "amsgrad": True,
            "maximize": False,
        }
        torch.ops.aten._fused_adamw_.default(
            *cpu_groups, grad_scale=cpu_grad_scale, found_inf=cpu_found_inf, **kwargs
        )
        torch.ops.aten._fused_adamw_.default(
            *mojo_groups, grad_scale=mojo_grad_scale, found_inf=mojo_found_inf, **kwargs
        )
        for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
            for expected, actual in zip(cpu_group, mojo_group, strict=True):
                torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
        if found_inf_value == 1.0:
            for before_group, after_group in zip(before, mojo_groups, strict=True):
                for expected, actual in zip(before_group, after_group, strict=True):
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    _assert_ran("aten::_fused_adamw_")


def test_fused_adamw_validates_every_tensor_before_write(mojo_gpu: str):
    _watch("aten::_fused_adamw_")
    parameters = [
        torch.tensor([1.0, -2.0], device=mojo_gpu),
        torch.tensor([3.0, -4.0, 5.0], device=mojo_gpu),
    ]
    grads = [torch.ones_like(p) for p in parameters]
    exp_avgs = [torch.zeros_like(p) for p in parameters]
    exp_avg_sqs = [torch.ones_like(p) for p in parameters]
    # Only the SECOND entry is malformed: a partial-launch implementation
    # would already have updated the valid first one before discovering this.
    exp_avg_sqs[1] = torch.ones(4, device=mojo_gpu)
    steps = [torch.tensor(1.0, device=mojo_gpu) for _ in parameters]
    mutable = (parameters, grads, exp_avgs, exp_avg_sqs)
    snapshot = [[t.cpu().clone() for t in group] for group in mutable]

    with pytest.raises(RuntimeError, match="same dtype, device, shape"):
        torch.ops.aten._fused_adamw_.default(
            parameters,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            steps,
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
    for snap_group, actual_group in zip(snapshot, mutable, strict=True):
        for snap, actual in zip(snap_group, actual_group, strict=True):
            torch.testing.assert_close(actual.cpu(), snap, rtol=0, atol=0)


def test_fused_adamw_optimizer_two_groups_matches_cpu(mojo_gpu: str):
    """The public torch.optim.AdamW(fused=True) entry point end to end."""
    _watch("aten::_fused_adamw_")
    initial = [
        torch.linspace(-1.0, 1.0, 31, dtype=torch.float32),
        torch.linspace(0.5, -0.75, 35, dtype=torch.float32).reshape(5, 7),
    ]
    cpu_parameters = [torch.nn.Parameter(v.clone()) for v in initial]
    mojo_parameters = [torch.nn.Parameter(v.to(mojo_gpu)) for v in initial]
    cpu_optimizer = torch.optim.AdamW(
        [
            {"params": [cpu_parameters[0]], "weight_decay": 0.1},
            {"params": [cpu_parameters[1]], "weight_decay": 0.0},
        ],
        lr=0.0125,
        betas=(0.8, 0.95),
        eps=1e-8,
        fused=True,
    )
    mojo_optimizer = torch.optim.AdamW(
        [
            {"params": [mojo_parameters[0]], "weight_decay": 0.1},
            {"params": [mojo_parameters[1]], "weight_decay": 0.0},
        ],
        lr=0.0125,
        betas=(0.8, 0.95),
        eps=1e-8,
        fused=True,
    )

    for step in range(2):
        for index, (cpu_parameter, mojo_parameter) in enumerate(
            zip(cpu_parameters, mojo_parameters, strict=True)
        ):
            gradient = torch.linspace(
                -0.2 + step * 0.03,
                0.35 - index * 0.02,
                cpu_parameter.numel(),
                dtype=torch.float32,
            ).reshape(cpu_parameter.shape)
            cpu_parameter.grad = gradient.clone()
            mojo_parameter.grad = gradient.to(mojo_gpu)
        cpu_optimizer.step()
        mojo_optimizer.step()

    assert all(group["fused"] is True for group in mojo_optimizer.param_groups)
    for cpu_parameter, mojo_parameter in zip(
        cpu_parameters, mojo_parameters, strict=True
    ):
        torch.testing.assert_close(
            mojo_parameter.cpu(), cpu_parameter, rtol=2e-6, atol=2e-7
        )
        cpu_state = cpu_optimizer.state[cpu_parameter]
        mojo_state = mojo_optimizer.state[mojo_parameter]
        for name in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                mojo_state[name].cpu(), cpu_state[name], rtol=2e-6, atol=2e-7
            )
        assert mojo_state["step"].device == torch.device(mojo_gpu)
    _assert_ran("aten::_fused_adamw_")


def _foreach_lists(device: str) -> list[list[torch.Tensor]]:
    """Nonuniform FP32 lists: empties, multi-dim, and one tensor crossing the
    65_536-element chunk boundary. The third list stays bounded away from 0
    (used as a divisor by callers that need one)."""
    shapes = ((7,), (17, 65), (0,), (5, 3, 2), (65_539,))

    def values(shape: tuple[int, ...], *, scale: float, offset: float) -> torch.Tensor:
        numel = torch.empty(shape).numel()
        return (
            torch.arange(numel, dtype=torch.float32)
            .mul(scale)
            .add(offset)
            .reshape(shape)
        )

    mutated = [
        values(shape, scale=0.003, offset=-0.4 + index * 0.1)
        for index, shape in enumerate(shapes)
    ]
    operands = [
        values(shape, scale=-0.0007, offset=0.3 - index * 0.02)
        for index, shape in enumerate(shapes)
    ]
    divisors = [
        values(shape, scale=0.0002, offset=-0.08 + index * 0.01).abs().add(0.5)
        for index, shape in enumerate(shapes)
    ]
    return [[t.to(device) for t in group] for group in (mutated, operands, divisors)]


_FOREACH_BATCHED_OPS = [
    ("aten::_foreach_add_.Scalar", lambda a, b, c: torch._foreach_add_(a, 1e-3)),
    (
        "aten::_foreach_addcmul_.Scalar",
        lambda a, b, c: torch._foreach_addcmul_(a, b, c, value=0.01),
    ),
    ("aten::_foreach_lerp_.Scalar", lambda a, b, c: torch._foreach_lerp_(a, b, 0.1)),
    ("aten::_foreach_mul_.Scalar", lambda a, b, c: torch._foreach_mul_(a, 0.998)),
]


@pytest.mark.parametrize(("op_name", "apply"), _FOREACH_BATCHED_OPS)
def test_batched_foreach_elementwise_matches_cpu(mojo_gpu: str, op_name: str, apply):
    """The batched kernel launch runs and matches CPU's sequential result.

    Not checked here: `Tensor(a!)[]` mutations get no automatic `_version`
    bump the way a single `Tensor(a!)` op does (`add_.Scalar`/`zero_` do;
    `_foreach_add_.Scalar` measurably does not) -- see this group's final
    report for why (it needs a C-level primitive the shim does not expose
    yet, and this file cannot add one).
    """
    _watch(op_name)
    cpu_lists = _foreach_lists("cpu")
    mojo_lists = _foreach_lists(mojo_gpu)
    apply(*cpu_lists)
    apply(*mojo_lists)
    for expected, actual in zip(cpu_lists[0], mojo_lists[0], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    _assert_ran(op_name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
def test_batched_foreach_mul_scalar_reduced_precision(
    mojo_gpu: str, dtype: torch.dtype
):
    """Half/bfloat16 lists also take the batched path (mul_/add_ widen to
    FP32 and narrow back, matching the per-tensor `mul_.Scalar` route)."""
    _watch("aten::_foreach_mul_.Scalar")
    host = [torch.linspace(-3.0, 4.0, length).to(dtype) for length in (7, 65_539, 1)]
    device = [tensor.to(mojo_gpu) for tensor in host]
    torch._foreach_mul_(device, 0.998)
    for expected, actual in zip(host, device, strict=True):
        torch.testing.assert_close(
            actual.cpu(), (expected.float() * 0.998).to(dtype), rtol=0, atol=0
        )
    _assert_ran("aten::_foreach_mul_.Scalar")


def test_foreach_mul_tensor_matches_cpu(mojo_gpu: str):
    _watch("aten::_foreach_mul_.Tensor")
    cpu_lists = _foreach_lists("cpu")
    mojo_lists = _foreach_lists(mojo_gpu)
    cpu_scalar = torch.tensor(0.25, dtype=torch.float32)
    mojo_scalar = cpu_scalar.to(mojo_gpu)
    torch._foreach_mul_(cpu_lists[0], cpu_scalar)
    torch._foreach_mul_(mojo_lists[0], mojo_scalar)
    for expected, actual in zip(cpu_lists[0], mojo_lists[0], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    _assert_ran("aten::_foreach_mul_.Tensor")


def test_foreach_mul_tensor_rejects_non_0d_other(mojo_gpu: str):
    tensors = [torch.tensor([1.0, 2.0], device=mojo_gpu)]
    other = torch.tensor([1.0, 2.0], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="0 dim"):
        torch._foreach_mul_(tensors, other)


def test_batched_foreach_sqrt_matches_cpu(mojo_gpu: str, call_checker: CallChecker):
    call_checker.register(aten_functions.aten__foreach_sqrt)
    # abs() on the CPU tensor, then transfer: aten::abs is another group's op,
    # not yet registered in this worktree.
    cpu_inputs = [tensor.abs() for tensor in _foreach_lists("cpu")[0]]
    mojo_inputs = [tensor.to(mojo_gpu) for tensor in cpu_inputs]
    # Correctly-rounded float32 sqrt (via float64), not torch's CPU float32
    # sqrt (MKL vsSqrt is documented <=1ulp and is not itself bit-exact).
    expected = [tensor.double().sqrt().to(tensor.dtype) for tensor in cpu_inputs]
    actual = torch._foreach_sqrt(mojo_inputs)
    for expected_out, actual_out in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_out.cpu(), expected_out, rtol=0, atol=0)


def test_foreach_norm_scalar_matches_cpu(mojo_gpu: str):
    _watch("aten::_foreach_norm.Scalar")
    tensors = [t.abs().add(0.1) for t in _foreach_lists("cpu")[0] if t.numel() > 0]
    mojo_tensors = [t.to(mojo_gpu) for t in tensors]
    expected = torch._foreach_norm(tensors, 2)
    actual = torch._foreach_norm(mojo_tensors, 2)
    for e, a in zip(expected, actual, strict=True):
        torch.testing.assert_close(a.cpu(), e, rtol=2e-6, atol=2e-7)
    _assert_ran("aten::_foreach_norm.Scalar")


@pytest.mark.xfail(
    reason=(
        "sequential fallback needs linalg_vector_norm registered by the"
        " reductions group; not yet present in this worktree"
    ),
    strict=False,
)
def test_foreach_norm_scalar_ord_one_uses_fallback(mojo_gpu: str):
    """ord != 2 always takes the per-tensor `linalg_vector_norm` fallback,
    for any ord -- it is ATen's own `foreach_tensor_norm_slow` composition."""
    tensors = [t.abs().add(0.1) for t in _foreach_lists("cpu")[0] if t.numel() > 0]
    mojo_tensors = [t.to(mojo_gpu) for t in tensors]
    expected = torch._foreach_norm(tensors, 1)
    actual = torch._foreach_norm(mojo_tensors, 1)
    for e, a in zip(expected, actual, strict=True):
        torch.testing.assert_close(a.cpu(), e, rtol=2e-6, atol=2e-7)


@pytest.mark.xfail(
    reason=(
        "sequential fallback needs mul_.Scalar registered by the binary"
        " group; not yet present in this worktree"
    ),
    strict=False,
)
def test_batched_foreach_falls_back_for_non_f32(mojo_gpu: str):
    """Unsupported regimes (here: integer dtype) reach ATen's ordinary
    per-tensor semantics unchanged, via the sequential fallback."""
    cpu_tensors = [torch.arange(5), torch.arange(3)]
    mojo_tensors = [tensor.to(mojo_gpu) for tensor in cpu_tensors]
    torch._foreach_mul_(cpu_tensors, 3)
    torch._foreach_mul_(mojo_tensors, 3)
    for expected, actual in zip(cpu_tensors, mojo_tensors, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.xfail(
    reason=(
        "sequential fallback needs mul_.Scalar registered by the binary"
        " group; not yet present in this worktree"
    ),
    strict=False,
)
def test_batched_foreach_mixed_dtype_list_falls_back(mojo_gpu: str):
    """A two-dtype list is not one launch; still gets ATen's answer via the
    per-tensor fallback."""
    mixed = [
        torch.tensor([1.5, -2.0]).to(mojo_gpu),
        torch.tensor([1.5, -2.0], dtype=torch.bfloat16).to(mojo_gpu),
    ]
    torch._foreach_mul_(mixed, 2.0)
    torch.testing.assert_close(
        mixed[0].cpu(), torch.tensor([3.0, -4.0]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        mixed[1].cpu(), torch.tensor([3.0, -4.0], dtype=torch.bfloat16), rtol=0, atol=0
    )


@pytest.mark.xfail(
    reason=(
        "unregistered Scalar[] ops (see ops_foreach.mojo module docstring):"
        " ATen's own sequential decomposition needs div_.Scalar / addcdiv_"
        " registered by the binary group; not yet present in this worktree"
    ),
    strict=False,
)
@pytest.mark.parametrize(
    ("op_name", "apply"),
    [
        ("div", lambda a, b, c: torch._foreach_div_(a, [0.5, -1.5, 2.0, 0.25, -0.125])),
        (
            "addcdiv",
            lambda a, b, c: torch._foreach_addcdiv_(
                a, b, c, [0.5, -1.5, 2.0, 0.25, -0.125]
            ),
        ),
    ],
)
def test_foreach_scalarlist_ops_unregistered_still_correct(
    mojo_gpu: str, op_name: str, apply
):
    """`_foreach_div_.ScalarList` / `_foreach_addcdiv_.ScalarList` are not
    registered here at all (Scalar[] cannot reach our boxed kernel yet), so
    these must produce ATen's ordinary answer through its own
    CompositeExplicitAutograd decomposition end to end."""
    cpu_lists = _foreach_lists("cpu")
    mojo_lists = _foreach_lists(mojo_gpu)
    apply(*cpu_lists)
    apply(*mojo_lists)
    for expected, actual in zip(cpu_lists[0], mojo_lists[0], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)


def test_batched_inplace_foreach_bumps_the_version_counter(mojo_gpu):
    """The dispatcher bumps `Tensor(a!)` arguments but not the members of a
    `Tensor(a!)[]`; the batched kernels do it themselves so autograd notices
    an in-place update of a saved tensor."""
    xs = [torch.ones(8, device=mojo_gpu) for _ in range(3)]
    before = [x._version for x in xs]
    torch._foreach_add_(xs, 1.0)
    torch._foreach_mul_(xs, 2.0)
    assert [x._version - v for x, v in zip(xs, before)] == [2, 2, 2]
    assert xs[0].cpu().tolist() == [4.0] * 8
