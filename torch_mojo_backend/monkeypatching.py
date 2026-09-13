"""Every monkeypatch this project applies, in one place.

A monkeypatch here means replacing or mutating, at runtime, an attribute of
a module or class we do not own -- mostly PyTorch internals that have no
extension point for a Python-implemented PrivateUse1 backend yet. Each patch
is one function with a docstring saying what upstream lacks, so that the
patch can be turned into an upstream PR and deleted from here. Nothing else
in the package may patch a third-party module; keep new patches in this file
(``tests/test_monkeypatching_is_centralized.py`` enforces that for
``torch``-rooted assignments).

Every patch installer is called by ``register.register_mojo_devices``, never
at import time. :func:`apply_torch_monkeypatches` bundles the patches that
are specific to the old wrapper-subclass eager device (``TorchMojoTensor``);
it is not currently wired up now that the ``mojo`` device is a native
PrivateUse1 backend with real tensor storage (see docs/native_backend.md),
and reinstalling it wholesale would be wrong -- e.g.
:func:`_install_torch_stream_event_dispatch` would redirect
``torch.Stream``/``torch.Event`` away from the native backend's real,
generic ones. A patch that is still needed regardless of which device
layer is active (e.g. :func:`fix_privateuse1_dlpack_device_type`) is its own
public function, called directly. Official registration APIs
(``torch.library.impl``, the PrivateUse1 backend module, ``torch.__future__``
toggles) are not monkeypatches and stay in ``mojo_device/register.py``.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from types import ModuleType
from typing import TypeVar

import torch
import torch._functorch._aot_autograd.runtime_wrappers as runtime_wrappers
import torch.distributed.distributed_c10d as c10d
import torch._subclasses.fake_tensor as fake_tensor_module
import torch.fx.experimental.proxy_tensor as proxy_tensor
import torch.optim.optimizer as optimizer_module
import torch.utils._foreach_utils as foreach_utils
from torch._dynamo.variables.builder import VariableBuilder
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode

from torch_mojo_backend.mojo_device import streams as mojo_streams
from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

_T = TypeVar("_T")


def _install_torch_accelerator_synchronize(torch_mojo_device_module: ModuleType):
    """Route generic accelerator synchronization to the Mojo device module.

    PyTorch's Python PrivateUse1 guard does not yet forward synchronizeDevice
    to the registered Python module, so preserve public device validation and
    delegate the actual queue drain here until that hook exists upstream.
    """
    original_synchronize = torch.accelerator.synchronize
    if getattr(original_synchronize, "_torch_mojo_backend", False):
        return

    mojo_device = torch.device("mojo")
    current_accelerator = torch.accelerator.current_accelerator()
    if current_accelerator != mojo_device:
        raise RuntimeError(
            "registering Mojo did not make it the current torch.accelerator: "
            f"{current_accelerator}"
        )

    @wraps(original_synchronize)
    def synchronize(device: torch.device | str | int | None = None):
        current = torch.accelerator.current_accelerator()
        if current != mojo_device:
            original_synchronize(device)
            return

        if device is None:
            device_index = torch_mojo_device_module.current_device()
        elif isinstance(device, int):
            device_index = device
        else:
            selected = torch.device(device)
            if selected.type != "mojo":
                raise ValueError(
                    f"{selected.type} doesn't match the current accelerator {current}."
                )
            device_index = (
                torch_mojo_device_module.current_device()
                if selected.index is None
                else selected.index
            )
        torch_mojo_device_module.synchronize(device_index)

    # The sentinel attribute is how this function detects, on a later call,
    # that its own replacement (not the original) is already installed. ty
    # can't type an ad hoc attribute on a stdlib callable.
    synchronize._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.accelerator.synchronize = synchronize  # ty: ignore[invalid-assignment]


def _install_torch_accelerator_stream_api(torch_mojo_device_module: ModuleType):
    """Route torch.accelerator.current_stream/set_stream to mojo streams.

    Same reason as the synchronize patch above: the Python PrivateUse1
    guard's stream hooks are stubs (every torch.Stream it mints is a silent
    no-op object), so the generic accelerator entry points would otherwise
    hand back streams that do nothing.
    """
    original_current_stream = torch.accelerator.current_stream
    if getattr(original_current_stream, "_torch_mojo_backend", False):
        return
    mojo_device = torch.device("mojo")

    @wraps(original_current_stream)
    def current_stream(device: torch.device | str | int | None = None) -> torch.Stream:
        if torch.accelerator.current_accelerator() != mojo_device:
            return original_current_stream(device)
        return torch_mojo_device_module.current_stream(device)

    # Same sentinel as `synchronize` above.
    current_stream._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.accelerator.current_stream = current_stream  # ty: ignore[invalid-assignment]

    original_set_stream = torch.accelerator.set_stream

    @wraps(original_set_stream)
    def set_stream(stream: torch.Stream):
        if isinstance(stream, mojo_streams.Stream):
            torch_mojo_device_module.set_stream(stream)
        else:
            original_set_stream(stream)

    set_stream._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.accelerator.set_stream = set_stream  # ty: ignore[invalid-assignment]


def _forwarding_constructor(cls: type[_T]) -> Callable[..., _T]:
    """`cls` as an opaque callable: the metaclasses below forward whatever
    constructor arguments the caller passed, and only `cls` knows their shape."""
    return cls


def _install_torch_stream_event_dispatch():
    """Dispatch torch.Stream/torch.Event on mojo devices to real classes.

    Construction through the original classes reaches the stub C++ guard and
    yields inert objects, so a metaclass routes mojo devices to
    mojo_device/streams.py while delegating everything else (and the
    explicit ``stream_id=``/``device_type=`` reconstruction overload) to the
    originals. ``isinstance`` accepts both families in both directions.
    """
    if getattr(torch.Stream, "_torch_mojo_backend", False):
        return
    original_stream = torch.Stream
    original_event = torch.Event
    construct_stream = _forwarding_constructor(original_stream)
    construct_mojo_stream = _forwarding_constructor(mojo_streams.Stream)
    construct_event = _forwarding_constructor(original_event)
    construct_mojo_event = _forwarding_constructor(mojo_streams.Event)

    def _wants_mojo(args: tuple[object, ...], kwargs: dict[str, object]) -> bool:
        device = kwargs.get("device", args[0] if args else None)
        if device is None or isinstance(device, int):
            accelerator = torch.accelerator.current_accelerator()
            return accelerator is not None and accelerator.type == "mojo"
        if not isinstance(device, str | torch.device):
            return False  # torch.device() would raise TypeError
        try:
            return torch.device(device).type == "mojo"
        except (TypeError, RuntimeError, ValueError):
            return False

    class _StreamMeta(type):
        def __call__(cls, *args: object, **kwargs: object) -> torch.Stream:
            if "stream_id" in kwargs or "device_type" in kwargs:
                return construct_stream(*args, **kwargs)
            if _wants_mojo(args, kwargs):
                return construct_mojo_stream(*args, **kwargs)
            return construct_stream(*args, **kwargs)

        def __instancecheck__(cls, instance: object) -> bool:
            return isinstance(instance, (original_stream, mojo_streams.Stream))

    class Stream(metaclass=_StreamMeta):
        _torch_mojo_backend = True

    class _EventMeta(type):
        def __call__(
            cls, *args: object, **kwargs: object
        ) -> torch.Event | mojo_streams.Event:
            if _wants_mojo(args, kwargs):
                return construct_mojo_event(*args, **kwargs)
            return construct_event(*args, **kwargs)

        def __instancecheck__(cls, instance: object) -> bool:
            return isinstance(instance, (original_event, mojo_streams.Event))

    class Event(metaclass=_EventMeta):
        _torch_mojo_backend = True

    torch.Stream = Stream  # ty: ignore[invalid-assignment]
    torch.Event = Event  # ty: ignore[invalid-assignment]


def _declare_mojo_tensor_foreach_capable():
    """Add TorchMojoTensor to torch's foreach-optimizer allowlists.

    PyTorch deliberately uses exact-type checks before selecting foreach
    optimizers. TorchMojoTensor is a transparent PrivateUse1 wrapper, so opt
    it into the same lists as DTensor and other supported tensor types.
    """
    for supported_types in (
        optimizer_module._foreach_supported_types,
        foreach_utils._foreach_supported_types,
    ):
        if TorchMojoTensor not in supported_types:
            # torch declared these lists' element type from their own
            # literal contents (Tensor/Parameter), which doesn't include our
            # subclass by construction.
            supported_types.append(TorchMojoTensor)  # ty: ignore[invalid-argument-type]


def _declare_mojo_tensor_as_plain_tensor():
    """Add TorchMojoTensor to torch's HANDLED_TYPES allowlists.

    TorchMojoTensor's wrapper dispatch is transparent to numerical operations;
    all backend logic still lives behind the PrivateUse1 dispatch key. A couple
    of tracing exact-type allowlists nevertheless need to know about it:

    - aot_autograd's first-invocation `_AnalyzeCustomOpInputOutputMode`
      returns NotImplemented for unknown tensor types, which makes every
      dispatched op fail on mojo tensors under eager-executing compile
      backends (e.g. "aot_eager"). runtime_wrappers imported the tuple by
      value, so patch both bindings.
    - FakeTensorMode returns NotImplemented for ops whose args include an
      unrecognized tensor subclass, to give that subclass's
      __torch_dispatch__ a chance to run. The Mojo wrapper must instead be
      fakeified rather than executing a real PrivateUse1 kernel — hit when
      dynamo lifts a mojo tensor constant created mid-trace (e.g.
      `torch.tensor([], device="mojo")` in HF generate).
    - Functionalization has the same gate twice (on `FunctionalTensor` and
      on `FunctionalTensorMode`), and it is not an allowlist we can append
      to, so the mojo wrapper is presented to those two gates as the plain
      tensor it behaves like. Without it, AOTAutograd tracing a graph that
      mixes a functional tensor with a real mojo constant dies with
      "unsupported operand type(s) for +: 'FunctionalTensor' and
      'TorchMojoTensor'".
    """
    if TorchMojoTensor not in proxy_tensor.HANDLED_TYPES:
        # torch infers HANDLED_TYPES' type from its own fixed-length tuple
        # literal, so extending it by one element is always a "wrong length"
        # mismatch to the checker.
        proxy_tensor.HANDLED_TYPES = (  # ty: ignore[invalid-assignment]
            *proxy_tensor.HANDLED_TYPES,
            TorchMojoTensor,
        )
    runtime_wrappers.HANDLED_TYPES = proxy_tensor.HANDLED_TYPES

    original_check = fake_tensor_module._check_for_subclass_arg

    def check_for_subclass_arg_except_mojo(x: object) -> bool:
        return original_check(x) and not isinstance(x, TorchMojoTensor)

    # ty treats each `def` as its own nominal type even with an identical
    # signature, so this never structurally matches the attribute it replaces.
    fake_tensor_module._check_for_subclass_arg = (  # ty: ignore[invalid-assignment]
        check_for_subclass_arg_except_mojo
    )

    # Once past the subclass check, lifting a real mojo tensor constant
    # still fails: the const-propagation path is gated on `type(out) is
    # Tensor` (its no_dispatch clone is unsafe for subclasses), and the
    # fallback validation rejects non-fake inputs. Fakeify lifted mojo
    # constants directly instead; at runtime AOTAutograd passes the real
    # constant as a graph input like any other mojo tensor.
    original_dispatch_impl = fake_tensor_module.FakeTensorMode._dispatch_impl

    def dispatch_impl_lifting_mojo_constants(
        self: fake_tensor_module.FakeTensorMode,
        func: torch._ops.OpOverload,
        types: Sequence[type],
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> fake_tensor_module.FakeTensor | None:
        if func in self.lift_fns and args and isinstance(args[0], TorchMojoTensor):
            return self.fake_tensor_converter.from_real_tensor(self, args[0])
        return original_dispatch_impl(self, func, types, args, kwargs)

    fake_tensor_module.FakeTensorMode._dispatch_impl = (
        dispatch_impl_lifting_mojo_constants
    )

    def as_plain_tensor_types(types: Sequence[type]) -> tuple[type, ...]:
        return tuple(torch.Tensor if t is TorchMojoTensor else t for t in types)

    original_tensor_dispatch = FunctionalTensor.__torch_dispatch__

    def functional_tensor_dispatch(
        self: object,
        func: object,
        types: Sequence[type],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        assert isinstance(self, FunctionalTensor)
        assert isinstance(func, torch._ops.OpOverload)
        return original_tensor_dispatch(
            self, func, as_plain_tensor_types(types), args, kwargs
        )

    FunctionalTensor.__torch_dispatch__ = functional_tensor_dispatch

    original_mode_dispatch = FunctionalTensorMode.__torch_dispatch__

    def functional_mode_dispatch(
        self: object,
        func: object,
        types: Sequence[type],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        assert isinstance(self, FunctionalTensorMode)
        assert isinstance(func, torch._ops.OpOverload)
        return original_mode_dispatch(
            self, func, as_plain_tensor_types(types), args, kwargs
        )

    FunctionalTensorMode.__torch_dispatch__ = functional_mode_dispatch


def _trace_mojo_tensor_as_a_plain_tensor_in_dynamo():
    """Let TorchDynamo model a mojo tensor as an ordinary device tensor.

    ``VariableBuilder`` only builds a ``TensorVariable`` for a tensor
    subclass that either leaves ``__torch_dispatch__`` untouched or is a
    *traceable wrapper subclass* (``__tensor_flatten__`` /
    ``__tensor_unflatten__``, i.e. a wrapper holding inner tensors that
    AOTAutograd desugars into separate graph inputs).

    TorchMojoTensor is neither, by design: its ``__torch_dispatch__`` is
    only a bracket around a redispatch to the PrivateUse1 kernels, and it
    holds no inner tensors -- its payload is a device allocation, exactly
    like a cuda tensor's. Left alone, dynamo models every mojo tensor as a
    ``UserDefinedObjectVariable`` and refuses to trace ``x * y``.

    ``VariableBuilder._type_dispatch`` is the exact-type table consulted
    before any of those checks; it is where torch itself routes
    ``Parameter``, ``FakeTensor`` and ``FunctionalTensor`` to
    ``wrap_tensor``. Add the mojo wrapper there. The table is memoized per
    ``config.trace_numpy`` value, so populate both.
    """
    for trace_numpy in (False, True):
        table = VariableBuilder._type_dispatch_impl(trace_numpy)
        table[TorchMojoTensor] = VariableBuilder.wrap_tensor


def _keep_mojo_kernels_out_of_fake_tensor_construction():
    """Make FakeTensor construction skip the PrivateUse1 Python kernels.

    `FakeTensor.__new__` calls `Tensor._make_subclass(cls, elem, ...,
    device_for_backend_keys=mojo)`, which internally dispatches
    `aten::detach` on `elem`. While torch.compile traces mojo graphs, `elem`
    is regularly a meta tensor whose dispatch keys carry PrivateUse1 (view
    and matmul meta outputs inherit the fake input's keys), so that detach
    lands in our Python kernel — and `_make_subclass` requires a result
    with no Python object associated, which a Python kernel can never
    produce ("already associated to a python object" RuntimeError).

    Excluding PrivateUse1 for the duration of `FakeTensor.__new__` routes
    that internal detach to the stock C++ meta kernel. Real mojo tensors
    are never legitimately consumed during FakeTensor *construction*, so
    nothing of ours belongs there. (PyTorch's own python-backend reference
    test has this same rough edge — see test_privateuseone_python_backend
    "prevent compile-time FakeTensor crashes".)
    """
    exclude_privateuse1 = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)
    original_new = FakeTensor.__new__

    def fake_new_without_mojo_kernels(
        cls: type[FakeTensor], *args: object, **kwargs: object
    ) -> FakeTensor:
        with torch._C._ExcludeDispatchKeyGuard(exclude_privateuse1):
            # Forwarded blindly (see version-compatibility note below): the
            # checker can't match object-typed *args/**kwargs against
            # __new__'s concrete parameter list.
            return original_new(
                cls,
                *args,  # ty: ignore[invalid-argument-type]
                **kwargs,
            )

    # Deliberate *args/**kwargs passthrough: FakeTensor.__new__'s real
    # signature is torch-version-dependent (AGENTS.md: support many
    # versions), so this wrapper forwards blindly rather than hardcoding it.
    FakeTensor.__new__ = staticmethod(  # ty: ignore[invalid-assignment]
        fake_new_without_mojo_kernels
    )


def fix_privateuse1_dlpack_device_type():
    """`Tensor.__dlpack_device__` doesn't recognize a *renamed* PrivateUse1
    backend.

    ``torch/_tensor.py``'s ``Tensor.__dlpack_device__`` maps a PrivateUse1
    tensor to DLPack's ``kDLExtDev`` by comparing ``self.device.type``
    against the string literal ``"privateuse1"`` -- so after
    ``torch.utils.rename_privateuse1_backend("mojo")`` it never matches, and
    every mojo tensor's ``__dlpack_device__()`` raises ``ValueError("Unknown
    device type mojo for Dlpack")``. Two other call sites in that very same
    file (the ``__cuda_array_interface__`` gate) correctly compare against
    ``torch._C._get_privateuse1_backend_name()`` instead of the literal;
    this one method just didn't get the memo.

    ``Tensor.__dlpack__`` itself (the capsule export) is unaffected --
    ATen's C++ DLConvertor keys off the ``DeviceType`` enum, not the
    Python-visible name -- so only the device-query half needs patching.
    ``torch_compile_backend/compiler.py``'s ``fast_from_dlpack`` routes
    around this bug for its own zero-copy exchange (it never calls
    ``__dlpack_device__``), but plain ``torch.utils.dlpack`` /
    ``max.driver.Buffer.from_dlpack(t)`` usage elsewhere (user code,
    ``test_compile_mojo_device.py``) goes through the single-arg DLPack
    protocol, which calls ``__dlpack_device__()`` first and needs this fix.
    """
    original = torch.Tensor.__dlpack_device__
    if getattr(original, "_torch_mojo_backend", False):
        return

    from torch.utils.dlpack import DLDeviceType  # noqa: PLC0415 -- mirrors the private import inside the method being patched

    @wraps(original)
    def __dlpack_device__(self: torch.Tensor) -> tuple[int, int]:
        if self.device.type == torch._C._get_privateuse1_backend_name():
            index = self.device.index if self.device.index is not None else 0
            return (DLDeviceType.kDLExtDev, index)
        return original(self)

    __dlpack_device__._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.Tensor.__dlpack_device__ = (  # ty: ignore[invalid-assignment]
        __dlpack_device__
    )


def fix_batch_isend_irecv_for_python_process_groups():
    """`batch_isend_irecv` never coalesces for a Python `ProcessGroup`
    subclass, and a bidirectional exchange then deadlocks.

    ``torch/distributed/distributed_c10d.py``'s ``batch_isend_irecv`` gates
    its NCCL-style coalescing on ``type(group) is ProcessGroup``. The mojo
    backend IS a Python subclass of ``ProcessGroup`` (it has to be: the
    C++ ``Backend`` cannot be implemented in Python), so the check is False
    however capable the backend is, and every operation in the list goes out
    as its own NCCL group. A two-rank exchange then deadlocks on the device:
    each rank's comm stream holds ``[send(->peer), recv(<-peer)]``, and the
    send cannot retire until the peer posts its recv, which sits behind that
    peer's own send. The very next line asks the backend whether it
    ``supports_coalescing``, which is the real question; ``isinstance`` is
    what the type check means. Everything else is torch's own code path,
    called unchanged.
    """
    original = c10d.batch_isend_irecv
    if getattr(original, "_torch_mojo_backend", False):
        return

    @wraps(original)
    def batch_isend_irecv(p2p_op_list: list[c10d.P2POp]) -> list[c10d.Work]:
        c10d._check_p2p_op_list(p2p_op_list)
        group = p2p_op_list[0].group or c10d._get_default_group()
        device = p2p_op_list[0].tensor.device
        coalesces = (
            type(group) is not torch.distributed.ProcessGroup
            and isinstance(group, torch.distributed.ProcessGroup)
            and group._get_backend(device).supports_coalescing
        )
        if not coalesces:
            return original(p2p_op_list)
        with c10d._coalescing_manager(group, device, async_ops=True) as manager:
            for op in p2p_op_list:
                peer = "group_dst" if op.op is c10d.isend else "group_src"
                op.op(op.tensor, group=op.group, tag=op.tag, **{peer: op.group_peer})
        return manager.works

    batch_isend_irecv._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    c10d.batch_isend_irecv = batch_isend_irecv  # ty: ignore[invalid-assignment]
    torch.distributed.batch_isend_irecv = batch_isend_irecv  # ty: ignore[invalid-assignment]


def apply_torch_monkeypatches(torch_mojo_device_module: ModuleType):
    """Install every PyTorch patch. Requires the ``mojo`` PrivateUse1 backend
    to be registered already (the accelerator patches check that it is the
    current accelerator). Idempotent: each patch detects its own sentinel or
    membership."""
    _install_torch_accelerator_synchronize(torch_mojo_device_module)
    _install_torch_accelerator_stream_api(torch_mojo_device_module)
    _install_torch_stream_event_dispatch()
    _declare_mojo_tensor_foreach_capable()
    _declare_mojo_tensor_as_plain_tensor()
    _trace_mojo_tensor_as_a_plain_tensor_in_dynamo()
    _keep_mojo_kernels_out_of_fake_tensor_construction()
