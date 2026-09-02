from collections.abc import Callable
from functools import wraps
from types import ModuleType
from typing import TypeVar

import torch

from torch_mojo_backend.mojo_device.mojo_device_aten_ops import _aten_ops_registry

_T = TypeVar("_T")
_registered = False


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

    # Deliberate monkeypatch of torch.accelerator.synchronize: the sentinel
    # attribute is how this function detects, on a later call, that its own
    # replacement (not the original) is already installed. ty can't type an
    # ad hoc attribute on a stdlib callable.
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

    # Same deliberate monkeypatch and sentinel as `synchronize` above.
    current_stream._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.accelerator.current_stream = current_stream  # ty: ignore[invalid-assignment]

    original_set_stream = torch.accelerator.set_stream

    @wraps(original_set_stream)
    def set_stream(stream: torch.Stream):
        from torch_mojo_backend.mojo_device.streams import Stream as MojoStream

        if isinstance(stream, MojoStream):
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
    from torch_mojo_backend.mojo_device import streams as mojo_streams

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

    # Deliberate replacement of torch's classes by the dispatching wrappers.
    torch.Stream = Stream  # ty: ignore[invalid-assignment]
    torch.Event = Event  # ty: ignore[invalid-assignment]


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
    from collections.abc import Mapping, Sequence

    import torch._functorch._aot_autograd.runtime_wrappers as runtime_wrappers
    import torch._subclasses.fake_tensor as fake_tensor_module
    import torch.fx.experimental.proxy_tensor as proxy_tensor
    from torch._subclasses.functional_tensor import (
        FunctionalTensor,
        FunctionalTensorMode,
    )

    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    if TorchMojoTensor not in proxy_tensor.HANDLED_TYPES:
        # Deliberate monkeypatch: torch infers HANDLED_TYPES' type from its
        # own fixed-length tuple literal, so extending it by one element is
        # always a "wrong length" mismatch to the checker.
        proxy_tensor.HANDLED_TYPES = (  # ty: ignore[invalid-assignment]
            *proxy_tensor.HANDLED_TYPES,
            TorchMojoTensor,
        )
    runtime_wrappers.HANDLED_TYPES = proxy_tensor.HANDLED_TYPES

    original_check = fake_tensor_module._check_for_subclass_arg

    def check_for_subclass_arg_except_mojo(x: object) -> bool:
        return original_check(x) and not isinstance(x, TorchMojoTensor)

    # Deliberate monkeypatch: ty treats each `def` as its own nominal type
    # even with an identical signature, so this never structurally matches
    # the attribute it replaces.
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
    from torch._dynamo.variables.builder import VariableBuilder

    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

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
    from torch._subclasses.fake_tensor import FakeTensor

    exclude_privateuse1 = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)
    original_new = FakeTensor.__new__

    def fake_new_without_mojo_kernels(
        cls: type[FakeTensor], *args: object, **kwargs: object
    ) -> FakeTensor:
        with torch._C._ExcludeDispatchKeyGuard(exclude_privateuse1):
            # Forwarded blindly (see version-compatibility note above): the
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


def register_mojo_devices():
    """Enable the mojo device globally and register all aten ops"""
    from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

    from torch_mojo_backend.mojo_device import torch_mojo_device_module

    # since it's so recent we import it here.
    global _registered
    if _registered:
        # Already registered
        return

    # Module._apply otherwise replaces a shared CPU Parameter independently in
    # each child module when its converted tensor has a different subclass.
    # Swapping preserves tied weights (for example GPT-2's token embedding and
    # lm_head) as one Parameter and one Mojo allocation.
    torch.__future__.set_swap_module_params_on_conversion(True)

    _setup_privateuseone_for_python_backend(
        "mojo", backend_module=torch_mojo_device_module
    )
    _install_torch_accelerator_synchronize(torch_mojo_device_module)
    _install_torch_accelerator_stream_api(torch_mojo_device_module)
    _install_torch_stream_event_dispatch()

    # PyTorch deliberately uses exact-type checks before selecting foreach
    # optimizers. TorchMojoTensor is a transparent PrivateUse1 wrapper, so
    # opt it into the same lists as DTensor and other supported tensor types.
    import torch.optim.optimizer as optimizer_module
    import torch.utils._foreach_utils as foreach_utils

    from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

    for supported_types in (
        optimizer_module._foreach_supported_types,
        foreach_utils._foreach_supported_types,
    ):
        if TorchMojoTensor not in supported_types:
            # Deliberate monkeypatch: torch declared these lists' element
            # type from their own literal contents (Tensor/Parameter), which
            # doesn't include our subclass by construction.
            supported_types.append(TorchMojoTensor)  # ty: ignore[invalid-argument-type]

    # Register all collected aten operations
    for op_name, func in _aten_ops_registry:
        torch.library.impl(op_name, "privateuseone")(func)

    from torch_mojo_backend.mojo_device.mojo_device_autograd import (
        register_autograd_ops,
    )

    register_autograd_ops()

    from torch_mojo_backend.mojo_device.mojo_device_autocast import (
        register_autocast_ops,
    )

    register_autocast_ops()

    _declare_mojo_tensor_as_plain_tensor()
    _trace_mojo_tensor_as_a_plain_tensor_in_dynamo()
    _keep_mojo_kernels_out_of_fake_tensor_construction()

    from torch_mojo_backend.mojo_device.apple_optimizations import (
        register_apple_optimizations,
    )

    register_apple_optimizations()

    _registered = True
