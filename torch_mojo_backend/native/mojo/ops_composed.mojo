"""Ops with no kernel of their own composed from registered ops through the
dispatcher (`call_op`): backward formulas ATen only ships as per-backend
kernels (threshold / sigmoid / tanh backward), the signed-infinity tests, and
out= overloads of ops whose functional form exists. Each costs a few extra
launches; a fused kernel can replace any of them later without changing the
registration."""
from abi import (
    ST_BOOL,
    T,
    TAG_NONE,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    Value,
    Values,
    own,
    release,
    ret_owned,
    ret_ref,
    new_tensor,
    v_tensor,
)
from ops_common import call_op, fill_value, resize_out
from registry import Lib, impl


comptime NEG_INF_BITS: Int64 = -4503599627370496  # 0xFFF0000000000000
comptime POS_INF_BITS: Int64 = 9218868437227405312  # 0x7FF0000000000000


def _tensor_value(t: T) -> Value:
    return Value(TAG_TENSOR, 0, Int64(t.h), 0)


def _dispatch(
    name: StaticString, overload: StaticString, var args: List[Value]
) raises -> T:
    """One aten op through the dispatcher, one Tensor result (owned)."""
    var rets = call_op(String(name), String(overload), args^, 1)
    return T(Int(rets[0].a))


def _dispatch_into(
    name: StaticString, overload: StaticString, var args: List[Value], target: T
) raises:
    """An out= overload through the dispatcher; its result handle is a fresh
    reference to `out` and is released here."""
    var rets = call_op(String(name), String(overload), args^, 1)
    if rets[0].tag == TAG_TENSOR:
        release(Int(rets[0].a))
    _ = target


def _one_minus(x: T) raises -> T:
    """1 - x as neg(x - 1): sub.Scalar then neg, both registered."""
    var xm1 = _dispatch(
        "aten::sub",
        "Scalar",
        [
            _tensor_value(x),
            Value(TAG_SCALAR_INT, 0, 1, 0),
            Value(TAG_SCALAR_INT, 0, 1, 0),
        ],
    )
    var r = _dispatch("aten::neg", "", [_tensor_value(xm1)])
    release(xm1.h)
    return r^


# aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
def _threshold_backward_mask(args: Values) raises -> T:
    return _dispatch(
        "aten::gt",
        "Scalar",
        [
            _tensor_value(v_tensor(args[unsafe_offset=1])),
            args[unsafe_offset=2].copy(),
        ],
    )


def op_threshold_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var mask = _threshold_backward_mask(args)
    var g = own(
        _dispatch(
            "aten::mul", "Tensor", [_tensor_value(grad), _tensor_value(mask)]
        )
    )
    release(mask.h)
    ret_owned(rets, 0, g)


# aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input)
def op_threshold_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=3])
    var mask = _threshold_backward_mask(args)
    _dispatch_into(
        "aten::mul",
        "out",
        [_tensor_value(grad), _tensor_value(mask), _tensor_value(out)],
        out,
    )
    release(mask.h)
    ret_ref(rets, 0, out)


# aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor: grad * out * (1 - out)
def _sigmoid_backward_factor(output: T) raises -> T:
    var om = _one_minus(output)
    var f = _dispatch(
        "aten::mul", "Tensor", [_tensor_value(output), _tensor_value(om)]
    )
    release(om.h)
    return f^


def op_sigmoid_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var f = _sigmoid_backward_factor(v_tensor(args[unsafe_offset=1]))
    var g = own(
        _dispatch(
            "aten::mul", "Tensor", [_tensor_value(grad), _tensor_value(f)]
        )
    )
    release(f.h)
    ret_owned(rets, 0, g)


# aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input)
def op_sigmoid_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=2])
    var f = _sigmoid_backward_factor(v_tensor(args[unsafe_offset=1]))
    _dispatch_into(
        "aten::mul",
        "out",
        [_tensor_value(grad), _tensor_value(f), _tensor_value(out)],
        out,
    )
    release(f.h)
    ret_ref(rets, 0, out)


# aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor: grad * (1 - out^2)
def _tanh_backward_factor(output: T) raises -> T:
    var sq = _dispatch(
        "aten::mul", "Tensor", [_tensor_value(output), _tensor_value(output)]
    )
    var f = _one_minus(sq)
    release(sq.h)
    return f^


def op_tanh_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var f = _tanh_backward_factor(v_tensor(args[unsafe_offset=1]))
    var g = own(
        _dispatch(
            "aten::mul", "Tensor", [_tensor_value(grad), _tensor_value(f)]
        )
    )
    release(f.h)
    ret_owned(rets, 0, g)


# aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input)
def op_tanh_backward_grad_input(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var out = v_tensor(args[unsafe_offset=2])
    var f = _tanh_backward_factor(v_tensor(args[unsafe_offset=1]))
    _dispatch_into(
        "aten::mul",
        "out",
        [_tensor_value(grad), _tensor_value(f), _tensor_value(out)],
        out,
    )
    release(f.h)
    ret_ref(rets, 0, out)


# aten::isneginf(Tensor self) -> Tensor / aten::isposinf(Tensor self) -> Tensor (+ .out):
# eq.Scalar against the signed infinity (an integer tensor is never infinite:
# eq against a double that no integer equals is all-false, like ATen).
def _is_inf(args: Values, rets: Values, bits: Int64, with_out: Bool) raises:
    var x = v_tensor(args[unsafe_offset=0])
    var scalar = Value(TAG_SCALAR_DOUBLE, 0, bits, 0)
    if with_out:
        var out = v_tensor(args[unsafe_offset=1])
        if not x.dtype.is_floating_point():
            resize_out(out, x.shape, x.rank)
            fill_value(out, 0.0)
        else:
            _dispatch_into(
                "aten::eq",
                "Scalar_out",
                [_tensor_value(x), scalar.copy(), _tensor_value(out)],
                out,
            )
        ret_ref(rets, 0, out)
        return
    if not x.dtype.is_floating_point():
        var zeros = own(new_tensor(x.shape, x.rank, ST_BOOL, x.device))
        fill_value(zeros.t, 0.0)
        ret_owned(rets, 0, zeros)
        return
    var r = own(
        _dispatch("aten::eq", "Scalar", [_tensor_value(x), scalar.copy()])
    )
    ret_owned(rets, 0, r)


def op_isneginf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _is_inf(args, rets, NEG_INF_BITS, False)


def op_isneginf_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _is_inf(args, rets, NEG_INF_BITS, True)


def op_isposinf(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _is_inf(args, rets, POS_INF_BITS, False)


def op_isposinf_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    _is_inf(args, rets, POS_INF_BITS, True)


def register_composed(lib: Lib) raises:
    impl[op_threshold_backward](lib, "threshold_backward")
    impl[op_threshold_backward_grad_input](lib, "threshold_backward.grad_input")
    impl[op_sigmoid_backward](lib, "sigmoid_backward")
    impl[op_sigmoid_backward_grad_input](lib, "sigmoid_backward.grad_input")
    impl[op_tanh_backward](lib, "tanh_backward")
    impl[op_tanh_backward_grad_input](lib, "tanh_backward.grad_input")
    impl[op_isneginf](lib, "isneginf")
    impl[op_isneginf_out](lib, "isneginf.out")
    impl[op_isposinf](lib, "isposinf")
    impl[op_isposinf_out](lib, "isposinf.out")
