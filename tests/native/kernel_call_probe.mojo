from std.testing import assert_equal, assert_true
from std.utils import IndexList
from kernels import Defines, KernelCall, MAX_CALL_SPECS, TUPLE_POOL_WORDS
from op_utils import MAX_RANK, TensorSpec


def main() raises:
    var defines = Defines("Cast")
    defines.arg(12, DType.float32)
    defines.arg(0, DType.bfloat16)
    defines.out(DType.float16)
    defines.out_i(2, DType.int64)
    defines.flag("SIGNED", -7)
    var expected = List[String]()
    expected.append("DTYPE_ARG_0=bfloat16")
    expected.append("DTYPE_ARG_12=float32")
    expected.append("DTYPE_OUT=float16")
    expected.append("DTYPE_OUT_2=int64")
    expected.append("OP=Cast")
    expected.append("SIGNED=-7")
    assert_equal(defines.sorted(), expected)
    # A hot-path key must distinguish dtype, output index, and flag value.
    var dtype_a = Defines("Cast")
    var dtype_b = Defines("Cast")
    dtype_a.arg(12, DType.float32)
    dtype_b.arg(12, DType.float16)
    assert_true(dtype_a.key != dtype_b.key)
    var output_a = Defines("Cast")
    var output_b = Defines("Cast")
    output_a.out_i(0, DType.float32)
    output_b.out_i(1, DType.float32)
    assert_true(output_a.key != output_b.key)
    var flag_a = Defines("Cast")
    var flag_b = Defines("Cast")
    flag_a.flag("SIGNED", -7)
    flag_b.flag("SIGNED", 7)
    assert_true(flag_a.key != flag_b.key)

    var call = KernelCall("data_movement_ops", "Cast")
    call.int(42)
    assert_equal(call.nspecs, 0)
    for i in range(MAX_CALL_SPECS):
        call.spec(
            TensorSpec(
                i,
                1,
                IndexList[MAX_RANK](1),
                IndexList[MAX_RANK](0),
                0,
                DType.float32,
                4,
                1,
                True,
                0,
            )
        )
    # Every exposed pointer must remain valid after all subsequent appends.
    for i in range(MAX_CALL_SPECS):
        var p = Pointer[TensorSpec, MutUntrackedOrigin](
            unsafe_from_address=call.slots[i + 1]
        )
        assert_equal(p[].ptr, i)
        assert_equal(p[].dtype, DType.float32)
    # One spec too many is reported by run(), not by the builder.
    assert_equal(call.defines.bad, "")
    call.spec(
        TensorSpec(
            0,
            1,
            IndexList[MAX_RANK](1),
            IndexList[MAX_RANK](0),
            0,
            DType.float32,
            4,
            1,
            True,
            0,
        )
    )
    assert_true(call.defines.bad != "")

    # Tuple slots read back as `[len, e0, ...]`, from the inline pool and
    # from the heap spill a tuple longer than the pool takes.
    var tuples = KernelCall("data_movement_ops", "Cast")
    var small = List[Int]()
    small.append(3)
    small.append(5)
    small.append(7)
    var big = List[Int]()
    for i in range(TUPLE_POOL_WORDS + 5):
        big.append(i)
    tuples.tuple(small)
    tuples.tuple(big)
    tuples.tuple(small)
    assert_equal(tuples.defines.bad, "")
    for slot in range(3):
        ref expected = big if slot == 1 else small
        var p = Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=tuples.slots[slot]
        )
        assert_equal(p[], len(expected))
        for i in range(len(expected)):
            assert_equal(p[unsafe_offset=i + 1], expected[i])
