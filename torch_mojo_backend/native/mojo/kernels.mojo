"""The process-wide kernel loader and the spec-op calling convention.

`KernelCall(family, op)` is what an op uses to run one kernel: it derives the
specialization defines the way the Python loader did (OP, DTYPE_ARG_i,
DTYPE_OUT, flags), builds the argument slots and calls the family's C entry
through the loader.

Nothing on the warm path allocates or formats a string: the defines are kept
as PODs beside a running hash and only spelled out on a cache miss, and every
buffer the call needs lives inline in the struct.
"""
from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import unsafe_alloc

from abi import dtype_code
from loader import Loader, invoke_family
from op_utils import Argv, TensorSpec, _f64_slot

comptime LOADER_GLOBAL = "TMB_NATIVE_LOADER"


def init_loader(
    kernels_dir: String,
    mojo_dir: String,
    cache_dir: String,
    mojo_exe: String,
    toolchain: String,
    trace: Bool,
):
    if _get_global_or_null(LOADER_GLOBAL):
        return
    var box = unsafe_alloc[Loader](1)
    box.unsafe_write(
        Loader(kernels_dir, mojo_dir, cache_dir, mojo_exe, toolchain, trace)
    )
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(LOADER_GLOBAL), box.unsafe_bitcast[NoneType]()
    )


@always_inline
def loader() -> Pointer[Loader, MutUntrackedOrigin]:
    return _get_global_or_null(LOADER_GLOBAL).value().unsafe_bitcast[Loader]()


def dtype_name(dt: DType) -> String:
    """The `max.dtype.DType.name` spelling the gates compare against."""
    if dt == DType.float32:
        return "float32"
    if dt == DType.bfloat16:
        return "bfloat16"
    if dt == DType.float16:
        return "float16"
    if dt == DType.float64:
        return "float64"
    if dt == DType.int64:
        return "int64"
    if dt == DType.int32:
        return "int32"
    if dt == DType.int16:
        return "int16"
    if dt == DType.int8:
        return "int8"
    if dt == DType.uint8:
        return "uint8"
    if dt == DType.uint16:
        return "uint16"
    if dt == DType.uint32:
        return "uint32"
    if dt == DType.uint64:
        return "uint64"
    if dt == DType.bool:
        return "bool"
    return String(dt)


@always_inline
def _mix(mut h: UInt64, x: UInt64):
    h ^= x
    h *= 1099511628211


comptime NAME_WORDS = 6
comptime NAME_CAP = NAME_WORDS * 8  # bytes, NUL included


struct Name(Copyable, Movable):
    """A family or OP name, copied inline.

    Held as bytes rather than as a `String` so a call allocates nothing: the
    name itself is only needed again on a cache miss, and the two longest
    family names already exceed the 23 bytes a `String` keeps inline.
    """

    # 64-bit words, so the hash below can read them eight bytes at a time;
    # a byte array would not be aligned for that.
    var words: InlineArray[UInt64, NAME_WORDS]  # NUL-terminated bytes
    var count: Int
    var hash: UInt64
    var whole: Bool  # False if the name did not fit; `run()` reports it

    def __init__(out self, s: StringSlice):
        var b = s.as_bytes()
        var n = min(len(b), NAME_CAP - 1)
        self.words = InlineArray[UInt64, NAME_WORDS](uninitialized=True)
        # `used` covers the NUL at index n; zeroing first makes the padding
        # after it deterministic, which the word-wise hash relies on.
        var used = n // 8 + 1
        var w = self.words.unsafe_ptr()
        for i in range(used):
            w[unsafe_offset=i] = 0
        var p = w.unsafe_bitcast[UInt8]()
        for i in range(n):
            p[unsafe_offset=i] = b[i]
        var h = UInt64(14695981039346656037)
        for i in range(used):
            _mix(h, w[unsafe_offset=i])
        self.count = n
        self.hash = h
        self.whole = n == len(b)

    def text(self) -> String:
        """The name as a `String` — miss path only."""
        return String(
            unsafe_from_utf8_ptr=self.words.unsafe_ptr().unsafe_bitcast[UInt8]()
        )


# Define kinds. `OP` is not one of them: it is always present and lives in
# `Defines.op`.
comptime DEF_ARG = 1
comptime DEF_OUT = 2
comptime DEF_OUT_I = 3
comptime DEF_FLAG = 4

comptime MAX_DEFINES = 12


@fieldwise_init
struct Define(Copyable, Movable):
    """One `-D` entry as plain data. `name` is only read for `DEF_FLAG`, and
    every flag name in the tree is a literal, so nothing here owns memory."""

    var kind: Int
    var index: Int
    var dtype: DType
    var value: Int
    var name: StaticString


struct Defines(Movable):
    """The -D set of one specialization. `key` is a running hash of the same
    information so the loader's hot path never builds a string: the strings
    are only produced on a miss (`sorted()`).

    A builder never raises — the ~200 call sites include non-raising
    helpers — so anything that does not fit sets `bad` and `KernelCall.run()`
    reports it rather than launching a half-described kernel.
    """

    var op: Name
    var items: InlineArray[Define, MAX_DEFINES]
    var count: Int
    var key: UInt64
    var bad: StaticString

    def __init__(out self, op: StringSlice):
        self.op = Name(op)
        self.items = InlineArray[Define, MAX_DEFINES](uninitialized=True)
        self.count = 0
        self.key = self.op.hash
        self.bad = "" if self.op.whole else "OP name too long"

    @always_inline
    def _add(mut self, var d: Define):
        if self.count >= MAX_DEFINES:
            self.bad = "too many defines in one kernel call"
            return
        self.items[self.count] = d^
        self.count += 1

    def arg(mut self, i: Int, dt: DType):
        self._add(Define(DEF_ARG, i, dt, 0, ""))
        _mix(self.key, UInt64(0x1000 + i))
        _mix(self.key, UInt64(dtype_code(dt)))

    def out(mut self, dt: DType):
        self._add(Define(DEF_OUT, 0, dt, 0, ""))
        _mix(self.key, UInt64(0x2000))
        _mix(self.key, UInt64(dtype_code(dt)))

    def out_i(mut self, i: Int, dt: DType):
        self._add(Define(DEF_OUT_I, i, dt, 0, ""))
        _mix(self.key, UInt64(0x3000 + i))
        _mix(self.key, UInt64(dtype_code(dt)))

    def flag(mut self, name: StaticString, value: Int):
        # The dtype slot is unread for a flag.
        self._add(Define(DEF_FLAG, 0, DType.bool, value, name))
        _mix(self.key, UInt64(0x4000))
        for b in name.as_bytes():
            _mix(self.key, UInt64(b))
        _mix(self.key, UInt64(value))

    def sorted(self) -> List[String]:
        """The `-D` strings in the canonical (sorted) order the .so cache key
        uses — built on a miss only."""
        var out = List[String](capacity=self.count + 1)
        out.append("OP=" + self.op.text())
        for i in range(self.count):
            ref d = self.items[i]
            if d.kind == DEF_ARG:
                out.append(
                    "DTYPE_ARG_" + String(d.index) + "=" + dtype_name(d.dtype)
                )
            elif d.kind == DEF_OUT:
                out.append("DTYPE_OUT=" + dtype_name(d.dtype))
            elif d.kind == DEF_OUT_I:
                out.append(
                    "DTYPE_OUT_" + String(d.index) + "=" + dtype_name(d.dtype)
                )
            else:
                out.append(String(d.name) + "=" + String(d.value))
        sort(out)
        return out^


comptime MAX_CALL_SPECS = 16
# One more than the widest `_spec_dispatcher`: an argument list longer than
# that cannot be read back by any family.
comptime MAX_CALL_SLOTS = 17
# Words for the `[len, e0, ...]` tuple slots of one call. Anything longer
# (a foreach launch over many tensors) spills to the heap.
comptime TUPLE_POOL_WORDS = 96


struct KernelCall(Movable):
    """One kernel invocation: the specialization defines plus the argument
    slots, and the storage behind them (TensorSpecs, tuples) that must outlive
    the call. Mojo destroys a local right after its last use, so a spec whose
    address was taken and then handed over as an Int would be gone by the
    time the kernel reads it; owning everything here and calling `run()` on
    the struct keeps every address valid for exactly the call.

    Every buffer is inline, so a warm call allocates nothing; the capacities
    are therefore fixed, and a builder that does not fit records the reason
    for `run()` to raise instead of raising itself (the builders are called
    from non-raising helpers).
    """

    var family: Name
    var defines: Defines
    var specs: InlineArray[TensorSpec, MAX_CALL_SPECS]
    var nspecs: Int
    var pool: InlineArray[Int, TUPLE_POOL_WORDS]
    var npool: Int
    var spill: List[List[Int]]  # tuples too long for `pool`
    var slots: InlineArray[Int, MAX_CALL_SLOTS]
    var nslots: Int

    def __init__(out self, family: StringSlice, op: StringSlice):
        self.family = Name(family)
        self.defines = Defines(op)
        self.specs = InlineArray[TensorSpec, MAX_CALL_SPECS](uninitialized=True)
        self.nspecs = 0
        self.pool = InlineArray[Int, TUPLE_POOL_WORDS](uninitialized=True)
        self.npool = 0
        self.spill = List[List[Int]]()
        self.slots = InlineArray[Int, MAX_CALL_SLOTS](uninitialized=True)
        self.nslots = 0
        if not self.family.whole:
            self.defines.bad = "family name too long"
        _mix(self.defines.key, self.family.hash)

    def arg_dtype(mut self, i: Int, dt: DType):
        self.defines.arg(i, dt)

    def out_dtype(mut self, dt: DType):
        self.defines.out(dt)

    def out_dtype_i(mut self, i: Int, dt: DType):
        self.defines.out_i(i, dt)

    def flag(mut self, name: StaticString, value: Int):
        self.defines.flag(name, value)

    @always_inline
    def _slot(mut self, v: Int):
        if self.nslots >= MAX_CALL_SLOTS:
            self.defines.bad = "too many arguments in one kernel call"
            return
        self.slots[self.nslots] = v
        self.nslots += 1

    def spec(mut self, var s: TensorSpec):
        if self.nspecs >= MAX_CALL_SPECS:
            self.defines.bad = "too many spec arguments in one kernel call"
            return
        var at = self.specs.unsafe_ptr().unsafe_offset(self.nspecs)
        at.unsafe_write(s^)
        self.nspecs += 1
        self._slot(Int(at))

    def int(mut self, v: Int):
        self._slot(v)

    def f64(mut self, v: Float64):
        self._slot(_f64_slot(v))

    def tuple(mut self, values: List[Int]):
        """A `[len, e0, e1, ...]` tuple slot (the kernels read it with _raw_tuple_*).
        """
        var words = len(values) + 1
        if self.npool + words <= TUPLE_POOL_WORDS:
            var at = self.pool.unsafe_ptr().unsafe_offset(self.npool)
            at[] = len(values)
            for i in range(len(values)):
                at[unsafe_offset=i + 1] = values[i]
            self.npool += words
            self._slot(Int(at))
            return
        var t = List[Int](capacity=words)
        t.append(len(values))
        for v in values:
            t.append(v)
        self.spill.append(t^)
        self._slot(Int(self.spill[len(self.spill) - 1].unsafe_ptr()))

    def run(self) raises:
        if self.defines.bad:
            raise Error(self.defines.bad, " (", self.family.text(), ")")
        var l = loader()
        var entry: Int
        var hit = l[].fast.find(self.defines.key)
        if hit:
            entry = hit.value()
        else:
            entry = l[].entry(self.family.text(), self.defines.sorted())
            l[].fast[self.defines.key] = entry
        invoke_family(
            entry,
            Argv(unsafe_from_address=Int(self.slots.unsafe_ptr())),
            self.nslots,
        )
