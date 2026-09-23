"""Fixed-width random-kernel metadata with a supported Metal launch ABI.

Bare SIMD kernel arguments (even inside a register-passable struct) retain
invalid vector argument metadata in AIR. InlineArray encodes its indices as
an aggregate, like the existing Metal foreach launch metadata.
"""

comptime I64x8 = InlineArray[Int64, 8]
