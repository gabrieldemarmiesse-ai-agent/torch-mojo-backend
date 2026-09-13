"""On-demand kernel builds, from Mojo.

A kernel family (torch_mojo_backend/eager_kernels/<family>/<family>.mojo)
exposes one C entry, `tmb_call`, and compiles exactly one (OP, dtypes, flags)
specialization per build, selected with -D defines. This loader is the
in-process registry of those builds: hash the family's import closure, look
the .so up in the cache, otherwise run `mojo build` in a subprocess (under a
cross-process flock, atomic rename), dlopen it, and cache the entry pointer.
"""
from std.builtin.sort import sort
from std.collections import Dict
from std.ffi import OwnedDLHandle, external_call
from std.os import getenv, makedirs
from std.os.path import exists, isdir
from std.pathlib import Path
from std.subprocess import run as run_command
from std.time import perf_counter_ns

from op_utils import Argv


comptime CACHE_ABI = "native-v1"


def _fnv1a(mut h: UInt64, bytes: Span[UInt8, _]):
    for i in range(len(bytes)):
        h ^= UInt64(bytes[i])
        h *= 1099511628211


def _hex(h: UInt64) -> String:
    return String(hex(h)[byte=2:])


def _slug(defines: List[String]) -> String:
    var h: UInt64 = 14695981039346656037
    for d in defines:
        _fnv1a(h, d.as_bytes())
        _fnv1a(h, "\n".as_bytes())
    return String(_hex(h)[byte=:12])


struct Family(Movable):
    var lib: OwnedDLHandle
    var entry: Int  # address of tmb_call

    def __init__(out self, path: String) raises:
        self.lib = OwnedDLHandle(path)
        var sym = self.lib.get_symbol[NoneType]("tmb_call")
        if not sym:
            raise Error("tmb_call not exported by ", path)
        self.entry = Int(sym.value())


struct Loader(Movable):
    var kernels_dir: String  # torch_mojo_backend/eager_kernels
    var cache_dir: String  # <kernels_dir>/__mojocache__/native
    var mojo_exe: String
    var toolchain: String  # versions of mojo/max/python, from the Python side
    var trace: Bool
    var families: Dict[String, Family]  # "<family>.<slug>" -> loaded build
    var failures: Dict[String, String]  # same key -> permanent error
    var source_hashes: Dict[String, String]  # family -> closure hash
    var fast: Dict[
        UInt64, Int
    ]  # (family, defines) hash -> entry address (hot path)

    def __init__(
        out self,
        kernels_dir: String,
        cache_dir: String,
        mojo_exe: String,
        toolchain: String,
        trace: Bool,
    ):
        self.kernels_dir = kernels_dir
        self.cache_dir = cache_dir
        self.mojo_exe = mojo_exe
        self.toolchain = toolchain
        self.trace = trace
        self.families = Dict[String, Family]()
        self.failures = Dict[String, String]()
        self.source_hashes = Dict[String, String]()
        self.fast = Dict[UInt64, Int]()

    def _closure(self, family: String) raises -> List[String]:
        """Every .mojo file the family's entry file reaches through
        `from X import` / `import X` (resolved in the family dir, then the
        package root), plus every op_utils/*.mojo, in a deterministic order."""
        var fam_dir = self.kernels_dir + "/" + family
        var files = List[String]()
        var seen = Dict[String, Bool]()
        var todo = List[String]()
        todo.append(fam_dir + "/" + family + ".mojo")
        while len(todo) > 0:
            var f = todo.pop()
            if f in seen:
                continue
            seen[f] = True
            files.append(f)
            var text = Path(f).read_text()
            for line in text.splitlines():
                var s = String(line)
                var name = String()
                if s.startswith("from "):
                    var rest = String(s[byte=5:])
                    var sp = rest.find(" ")
                    name = String(rest[byte=:sp]) if sp > 0 else rest
                elif s.startswith("import "):
                    name = String(String(s[byte=7:]).strip())
                else:
                    continue
                var dot = name.find(".")
                if dot > 0:
                    var short = String(name[byte=:dot])
                    name = short^
                if (
                    name == ""
                    or name == "std"
                    or name == "max"
                    or name == "nn"
                    or name == "linalg"
                    or name == "layout"
                ):
                    continue
                var cand = fam_dir + "/" + name + ".mojo"
                if not exists(cand):
                    cand = self.kernels_dir + "/" + name + ".mojo"
                if exists(cand) and cand not in seen:
                    todo.append(cand)
        var op_utils = self.kernels_dir + "/op_utils"
        if isdir(op_utils):
            var names = List[String]()
            for p in Path(op_utils).listdir():
                var ps = String(p)
                if ps.endswith(".mojo"):
                    names.append(op_utils + "/" + ps)
            sort(names)
            for n in names:
                if n not in seen:
                    files.append(n)
        sort(files)
        return files^

    def source_hash(mut self, family: String) raises -> String:
        if family in self.source_hashes:
            return self.source_hashes[family]
        var h: UInt64 = 14695981039346656037
        _fnv1a(h, CACHE_ABI.as_bytes())
        _fnv1a(h, self.toolchain.as_bytes())
        for f in self._closure(family):
            var rel = String(f[byte = self.kernels_dir.byte_length() :])
            _fnv1a(h, rel.as_bytes())
            var bytes = Path(f).read_bytes()
            _fnv1a(h, Span(bytes))
        var out = _hex(h)
        self.source_hashes[family] = out
        return out

    def _build(
        mut self, family: String, defines: List[String], out_path: String
    ) raises:
        var fam_dir = self.kernels_dir + "/" + family
        var src = fam_dir + "/" + family + ".mojo"
        var tmp = out_path + ".tmp" + String(perf_counter_ns())
        # The compiler writes to local scratch (the cache may be on NFS,
        # where its intermediate archive went missing under load); the
        # finished library is then moved next to its final name.
        var scratch = String("${TMPDIR:-/tmp}/torch-mojo-backend-") + String(
            external_call["getuid", UInt32]()
        )
        var local = (
            scratch + "/" + family + "." + String(perf_counter_ns()) + ".so"
        )
        var cmd = (
            String("mkdir -p '")
            + scratch
            + "' && '"
            + self.mojo_exe
            + "' build '"
            + src
            + "' --emit shared-lib -I '"
            + fam_dir
            + "' -I '"
            + self.kernels_dir
            + "'"
        )
        for d in defines:
            cmd += " -D " + d
        cmd += (
            " -o '"
            + local
            + "' 2>&1 && mv -f '"
            + local
            + "' '"
            + tmp
            + "' 2>&1; echo __TMB_RC=$?"
        )
        var t0 = perf_counter_ns()
        var output = run_command(cmd)
        var ms = (perf_counter_ns() - t0) // 1_000_000
        var marker = output.rfind("__TMB_RC=")
        var rc = -1
        if marker >= 0:
            rc = Int(atol(String(String(output[byte = marker + 9 :]).strip())))
        if rc != 0:
            if exists(tmp):
                _ = external_call["unlink", Int32](
                    tmp.as_c_string_slice().unsafe_ptr()
                )
            raise Error(
                "mojo build of ",
                family,
                " failed (rc ",
                rc,
                ", ",
                ms,
                " ms):\n",
                String(output[byte=:marker]) if marker > 0 else output,
            )
        if exists(
            out_path
        ):  # another process installed the same build first: use theirs
            _ = external_call["unlink", Int32](
                tmp.as_c_string_slice().unsafe_ptr()
            )
        else:
            var dst = String(out_path)
            var r = external_call["rename", Int32](
                tmp.as_c_string_slice().unsafe_ptr(),
                dst.as_c_string_slice().unsafe_ptr(),
            )
            if r != 0 and not exists(out_path):
                raise Error("could not install ", out_path)
        if self.trace:
            print(
                "[TRACE] built ",
                family,
                " ",
                " ".join(defines),
                " in ",
                Float64(ms) / 1000.0,
                "s",
            )

    def entry(mut self, family: String, defines: List[String]) raises -> Int:
        """Address of the family's `tmb_call` for this specialization."""
        var key = family + "." + _slug(defines)
        if key in self.families:
            return self.families[key].entry
        if key in self.failures:
            raise Error(self.failures[key])
        try:
            var so = (
                self.cache_dir
                + "/"
                + key
                + ".hash-"
                + self.source_hash(family)
                + ".so"
            )
            if not exists(so):
                if not isdir(self.cache_dir):
                    makedirs(self.cache_dir, exist_ok=True)
                var lock_path = self.cache_dir + "/." + key + ".lock"
                var fd = external_call["creat", Int32](
                    lock_path.as_c_string_slice().unsafe_ptr(), Int32(0o644)
                )
                if fd >= 0:
                    _ = external_call["flock", Int32](
                        fd, Int32(2)
                    )  # LOCK_EX (best effort: NFS may refuse)

                try:
                    if not exists(so):
                        self._build(family, defines, so)
                finally:
                    if fd >= 0:
                        _ = external_call["flock", Int32](
                            fd, Int32(8)
                        )  # LOCK_UN
                        _ = external_call["close", Int32](fd)
            var fam = Family(so)
            var addr = fam.entry
            self.families[key] = fam^
            return addr
        except e:
            self.failures[key] = String(e)
            raise e^


comptime ERR_CAP = 4096
comptime FamilyFn = def(
    Argv, Int, Pointer[UInt8, MutUntrackedOrigin], Int
) thin abi("C") -> Int32


def call_family(
    mut loader: Loader,
    family: String,
    key: UInt64,
    defines: List[String],
    argv: Argv,
    argc: Int,
) raises:
    """Run one kernel: build/load the specialization on first use, then call
    its C entry with the argument slots. `key` hashes (family, defines) so a
    warm call is one dictionary probe; `defines` is only read on a miss.
    A non-zero return carries the kernel's own message (declined input, bad
    geometry, ...)."""
    var entry: Int
    var hit = loader.fast.find(key)
    if hit:
        entry = hit.value()
    else:
        entry = loader.entry(family, defines)
        loader.fast[key] = entry
    var err = InlineArray[UInt8, ERR_CAP](fill=0)
    var f = Pointer(to=entry).unsafe_bitcast[FamilyFn]()[]
    var rc = f(
        argv,
        argc,
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(err.unsafe_ptr())
        ),
        ERR_CAP,
    )
    if rc != 0:
        raise Error(String(unsafe_from_utf8_ptr=err.unsafe_ptr()))
