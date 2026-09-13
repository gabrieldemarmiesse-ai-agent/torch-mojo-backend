"""The native mojo device: a PrivateUse1 backend whose ops run in Mojo.

`register()` builds (once per torch/toolchain version, cached) and loads two
shared libraries:

* the C++ shim (csrc/): the c10 objects torch only accepts as C++ classes —
  allocator, device guard, hooks, generator, profiler stubs, autocast — and a
  boxed-kernel adapter that hands each op call to a C function;
* the Mojo backend (mojo/): device/stream/event management over MAX, the
  aten op implementations, and the on-demand kernel builds.

Nothing on the op path goes through Python.
"""

from __future__ import annotations

import contextlib
import ctypes
import re
import fcntl
import functools
import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import torch

_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent
_KERNELS_DIR = _PACKAGE / "eager_kernels"
# One cache for every checkout on a box (contents-addressed: every build is
# keyed by its sources and toolchain), overridable for shared scratch space.
_CACHE_DIR = Path(
    os.environ.get("TORCH_MOJO_BACKEND_CACHE_DIR")
    or (_KERNELS_DIR / "__mojocache__" / "native")
)
_CSRC = _HERE / "csrc"
_MOJO_SRC = _HERE / "mojo"

_lock = threading.Lock()
_state: dict[str, object] = {}


def _trace_enabled() -> bool:
    return os.environ.get("TORCH_MOJO_BACKEND_TRACE", "1") != "0"


def _trace(msg: str):
    if _trace_enabled():
        print(f"[TRACE] {msg}", file=sys.stderr, flush=True)


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


@functools.cache
def _compiler_identity() -> str:
    """The compiler actually invoked (not just the package version) and the
    PTX assembler it will use."""
    try:
        version = subprocess.run(
            [_find_mojo(), "--version"], capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        version = "unknown"
    return f"{version}|ptxas={os.environ.get('MODULAR_NVPTX_COMPILER_PATH', '')}"


@functools.cache
def _accelerator_identity() -> str:
    """The devices a build targets: the Mojo runtime selects its vendor path
    at compile time, so an H100 build must never serve a gfx942 node."""
    from torch_mojo_backend.torch_compile_backend.utils import (  # noqa: PLC0415 -- imports max.driver; keep it off the import path
        get_accelerators,
    )

    return ",".join(
        f"{getattr(d, 'api', '')}:{getattr(d, 'label', '')}" for d in get_accelerators()
    )


def toolchain_identity() -> str:
    """What the shim and the Mojo base library depend on besides their
    sources. Neither depends on the accelerator: the base library picks the
    device api and vendor driver at run time (device.mojo), so one build
    serves NVIDIA, AMD and Apple machines and can ship prebuilt."""
    return "|".join(
        [
            f"torch={torch.__version__}",
            f"mojo={_pkg_version('mojo-compiler')}",
            f"max={_pkg_version('max-core')}",
            f"python={sys.implementation.cache_tag}",
            f"platform={sys.platform}",
            f"machine={platform.machine()}",
            _compiler_identity(),
        ]
    )


def kernel_identity() -> str:
    """The above plus the accelerators: kernel specializations carry device
    code for the GPU they were compiled for."""
    return toolchain_identity() + f"|accelerators={_accelerator_identity()}"


def _find_mojo() -> str:
    exe = shutil.which("mojo", path=str(Path(sys.executable).parent))
    if exe is None:
        exe = shutil.which("mojo")
    if exe is None:
        raise RuntimeError(
            "the `mojo` compiler was not found (is the max package installed?)"
        )
    return exe


def _cxx() -> list[str]:
    for cand in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if cand and shutil.which(cand):
            return [cand]
    raise RuntimeError(
        "no C++ compiler found: install g++ or clang++ (only the shim needs it)"
    )


def _torch_include_dir() -> Path:
    return Path(torch.__file__).parent / "include"


def _torch_include_flags() -> list[str]:
    inc = _torch_include_dir()
    return [f"-I{inc}", f"-I{inc / 'torch' / 'csrc' / 'api' / 'include'}"]


_AUTOCAST_LISTS = {
    "AT_FORALL_LOWER_PRECISION_FP": 1,
    "AT_FORALL_FP32": 2,
    "AT_FORALL_FP32_SET_OPT_DTYPE": 3,
    "AT_FORALL_PROMOTE": 4,
}

# AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE (policy 6, fp32_append_dtype) names
# the source overload and its C++ redispatch *signature*, not the overload that
# signature belongs to -- so the target is written out here, from
# native_functions.yaml. A torch release adding an entry this map does not cover
# fails the build loudly rather than autocasting it wrongly.
_AUTOCAST_APPEND_DTYPE_TARGET = {
    "norm.Scalar": "ScalarOpt_dtype",
    "norm.ScalarOpt_dim": "ScalarOpt_dim_dtype",
    "norm.names_ScalarOpt_dim": "names_ScalarOpt_dim_dtype",
}


def _autocast_macro_block(header: str, macro: str) -> str | None:
    """The body of one `#define <macro>(_)` list, line continuations joined."""
    start = header.find(f"#define {macro}(_)")
    if start < 0:
        return None
    return header[start : header.find("\n\n", start)].replace("\\\n", " ")


def autocast_policy_table() -> str:
    """The CUDA autocast op lists of the installed torch, read from
    ATen/autocast_mode.h's AT_FORALL_* macros, as C initializers
    `{"aten::op.overload", policy, redispatch_overload},` (see
    csrc/shim_autocast.cpp; `redispatch_overload` is omitted -- and so
    null -- for every policy but fp32_append_dtype)."""
    header = (_torch_include_dir() / "ATen" / "autocast_mode.h").read_text()
    lines = []
    for macro, policy in _AUTOCAST_LISTS.items():
        block = _autocast_macro_block(header, macro)
        if block is None:
            raise RuntimeError(f"{macro} not found in ATen/autocast_mode.h")
        for m in re.finditer(
            r"_\(\s*([A-Za-z0-9_]+)\s*(?:,\s*([A-Za-z0-9_]+))?\s*\)", block
        ):
            name = f"aten::{m.group(1)}" + (f".{m.group(2)}" if m.group(2) else "")
            lines.append(f'{{"{name}", {policy}}},')
    block = _autocast_macro_block(header, "AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE")
    if block is not None:
        for m in re.finditer(
            r'_\(\s*ADD_NS\(\s*[A-Za-z0-9_]+\s*\)\s*,\s*"([^"]+)"', block
        ):
            key = m.group(1)
            target = _AUTOCAST_APPEND_DTYPE_TARGET.get(key)
            if target is None:
                raise RuntimeError(
                    f"aten::{key} is in AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE but "
                    "torch_mojo_backend.native._AUTOCAST_APPEND_DTYPE_TARGET does not "
                    "name the overload it redispatches to"
                )
            lines.append(f'{{"aten::{key}", 6, "{target}"}},')
    return "\n".join(lines) + "\n"


def _torch_version_number() -> int:
    """major * 100 + minor of the torch in use, for `#if` in the shim (the
    wheels ship no torch/version.h)."""
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return int(m.group(1)) * 100 + int(m.group(2)) if m else 0


def _cxx_standard() -> str:
    """torch 2.14's headers need C++20 (std::strong_ordering, requires
    clauses); older releases compile as C++17, which keeps older compilers
    usable there. Major.minor only, so a 2.14 nightly counts as 2.14."""
    m = re.match(r"(\d+)\.(\d+)", torch.__version__)
    new_enough = m is not None and (int(m.group(1)), int(m.group(2))) >= (2, 14)
    return "-std=c++20" if new_enough else "-std=c++17"


def _cxx_identity(cxx: list[str]) -> str:
    try:
        out = subprocess.run(
            [*cxx, "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = "unknown"
    return " ".join(cxx) + "|" + out.splitlines()[0] if out else " ".join(cxx)


@contextlib.contextmanager
def _build_lock(name: str):
    """Cross-process dedupe of one build (best effort: a filesystem without
    locks just builds twice; the atomic install keeps that harmless)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f".{name}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            pass
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _hash_files(paths: list[Path], extra: str) -> str:
    h = hashlib.sha256(extra.encode())
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _scratch_dir() -> Path:
    """Where the compilers write: local disk, never the cache directory.
    The cache is often on NFS (a cluster home), and a `mojo build` writing
    its intermediate archive there failed intermittently with "failed to
    produce an archive for the module: No such file or directory" under
    load; finished libraries are copied over once, whole."""
    d = Path(tempfile.gettempdir()) / f"torch-mojo-backend-{os.getuid()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def compiler_env() -> dict[str, str]:
    """Environment every `mojo build` subprocess runs with (loader.mojo's
    `_compiler_env` is the same thing on the Mojo side).

    MODULAR_HOME holds the compiler's own module cache. Its default sits in
    $HOME, which on a cluster is NFS shared by every node, and concurrent
    compilers then evict each other's entries — "failed to produce an archive
    for the module: No such file or directory". Node-local, it is per-machine
    and nobody else touches it; the first build on a machine pays about 25 s
    to fill it. A value the caller set deliberately wins."""
    home = Path(
        os.environ.get("MODULAR_HOME")
        or Path(tempfile.gettempdir()) / f"modular-home-{os.getuid()}"
    )
    home.mkdir(parents=True, exist_ok=True)
    env: dict[str, str] = {**os.environ, "MODULAR_HOME": str(home)}
    # The MAX runtime exports the interpreter it found on PATH into this
    # process's environment (children inherit it); with a venv that is not on
    # PATH the `mojo` launcher script would start /usr/bin/python3 against the
    # venv's prefix and die with "Could not find platform independent
    # libraries". loader.mojo's _compiler_env unsets the same two.
    for name in ("PYTHONEXECUTABLE", "PYTHONHOME"):
        env.pop(name, None)
    return env


def _atomic_install(tmp: Path, out: Path):
    """Move a finished build from scratch into the cache: a copy into the
    cache directory (scratch is another filesystem), then one rename, so a
    reader never sees a partial file."""
    if out.exists():
        tmp.unlink(missing_ok=True)
        return
    staged = out.parent / f".{out.name}.{os.getpid()}"
    try:
        os.replace(tmp, staged)
    except OSError:  # EXDEV: cross-device
        shutil.copy2(tmp, staged)
        tmp.unlink(missing_ok=True)
    os.replace(staged, out)


def build_shim() -> Path:
    """Compile csrc/*.cpp into one shared library (parallel per file), cached
    by the sources, the torch version and the compiler."""
    sources = sorted(_CSRC.glob("*.cpp"))
    headers = sorted(_CSRC.glob("*.h"))
    cxx = _cxx()
    abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
    cflags = [
        "-O1",
        _cxx_standard(),
        "-fPIC",
        "-fvisibility=hidden",  # tmb.h re-exports the C entries; 40% smaller library
        "-fvisibility-inlines-hidden",
        "-ffunction-sections",
        "-fdata-sections",
        "-c",
        abi,
        f"-DTMB_TORCH_VERSION={_torch_version_number()}",
        *_torch_include_flags(),
    ]
    key = _hash_files(
        sources + headers,
        toolchain_identity()
        + "|"
        + _cxx_identity(cxx)
        + "|"
        + " ".join(cflags)
        + "|"
        + autocast_policy_table(),  # generated into the build, not a source file
    )
    out = (
        _CACHE_DIR
        / f"libtmb_shim.hash-{key}{'.dylib' if sys.platform == 'darwin' else '.so'}"
    )
    if out.exists():
        return out
    with _build_lock(out.name):
        if out.exists():
            return out
        return _build_shim_locked(sources, cxx, cflags, out)


def _build_shim_locked(
    sources: list[Path], cxx: list[str], cflags: list[str], out: Path
) -> Path:
    key = out.stem.split("hash-")[-1]
    t0 = time.monotonic()
    torch_lib = Path(torch.__file__).parent / "lib"
    tmpdir = _scratch_dir() / f"shim-{os.getpid()}-{key}"
    tmpdir.mkdir(exist_ok=True)
    (tmpdir / "tmb_autocast_policies.inc").write_text(autocast_policy_table())
    procs = []
    for src in sources:
        obj = tmpdir / (src.stem + ".o")
        procs.append(
            (
                src,
                subprocess.Popen(
                    [*cxx, *cflags, f"-I{tmpdir}", str(src), "-o", str(obj)],
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
        )
    errors = []
    for src, proc in procs:
        _, err = proc.communicate()
        if proc.returncode != 0:
            errors.append(f"--- {src.name}\n{err}")
    if errors:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("building the C++ shim failed:\n" + "\n".join(errors))
    tmp = tmpdir / out.name
    link = [
        *cxx,
        "-shared",
        "-Wl,-dead_strip" if sys.platform == "darwin" else "-Wl,--gc-sections",
        "-o",
        str(tmp),
        *[str(tmpdir / (s.stem + ".o")) for s in sources],
        f"-L{torch_lib}",
        "-ltorch_cpu",
        "-lc10",
    ]
    if sys.platform == "darwin":
        link += ["-undefined", "dynamic_lookup"]
    else:
        link += [f"-Wl,-rpath,{torch_lib}"]
    proc = subprocess.run(link, capture_output=True, text=True)
    if proc.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("linking the C++ shim failed:\n" + proc.stderr)
    _atomic_install(tmp, out)
    shutil.rmtree(tmpdir, ignore_errors=True)
    _trace(f"built C++ shim in {time.monotonic() - t0:.2f}s")
    return out


def _mojo_closure() -> list[Path]:
    """Everything the backend build reads: its own sources plus the shared
    eager_kernels modules it imports (op_utils, variant_gates)."""
    files = sorted(_MOJO_SRC.glob("*.mojo"))
    files += sorted((_KERNELS_DIR / "op_utils").glob("*.mojo"))
    files.append(_KERNELS_DIR / "variant_gates.mojo")
    return files


def build_backend() -> Path:
    """Compile mojo/backend.mojo into a shared library, cached by its closure."""
    key = _hash_files(_mojo_closure(), toolchain_identity())
    out = _CACHE_DIR / f"libtmb_backend.hash-{key}.so"
    if out.exists():
        return out
    with _build_lock(out.name):
        if out.exists():
            return out
        return _build_backend_locked(key, out)


def _build_backend_locked(key: str, out: Path) -> Path:
    t0 = time.monotonic()
    tmp = _scratch_dir() / f"backend-{os.getpid()}-{key}.so"
    cmd = [
        _find_mojo(),
        "build",
        str(_MOJO_SRC / "backend.mojo"),
        "--emit",
        "shared-lib",
        "-I",
        str(_MOJO_SRC),
        "-I",
        str(_KERNELS_DIR),
        "-o",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=compiler_env())
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "building the Mojo backend failed:\n"
            + " ".join(cmd)
            + "\n"
            + proc.stdout
            + proc.stderr
        )
    _atomic_install(tmp, out)
    _trace(f"built Mojo backend in {time.monotonic() - t0:.2f}s")
    return out


def _mojo_import_closure(entry: Path, roots: list[Path]) -> list[Path]:
    """Every .mojo file `entry` reaches through top-level imports resolved in
    `roots` (the same rule the Mojo loader uses for kernel families)."""
    seen: dict[Path, None] = {}
    todo = [entry.resolve()]
    while todo:
        f = todo.pop()
        if f in seen or not f.exists():
            continue
        seen[f] = None
        for line in f.read_text().splitlines():
            m = re.match(r"^(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if not m or m.group(1) in ("std", "max", "nn", "linalg", "layout"):
                continue
            for root in [f.parent, *roots]:
                cand = root / f"{m.group(1)}.mojo"
                if cand.exists():
                    todo.append(cand.resolve())
                    break
    return sorted(seen)


def build_library(
    entry: Path, roots: list[Path] | None = None, defines: dict[str, str] | None = None
) -> Path:
    """Compile a plain Mojo shared library (a C-ABI export set, e.g. the mojoccl
    collectives) once per closure/toolchain, cached like the backend."""
    roots = [entry.parent, *(roots or [])]
    closure = _mojo_import_closure(entry, roots)
    tag = "|".join(f"{k}={v}" for k, v in sorted((defines or {}).items()))
    key = _hash_files(closure, kernel_identity() + "|" + tag)  # device code inside
    out = _CACHE_DIR / f"lib{entry.stem}.hash-{key}.so"
    if out.exists():
        return out
    with _build_lock(out.name):
        if out.exists():
            return out
        t0 = time.monotonic()
        tmp = _scratch_dir() / f"{entry.stem}-{os.getpid()}-{key}.so"
        cmd = [_find_mojo(), "build", str(entry), "--emit", "shared-lib"]
        for root in roots:
            cmd += ["-I", str(root)]
        for k, v in sorted((defines or {}).items()):
            cmd += ["-D", f"{k}={v}"]
        cmd += ["-o", str(tmp)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=compiler_env())
        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"building {entry.name} failed:\n" + proc.stdout + proc.stderr
            )
        _atomic_install(tmp, out)
        _trace(f"built {entry.name} in {time.monotonic() - t0:.2f}s")
        return out


def _load(path: Path, mode: int) -> ctypes.CDLL:
    return ctypes.CDLL(str(path), mode=mode)


def is_registered() -> bool:
    return bool(_state.get("registered"))


def op_counting(enabled: bool):
    """Count boxed-kernel calls per op (test support; off by default)."""
    shim().tmb_op_counting(ctypes.c_int32(1 if enabled else 0))


def op_counts_reset():
    shim().tmb_op_counts_reset()


def op_count(qualified_name: str) -> int:
    """Calls of e.g. "aten::add.Tensor" since the last reset."""
    fn = shim().tmb_op_count
    fn.restype = ctypes.c_int64
    return int(fn(qualified_name.encode()))


def op_counts() -> dict[str, int]:
    """Every counted op since the last reset."""
    fn = shim().tmb_op_counts_dump
    fn.restype = ctypes.c_int64
    need = fn(None, ctypes.c_int64(0))
    buf = ctypes.create_string_buffer(int(need) + 1)
    fn(buf, ctypes.c_int64(len(buf)))
    out: dict[str, int] = {}
    for line in buf.value.decode().splitlines():
        name, _, count = line.partition("=")
        if name:
            out[name] = int(count)
    return out


def prebuild_ops():
    """Compile every op's extension now rather than one per first call.

    Only useful up front: a test suite or a CI image pays the compilations
    here, outside any GPU lock, instead of inside the first call of each op.
    """
    fn = backend_lib().tmb_prebuild_ops
    fn.restype = ctypes.c_int32
    if fn() != 0:
        raise RuntimeError("prebuilding the mojo ops failed: " + last_error())


def device_count() -> int:
    return cast(int, _state.get("device_count", 0))


def shim() -> ctypes.CDLL:
    lib = _state.get("shim")
    if lib is None:
        raise RuntimeError("the mojo device is not registered yet")
    return cast(ctypes.CDLL, lib)


def backend_lib() -> ctypes.CDLL:
    lib = _state.get("backend")
    if lib is None:
        raise RuntimeError("the mojo device is not registered yet")
    return cast(ctypes.CDLL, lib)


def last_error() -> str:
    """The shim's thread-local error message (set by the last failing call)."""
    fn = shim().tmb_get_error
    fn.restype = ctypes.c_char_p
    return (fn() or b"").decode()


def register():
    """Build/load both libraries and register the backend. Idempotent."""
    with _lock:
        if _state.get("registered"):
            return
        t0 = time.monotonic()
        with ThreadPoolExecutor(
            max_workers=2
        ) as pool:  # the two builds are independent
            shim_future = pool.submit(build_shim)
            backend_future = pool.submit(build_backend)
            shim_path = shim_future.result()
            backend_path = backend_future.result()
        # libtorch's symbols must be visible to the Mojo library (external_call),
        # and the shim's to the kernel families it will dlopen.
        torch_lib = Path(torch.__file__).parent / "lib"
        _load(
            torch_lib
            / ("libtorch_cpu.dylib" if sys.platform == "darwin" else "libtorch_cpu.so"),
            ctypes.RTLD_GLOBAL,
        )
        shim_lib = _load(shim_path, ctypes.RTLD_GLOBAL)
        backend = _load(backend_path, ctypes.RTLD_GLOBAL)
        backend.tmb_native_init.restype = ctypes.c_int32
        backend.tmb_native_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        shim_lib.tmb_get_error.restype = ctypes.c_char_p
        n = backend.tmb_native_init(
            str(_KERNELS_DIR).encode(),
            str(_MOJO_SRC).encode(),
            str(_CACHE_DIR).encode(),
            _find_mojo().encode(),
            kernel_identity().encode(),  # the loader keys kernel builds with it
            1 if _trace_enabled() else 0,
        )
        if n < 0:
            raise RuntimeError(
                "mojo backend initialisation failed: "
                + (shim_lib.tmb_get_error() or b"").decode()
            )
        shim_lib.tmb_autocast_install_cuda_policies()
        _state.update(shim=shim_lib, backend=backend, device_count=n, registered=True)
        _trace(
            f"native mojo backend ready in {time.monotonic() - t0:.2f}s ({n} devices)"
        )
