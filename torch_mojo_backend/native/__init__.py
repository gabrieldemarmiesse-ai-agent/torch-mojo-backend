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

import ctypes
import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent
_KERNELS_DIR = _PACKAGE / "eager_kernels"
# One cache for every checkout on a box (contents-addressed: every build is
# keyed by its sources and toolchain), overridable for shared scratch space.
_CACHE_DIR = Path(os.environ.get("TORCH_MOJO_BACKEND_CACHE_DIR") or (_KERNELS_DIR / "__mojocache__" / "native"))
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


def toolchain_identity() -> str:
    """What both builds depend on besides their sources."""
    return "|".join(
        [
            f"torch={torch.__version__}",
            f"mojo={_pkg_version('mojo-compiler')}",
            f"max={_pkg_version('max-core')}",
            f"python={sys.implementation.cache_tag}",
            f"platform={sys.platform}",
            f"machine={platform.machine()}",
        ]
    )


def _find_mojo() -> str:
    exe = shutil.which("mojo", path=str(Path(sys.executable).parent))
    if exe is None:
        exe = shutil.which("mojo")
    if exe is None:
        raise RuntimeError("the `mojo` compiler was not found (is the max package installed?)")
    return exe


def _cxx() -> list[str]:
    for cand in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if cand and shutil.which(cand):
            return [cand]
    raise RuntimeError("no C++ compiler found: install g++ or clang++ (only the shim needs it)")


def _torch_include_flags() -> list[str]:
    from torch.utils.cpp_extension import include_paths  # noqa: PLC0415 -- pulls in ninja probing; keep it off the import path

    return [f"-I{p}" for p in include_paths()]


def _hash_files(paths: list[Path], extra: str) -> str:
    h = hashlib.sha256(extra.encode())
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _atomic_install(tmp: Path, out: Path):
    if out.exists():
        tmp.unlink(missing_ok=True)
    else:
        os.replace(tmp, out)


def build_shim() -> Path:
    """Compile csrc/*.cpp into one shared library (parallel per file), cached
    by the sources, the torch version and the compiler."""
    sources = sorted(_CSRC.glob("*.cpp"))
    headers = sorted(_CSRC.glob("*.h"))
    cxx = _cxx()
    key = _hash_files(sources + headers, toolchain_identity() + "|" + " ".join(cxx))
    out = _CACHE_DIR / f"libtmb_shim.hash-{key}{'.dylib' if sys.platform == 'darwin' else '.so'}"
    if out.exists():
        return out
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    torch_lib = Path(torch.__file__).parent / "lib"
    abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
    cflags = ["-O1", "-std=c++17", "-fPIC", "-c", abi, *_torch_include_flags()]
    tmpdir = _CACHE_DIR / f".shim-{os.getpid()}-{key}"
    tmpdir.mkdir(exist_ok=True)
    procs = []
    for src in sources:
        obj = tmpdir / (src.stem + ".o")
        procs.append((src, subprocess.Popen([*cxx, *cflags, str(src), "-o", str(obj)], stderr=subprocess.PIPE, text=True)))
    errors = []
    for src, proc in procs:
        _, err = proc.communicate()
        if proc.returncode != 0:
            errors.append(f"--- {src.name}\n{err}")
    if errors:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("building the C++ shim failed:\n" + "\n".join(errors))
    tmp = tmpdir / out.name
    link = [*cxx, "-shared", "-o", str(tmp), *[str(tmpdir / (s.stem + ".o")) for s in sources], f"-L{torch_lib}", "-ltorch_cpu", "-lc10"]
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
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    tmp = _CACHE_DIR / f".backend-{os.getpid()}-{key}.so"
    cmd = [_find_mojo(), "build", str(_MOJO_SRC / "backend.mojo"), "--emit", "shared-lib", "-I", str(_MOJO_SRC), "-I", str(_KERNELS_DIR), "-o", str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("building the Mojo backend failed:\n" + proc.stdout + proc.stderr)
    _atomic_install(tmp, out)
    _trace(f"built Mojo backend in {time.monotonic() - t0:.2f}s")
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


def device_count() -> int:
    return int(_state.get("device_count", 0))


def shim() -> ctypes.CDLL:
    lib = _state.get("shim")
    if lib is None:
        raise RuntimeError("the mojo device is not registered yet")
    return lib  # type: ignore[return-value]


def register():
    """Build/load both libraries and register the backend. Idempotent."""
    with _lock:
        if _state.get("registered"):
            return
        t0 = time.monotonic()
        shim_path = build_shim()
        backend_path = build_backend()
        # libtorch's symbols must be visible to the Mojo library (external_call),
        # and the shim's to the kernel families it will dlopen.
        torch_lib = Path(torch.__file__).parent / "lib"
        _load(torch_lib / ("libtorch_cpu.dylib" if sys.platform == "darwin" else "libtorch_cpu.so"), ctypes.RTLD_GLOBAL)
        shim_lib = _load(shim_path, ctypes.RTLD_GLOBAL)
        backend = _load(backend_path, ctypes.RTLD_GLOBAL)
        backend.tmb_native_init.restype = ctypes.c_int32
        backend.tmb_native_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]
        shim_lib.tmb_get_error.restype = ctypes.c_char_p
        n = backend.tmb_native_init(
            str(_KERNELS_DIR).encode(),
            str(_CACHE_DIR).encode(),
            _find_mojo().encode(),
            toolchain_identity().encode(),
            1 if _trace_enabled() else 0,
        )
        if n < 0:
            raise RuntimeError("mojo backend initialisation failed: " + (shim_lib.tmb_get_error() or b"").decode())
        shim_lib.tmb_autocast_install_cuda_policies()
        _state.update(shim=shim_lib, backend=backend, device_count=n, registered=True)
        _trace(f"native mojo backend ready in {time.monotonic() - t0:.2f}s ({n} devices)")
