"""Record the RNG parity cases on one device, or compare two records.

    # under stock CUDA torch (any venv that can init CUDA):
    python tests/native/rng_parity_dump.py dump cuda /tmp/ref_cuda.pt [--big] [--only a,b]
    # under this package:
    uv run python tests/native/rng_parity_dump.py dump mojo /tmp/ours.pt [--big]
    python tests/native/rng_parity_dump.py compare /tmp/ref_cuda.pt /tmp/ours.pt
    # refresh the checked-in digests from a CUDA record:
    python tests/native/rng_parity_dump.py golden /tmp/ref_cuda.pt

A record holds, per case, the tensor's digest, head/tail values and the device
generator state after the draw; `golden` keeps only the device-independent
cases (rng_parity_cases.GOLDEN) in tests/native/rng_golden.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import TypedDict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.native import rng_parity_cases as cases  # noqa: E402

GOLDEN_PATH = Path(__file__).with_name("rng_golden.json")


class Record(TypedDict):
    dtype: str
    shape: list[int]
    sha256: str
    state: list[int]
    head: list[float]
    tail: list[float]
    tensor: torch.Tensor | None


def digest_of(t: torch.Tensor) -> str:
    host = t.detach().cpu().contiguous().reshape(-1)
    raw = host.view(torch.uint8) if host.dtype != torch.bool else host.to(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def record(t: torch.Tensor, state: torch.Tensor) -> Record:
    host = t.detach().cpu().contiguous()
    flat = host.reshape(-1)
    return {
        "dtype": str(host.dtype),
        "shape": list(host.shape),
        "sha256": digest_of(host),
        "state": state.tolist(),
        "head": flat[:16].tolist(),
        "tail": flat[-16:].tolist(),
        "tensor": host if host.numel() <= 8192 else None,
    }


def state_of(device: str) -> torch.Tensor:
    if device.startswith("cuda"):
        return torch.cuda.get_rng_state().clone()
    return torch.get_device_module("mojo").get_rng_state().clone()


def run_case(fn, device: str) -> Record:
    torch.manual_seed(cases.SEED)
    t = fn(device)
    return record(t, state_of(device))


def dump(side: str, out: str, big: bool, only: str | None = None):
    if side == "mojo":
        import torch_mojo_backend  # noqa: PLC0415 -- the cuda side runs under stock torch without this package

        torch_mojo_backend.register_mojo_devices()
        device, props = "mojo:0", None
    else:
        device = "cuda:0"
        p = torch.cuda.get_device_properties(0)
        props = (p.name, p.multi_processor_count, p.max_threads_per_multi_processor)
    todo = dict(cases.CASES)
    if big:
        todo.update(cases.BIG_CASES)
    if only:
        wanted = only.split(",")
        todo = {k: v for k, v in todo.items() if any(w in k for w in wanted)}
    results = {}
    t0 = time.time()
    for name, fn in todo.items():
        try:
            results[name] = {"ok": True, **run_case(fn, device)}
        except Exception as e:  # noqa: BLE001 -- recorded, not raised
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
            print(f"[{side}] {name}: ERROR {results[name]['error']}", flush=True)
    torch.save(
        {"side": side, "props": props, "torch": torch.__version__, "results": results},
        out,
    )
    print(f"[{side}] {len(results)} cases in {time.time() - t0:.1f}s -> {out}")


def compare(ref_path: str, ours_path: str, verbose: bool) -> int:
    ref = torch.load(ref_path, weights_only=False)
    ours = torch.load(ours_path, weights_only=False)
    print(f"ref: {ref['side']} torch {ref['torch']} props {ref['props']}")
    print(f"ours: {ours['side']} torch {ours['torch']}")
    exact = bad = 0
    for name, r in ref["results"].items():
        o = ours["results"].get(name)
        if o is None:
            print(f"MISSING  {name}")
            bad += 1
            continue
        if not r["ok"] or not o["ok"]:
            print(
                f"ERROR    {name}: ref={r.get('error', 'ok')} ours={o.get('error', 'ok')}"
            )
            bad += 1
            continue
        same = (
            r["sha256"] == o["sha256"]
            and r["state"] == o["state"]
            and r["dtype"] == o["dtype"]
            and r["shape"] == o["shape"]
        )
        if same:
            exact += 1
            if verbose:
                print(f"EXACT    {name}")
            continue
        bad += 1
        detail = []
        if r["state"] != o["state"]:
            ro = int.from_bytes(bytes(r["state"][8:]), "little")
            oo = int.from_bytes(bytes(o["state"][8:]), "little")
            detail.append(f"offset ref={ro} ours={oo}")
        a, b = r.get("tensor"), o.get("tensor")
        if (
            a is not None
            and b is not None
            and a.shape == b.shape
            and a.dtype == b.dtype
        ):
            ne = a != b
            n_diff = int(ne.sum())
            if n_diff:
                idx = int(torch.nonzero(ne.reshape(-1))[0])
                detail.append(
                    f"{n_diff}/{a.numel()} differ, first at {idx}: ref={a.reshape(-1)[idx].item()} ours={b.reshape(-1)[idx].item()}"
                )
                if a.dtype.is_floating_point:
                    rel = (
                        (
                            (a.double() - b.double()).abs()
                            / a.double().abs().clamp_min(1e-30)
                        )
                        .max()
                        .item()
                    )
                    detail.append(f"max rel {rel:.3e}")
        elif r["sha256"] != o["sha256"]:
            detail.append(f"head ref={r['head'][:4]} ours={o['head'][:4]}")
        print(f"MISMATCH {name}: " + "; ".join(detail))
    print(f"\nexact {exact}, mismatch/error {bad}, of {len(ref['results'])}")
    return 1 if bad else 0


def golden(ref_path: str):
    ref = torch.load(ref_path, weights_only=False)
    out = {"source": f"{ref['side']} torch {ref['torch']} {ref['props']}", "cases": {}}
    for name in cases.GOLDEN:
        r = ref["results"][name]
        if not r["ok"]:
            raise SystemExit(f"{name} failed in the reference: {r['error']}")
        out["cases"][name] = {
            k: r[k] for k in ("dtype", "shape", "sha256", "state", "head")
        }
    lines = [f'{{"source": {json.dumps(out["source"])}, "cases": {{']
    items = list(out["cases"].items())
    for i, (name, rec) in enumerate(items):
        comma = "," if i + 1 < len(items) else ""
        lines.append(f"  {json.dumps(name)}: {json.dumps(rec)}{comma}")
    lines.append("}}")
    GOLDEN_PATH.write_text("\n".join(lines) + "\n")
    print(f"{len(out['cases'])} golden cases -> {GOLDEN_PATH}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dump":
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
        dump(sys.argv[2], sys.argv[3], "--big" in sys.argv, only)
    elif cmd == "compare":
        raise SystemExit(compare(sys.argv[2], sys.argv[3], "-v" in sys.argv))
    elif cmd == "golden":
        golden(sys.argv[2])
    else:
        raise SystemExit(__doc__)
