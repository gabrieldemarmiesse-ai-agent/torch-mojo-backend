"""Join two profile_gpt2xl_ops.py JSONs (cuda vs mojo) and rank op slowdowns.

Usage:
    uv run --no-sync python current_bench/compare_op_profiles.py \
        /tmp/prof_cuda.json /tmp/prof_mojo.json [--top 25] [--shapes-for aten::mm]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> tuple[dict, dict[str, float], dict[str, list[dict]]]:
    data = json.loads(path.read_text())
    steps = data["profile_steps"]
    per_op: dict[str, float] = defaultdict(float)
    shape_rows: dict[str, list[dict]] = defaultdict(list)
    for row in data["rows"]:
        us_per_step = row["self_device_us"] / steps
        if us_per_step <= 0:
            continue
        per_op[row["key"]] += us_per_step
        shape_rows[row["key"]].append(
            {
                "shapes": row["input_shapes"],
                "us_per_step": us_per_step,
                "calls_per_step": row["count"] / steps,
            }
        )
    return data, dict(per_op), dict(shape_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cuda_json", type=Path)
    parser.add_argument("mojo_json", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--shapes-for",
        default=None,
        help="print the per-shape breakdown of one op on both backends",
    )
    args = parser.parse_args()

    cuda_meta, cuda_ops, cuda_shapes = load(args.cuda_json)
    mojo_meta, mojo_ops, mojo_shapes = load(args.mojo_json)

    for meta in (cuda_meta, mojo_meta):
        print(
            f"{meta['device']:>5}: fwd+bwd {meta['fwd_bwd_ms_median']:8.1f} ms/step  "
            f"(B={meta['batch_size']}, T={meta['block_size']}, "
            f"L={meta['n_layer']}, dtype={meta['dtype']}, bias={meta['bias']})"
        )
    cuda_total = sum(cuda_ops.values()) / 1e3
    mojo_total = sum(mojo_ops.values()) / 1e3
    print(
        f"total self device time: cuda {cuda_total:.1f} ms/step, "
        f"mojo {mojo_total:.1f} ms/step, ratio {mojo_total / cuda_total:.2f}x\n"
    )

    if args.shapes_for:
        for name, shapes in (("cuda", cuda_shapes), ("mojo", mojo_shapes)):
            print(f"--- {args.shapes_for} on {name} ---")
            rows = sorted(
                shapes.get(args.shapes_for, []), key=lambda r: -r["us_per_step"]
            )
            for r in rows[: args.top]:
                print(
                    f"  {r['us_per_step'] / 1e3:9.3f} ms/step  "
                    f"{r['calls_per_step']:6.1f} calls/step  {r['shapes']}"
                )
        return

    keys = set(cuda_ops) | set(mojo_ops)
    joined = []
    for k in keys:
        c = cuda_ops.get(k, 0.0)
        m = mojo_ops.get(k, 0.0)
        joined.append((m - c, m / c if c > 0 else float("inf"), c, m, k))
    joined.sort(reverse=True)

    print(f"{'op':<48} {'cuda ms':>9} {'mojo ms':>9} {'ratio':>7} {'delta ms':>9}")
    print("-" * 88)
    for delta, ratio, c, m, k in joined[: args.top]:
        ratio_s = f"{ratio:7.2f}" if ratio != float("inf") else "   only"
        print(f"{k:<48} {c / 1e3:9.3f} {m / 1e3:9.3f} {ratio_s} {delta / 1e3:9.3f}")

    cuda_only = [k for k in cuda_ops if k not in mojo_ops]
    if cuda_only:
        print("\nops with device time only on cuda (different decomposition?):")
        for k in sorted(cuda_only, key=lambda k: -cuda_ops[k])[:15]:
            print(f"  {k:<48} {cuda_ops[k] / 1e3:9.3f} ms/step")


if __name__ == "__main__":
    main()
