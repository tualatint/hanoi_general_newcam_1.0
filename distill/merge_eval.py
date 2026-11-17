#!/usr/bin/env python
"""Merge sharded eval_general_chain.py JSONs and report SR with Wilson CI."""
import argparse
import glob
import json
import math


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = []
    for pat in args.jsons:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))
    rows = []
    for p in paths:
        with open(p) as f:
            j = json.load(f)
        rows.extend(j.get("rows", []))
    rows.sort(key=lambda r: r["seed"])
    seeds = [r["seed"] for r in rows]
    assert len(seeds) == len(set(seeds)), "duplicate seeds across shards"

    n = len(rows)
    k = sum(1 for r in rows if r["success"])
    lo, hi = wilson(k, n)
    print(f"[merge] shards={len(paths)} n={n} success={k} sr={k/max(1,n):.4f}")
    print(f"[merge] Wilson 95% CI: [{lo:.4f}, {hi:.4f}]")
    for reason in sorted({r.get("switch", "?") for r in rows}):
        sub = [r for r in rows if r.get("switch") == reason]
        s = sum(1 for r in sub if r["success"])
        print(f"  {reason}: {s}/{len(sub)}")
    for key, idx in (("disk", 0), ("src", 1), ("dst", 2), ("level", 3)):
        groups = {}
        for r in rows:
            g = r["task"][idx]
            a, b = groups.get(g, (0, 0))
            groups[g] = (a + int(r["success"]), b + 1)
        print(f"  by {key}: " + "  ".join(
            f"{g}:{a}/{b}" for g, (a, b) in sorted(groups.items())))
    fails = [r for r in rows if not r["success"]]
    if fails:
        print(f"[merge] failures ({len(fails)}):")
        for r in fails:
            print(f"  seed={r['seed']} task={r['task']} switch={r['switch']} "
                  f"t={r.get('transport_steps')} i={r.get('insert_steps')} "
                  f"final_err={r.get('final_err_mm')}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n": n, "success": k, "sr": k / max(1, n),
                       "wilson95": [lo, hi], "rows": rows}, f, indent=1)
        print(f"[merge] wrote {args.out}")


if __name__ == "__main__":
    main()
