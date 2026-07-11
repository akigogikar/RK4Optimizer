"""Merge sharded cifar_experiment.py outputs into one results file.

Usage:
  python3 merge_cifar_results.py cifar_results_shard*.json --out cifar_results.json

Shards partition the config list round-robin (see cifar_experiment.py), so the
union of their "results" dicts is disjoint; we verify that and fail loudly on
any collision or metadata mismatch (budget / n_train / seeds must agree).
"""
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="shard JSON files")
    ap.add_argument("--out", default="cifar_results.json")
    args = ap.parse_args()

    merged, meta = {}, None
    for path in args.shards:
        with open(path) as f:
            blob = json.load(f)
        m = {k: blob[k] for k in ("budget", "n_train", "seeds")}
        if meta is None:
            meta = m
        elif m != meta:
            sys.exit(f"metadata mismatch in {path}: {m} != {meta}")
        for name, entry in blob["results"].items():
            if name in merged:
                sys.exit(f"duplicate config across shards: {name!r} (in {path})")
            merged[name] = entry

    out = dict(meta, num_shards=len(args.shards), results=merged)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"merged {len(merged)} configs from {len(args.shards)} shards -> {args.out}")


if __name__ == "__main__":
    main()
