"""Aggregate cifar_results_shard*.json into the paper's Table (tab:cifar)
plus automated verdicts on the three pre-registered CIFAR claims.

Tolerant of partial results: missing configs render as "---" and the
verdict block reports PENDING with a completion count. Run any time;
safe while shards are still writing (shard saves are atomic).

Usage:  python3 make_cifar_table.py [--latex]
"""
import argparse, glob, json

LRS = (0.001, 0.003)
POOL = [f"{n} lr={lr}" for lr in LRS
        for n in ("GD", "SGD+mom0.9", "Adam", "AdamW(wd=1e-2)",
                  "RMSprop", "NAdam")]
RK_GRID = [f"RK3(2)-Adam lr={lr} rtol={r} hmax={m}x"
           for lr in LRS for r in (0.01, 0.1, 1.0) for m in (2.0, 8.0)]
# Constructor defaults in RK4Optimizer.AdaptiveEmbeddedRK3Optimizer:
# rtol=1e-2, h_max_scale=2.0  (NOT rtol=0.1 -- verified against source).
RK_DEFAULTS = "RK3(2)-Adam lr=0.003 rtol=0.01 hmax=2.0x"
FIXED = [f"FixedRK3Adam h0={h} rtol={r}" for h in (0.001, 0.003)
         for r in (0.01, 0.1)]
PURE = "PureRK3(2) h0=0.1 rtol=0.01"
ALL = POOL + RK_GRID + FIXED + [PURE]


def load():
    res = {}
    for p in sorted(glob.glob("cifar_results_shard*.json")):
        with open(p) as f:
            res.update(json.load(f)["results"])
    return res


def fmt(entry, with_hmax=False):
    if entry is None:
        return "--- & --- & ---" if with_hmax else "--- & --- & n/a"
    loss = f"${entry['mean_final_loss']:.4g} \\pm {entry['std_final_loss']:.2g}$"
    acc = f"{entry['mean_test_acc'] * 100:.2f}"
    ctrl = entry.get("ctrl") or {}
    sat = (f"{ctrl['frac_at_hmax'] * 100:.0f}"
           if ctrl.get("frac_at_hmax") is not None else "n/a")
    return f"{loss} & {acc} & {sat}"


def best(res, names, key="mean_final_loss"):
    have = [(n, res[n]) for n in names if n in res]
    return min(have, key=lambda kv: kv[1][key]) if have else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    res = load()
    done = [n for n in ALL if n in res]
    missing = [n for n in ALL if n not in res]
    print(f"# progress: {len(done)}/{len(ALL)} configs complete")
    if missing:
        print("# missing:", "; ".join(missing))

    pool_best_name, pool_best = best(res, POOL)
    rk_best_name, rk_best = best(res, RK_GRID)
    fix_best_name, fix_best = best(res, FIXED)
    rows = [
        (f"Best first-order ({pool_best_name or 'TBD'})", pool_best, False),
        ("Adam ($\\mathrm{lr}{=}3{\\times}10^{-3}$)",
         res.get("Adam lr=0.003"), False),
        ("RK3(2)-Adam (defaults)", res.get(RK_DEFAULTS), True),
        (f"RK3(2)-Adam (best: {rk_best_name or 'TBD'})", rk_best, True),
        (f"FixedRK3Adam (best: {fix_best_name or 'TBD'})", fix_best, False),
        ("PureRK3(2)", res.get(PURE), False),
    ]
    if args.latex:
        for label, entry, with_hmax in rows:
            print(f"{label} & {fmt(entry, with_hmax)} \\\\")
        print()

    # ---- pre-registered claim verdicts -------------------------------
    v = {}
    if pool_best and rk_best:
        pool_acc = max(res[n]["mean_test_acc"] for n in POOL if n in res)
        v["claim_i_rk_never_clears_pool"] = {
            "rk_best_loss": rk_best["mean_final_loss"],
            "pool_best_loss": pool_best["mean_final_loss"],
            "rk_best_acc": max(res[n]["mean_test_acc"]
                               for n in RK_GRID if n in res),
            "pool_best_acc": pool_acc,
            "verdict": ("CONFIRMED"
                        if rk_best["mean_final_loss"]
                        > pool_best["mean_final_loss"] else "REFUTED"),
        }
    sats = [res[n]["ctrl"]["frac_at_hmax"] for n in RK_GRID
            if n in res and res[n].get("ctrl")]
    ident = {}
    for lr in LRS:
        for m in (2.0, 8.0):
            group = [f"RK3(2)-Adam lr={lr} rtol={r} hmax={m}x"
                     for r in (0.01, 0.1, 1.0)]
            got = [tuple(res[n]["losses"]) for n in group if n in res]
            if len(got) == 3:
                ident[f"lr={lr} hmax={m}x"] = len(set(got)) == 1
    if sats:
        v["claim_ii_inert_controller"] = {
            "frac_at_hmax_min": min(sats), "frac_at_hmax_max": max(sats),
            "n_configs": len(sats),
            "rtol_bit_identical_by_group": ident,
        }
    rej = {n: res[n]["ctrl"].get("rejects") for n in FIXED + [PURE]
           if n in res and res[n].get("ctrl")}
    if rej:
        fixed_losses = {n: tuple(res[n]["losses"]) for n in FIXED if n in res}
        v["claim_iii_repaired_adaptive"] = {
            "rejects": rej,
            "rtol_dependent": len(set(fixed_losses.values()))
            == len(fixed_losses) if len(fixed_losses) > 1 else None,
        }
    v["status"] = "COMPLETE" if not missing else f"PENDING {len(done)}/{len(ALL)}"
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
