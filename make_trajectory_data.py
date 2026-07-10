"""Capture step-size / error-estimate trajectories for the paper figures.

Runs (single seed, 600-eval budget, same protocol as fullbatch_experiment.py):
  1. As-designed RK3(2)-Adam, lr=0.003, defaults (hmax=2x) -> h_t, err_t
  2. As-designed RK3(2)-Adam, lr=0.003, hmax=8x            -> h_t, err_t
  3. Repaired FixedRK3Adam, h0=0.001, rtol=0.1             -> h_t + reject steps
  4. Repaired FixedRK3Adam, h0=0.001, rtol=0.01            -> h_t + reject steps

Saves trajectories.json. Cheap (<1 min); regenerate at will.
"""
import json
import torch
import torch.nn as nn

from fullbatch_experiment import make_model, get_data
from RK4Optimizer import AdaptiveEmbeddedRK3Optimizer
from fixed_controller_experiment import FixedRK3Adam

BUDGET = 600
SEED = 0


def as_designed(lr, rtol, hms, data):
    x, y = data
    model = make_model(SEED)
    opt = AdaptiveEmbeddedRK3Optimizer(model, lr=lr, rtol=rtol, h_max_scale=hms)
    loss_fn = nn.CrossEntropyLoss()
    t = 0
    while opt.grad_evals < BUDGET - 3:
        t += 1
        opt.step(loss_fn, x, y, t)
    return {"h": [float(h) for h in opt.h_history],
            "err": [float(e) for e in opt.err_history],
            "h_max": float(opt.h_max), "h_min": float(opt.h_min),
            "frac_at_hmax": opt.n_saturated_max / max(len(opt.h_history), 1)}


def repaired(h0, rtol, data):
    x, y = data
    model = make_model(SEED)
    opt = FixedRK3Adam(model, h0=h0, rtol=rtol)
    loss_fn = nn.CrossEntropyLoss()
    hs, rejects_at = [], []
    t = 0
    while opt.grad_evals < BUDGET - 4:
        t += 1
        r0 = opt.rejects
        opt.step(loss_fn, x, y, t)
        hs.append(float(opt.h))
        if opt.rejects > r0:
            rejects_at.append(len(hs) - 1)
    return {"h": hs, "rejects_at": rejects_at, "h_max": float(opt.h_max),
            "total_rejects": opt.rejects}


def main():
    data, _ = get_data(1024)
    out = {
        "budget": BUDGET, "seed": SEED,
        "as_designed_hmax2": as_designed(0.003, 0.01, 2.0, data),
        "as_designed_hmax8": as_designed(0.003, 0.01, 8.0, data),
        "repaired_rtol0.1": repaired(0.001, 0.1, data),
        "repaired_rtol0.01": repaired(0.001, 0.01, data),
    }
    # headline numbers for the saturation proposition
    for k in ("as_designed_hmax2", "as_designed_hmax8"):
        errs = out[k]["err"]
        out[k]["max_err"] = max(errs) if errs else None
    with open("trajectories.json", "w") as f:
        json.dump(out, f)
    print("max err (hmax=2x):", out["as_designed_hmax2"]["max_err"],
          " frac@hmax:", out["as_designed_hmax2"]["frac_at_hmax"])
    print("max err (hmax=8x):", out["as_designed_hmax8"]["max_err"],
          " frac@hmax:", out["as_designed_hmax8"]["frac_at_hmax"])
    print("repaired rejects rtol=0.1:", out["repaired_rtol0.1"]["total_rejects"],
          " rtol=0.01:", out["repaired_rtol0.01"]["total_rejects"])
    print("Saved -> trajectories.json")


if __name__ == "__main__":
    main()
