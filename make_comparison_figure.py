"""Regenerate comparison.png from fullbatch_results.json.

Two panels (lr=0.001, lr=0.003), horizontal bars of final full-batch train
loss (log scale, 600 grad evals, 3 seeds, mean±std). RK3(2)-Adam rtol
triplets are collapsed into single bars because they are bit-identical —
which is itself the paper's Section 5.1 finding.
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("fullbatch_results.json") as f:
    data = json.load(f)
results = data["results"]

BASELINES = [
    ("GD", "GD (Euler)"),
    ("Nesterov0.9", "Nesterov 0.9"),
    ("SGD+mom0.9", "SGD+mom 0.9"),
    ("Adagrad", "Adagrad"),
    ("RAdam", "RAdam"),
    ("NAdam", "NAdam"),
    ("RMSprop", "RMSprop"),
    ("AdamW(wd=1e-2)", "AdamW"),
    ("Adam", "Adam"),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=False)

for ax, lr in zip(axes, ["0.001", "0.003"]):
    labels, means, stds, colors = [], [], [], []

    # Baselines
    for key, label in BASELINES:
        v = results[f"{key} lr={lr}"]
        labels.append(label)
        means.append(v["mean_final_loss"])
        stds.append(v["std_final_loss"])
        colors.append("#d62728" if key == "Adam" else "#7f7f7f")

    # RK configs keyed by (hmax, h0) -> {rtol: result}.
    # rtol=0.1 and rtol=1.0 are bit-identical (collapsed into the bar);
    # rtol=0.01 can differ via its smaller first pre-saturation step and is
    # overlaid as a diamond marker where it does.
    rk = {}
    pat = re.compile(
        rf"RK3\(2\)-Adam lr={lr} rtol=([\d.]+) hmax=([\d.]+)x h0=([\d.]+)x")
    for key, v in results.items():
        m = pat.match(key)
        if not m:
            continue
        rtol, hmax, h0 = m.groups()
        rk.setdefault((float(hmax), float(h0)), {})[float(rtol)] = v

    diamonds = []  # (x, y) for rtol=0.01 where it deviates
    for (hmax, h0) in sorted(rk):
        cell = rk[(hmax, h0)]
        # sanity: rtol=0.1 and rtol=1.0 must be bit-identical
        assert cell[0.1]["mean_final_loss"] == cell[1.0]["mean_final_loss"], \
            (lr, hmax, h0)
        v = cell[0.1]
        labels.append(f"RK3(2)-Adam  hmax={hmax:g}x h0={h0:g}x  (rtol 0.1 = 1.0)")
        means.append(v["mean_final_loss"])
        stds.append(v["std_final_loss"])
        colors.append("#1f77b4")
        v01 = cell[0.01]["mean_final_loss"]
        if v01 != v["mean_final_loss"]:
            diamonds.append((v01, len(labels) - 1))

    y = np.arange(len(labels))
    ax.barh(y, means, xerr=stds, color=colors, height=0.72,
            error_kw=dict(lw=1, capsize=2))
    if diamonds:
        dx, dy = zip(*diamonds)
        ax.plot(dx, dy, "D", color="#ff7f0e", ms=4, mec="black", mew=0.4,
                ls="none", label="rtol=0.01 (first-step transient)", zorder=5)
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Final full-batch train loss (log scale)", labelpad=18)
    ax.set_title(f"lr = {lr}")
    ax.grid(axis="x", which="both", alpha=0.25, lw=0.5)

    # Adam reference line
    adam = results[f"Adam lr={lr}"]["mean_final_loss"]
    ax.axvline(adam, color="#d62728", ls="--", lw=1, alpha=0.7)
    ax.annotate(f"Adam {adam:.6f}", xy=(adam, len(labels) - 0.6),
                fontsize=7.5, color="#d62728", ha="center", va="top",
                xytext=(adam, len(labels) + 1.3), annotation_clip=False)

    # annotate value on each bar
    for yi, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f" {m:.6f}", xy=(m, yi), fontsize=7,
                    va="center", ha="left", color="#333333")

fig.suptitle(
    "Compute-matched full-batch comparison (1024-ex MNIST subset, 600 gradient evals, "
    "3 seeds, mean ± std)\nEach RK bar collapses rtol = 0.1 and 1.0 (bit-identical "
    "trajectories — the error controller is inert, Sec. 5.1)\nDiamonds mark the only "
    "rtol effect: rtol=0.01's smaller first pre-saturation step",
    fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("comparison.png", dpi=200)
print("Saved comparison.png")
