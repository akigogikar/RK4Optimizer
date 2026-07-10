"""Generate all paper figures into figs/ from the result JSONs.

Inputs: trajectories.json, fullbatch_n10_extended_results.json,
        fixed_controller_results.json, temperature_sweep_results.json
Outputs: figs/fig_controller.pdf, figs/fig_comparison.pdf,
         figs/fig_fragility.pdf, figs/fig_temperature.pdf
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight",
})
os.makedirs("figs", exist_ok=True)

SAT = (0.9 / 2.0) ** 3  # growth-factor saturation threshold ~0.0911


def fig_controller():
    d = json.load(open("trajectories.json"))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.9))

    ax = axes[0]
    for key, lab, c in (("as_designed_hmax2", r"$h_{\max}=2\,\mathrm{lr}$ (default)", "C0"),
                        ("as_designed_hmax8", r"$h_{\max}=8\,\mathrm{lr}$", "C1")):
        r = d[key]
        h = np.array(r["h"]) / r["h_max"]
        ax.plot(np.arange(1, len(h) + 1), h, c, lw=1, label=lab)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$h_t / h_{\max}$")
    ax.set_title("(a) as-designed: $h$ pinned at ceiling")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    for key, lab, c in (("as_designed_hmax2", r"$h_{\max}=2\,\mathrm{lr}$", "C0"),
                        ("as_designed_hmax8", r"$h_{\max}=8\,\mathrm{lr}$", "C1")):
        r = d[key]
        ax.semilogy(np.arange(1, len(r["err"]) + 1), r["err"], c, lw=1, label=lab)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axhline(SAT, color="0.4", ls=":", lw=0.8)
    ax.text(0.98, 1.25, "tolerance", ha="right", fontsize=6.5, color="k",
            transform=ax.get_yaxis_transform())
    ax.text(0.98, SAT * 1.25, "growth saturation", ha="right", fontsize=6.5,
            color="0.35", transform=ax.get_yaxis_transform())
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"normalized error $\widehat{\mathrm{err}}_t$")
    ax.set_title("(b) error signal vs. thresholds")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[2]
    for key, lab, c in (("repaired_rtol0.1", r"$\rho=0.1$", "C2"),
                        ("repaired_rtol0.01", r"$\rho=0.01$", "C3")):
        r = d[key]
        h = np.array(r["h"])
        steps = np.arange(1, len(h) + 1)
        ax.semilogy(steps, h, c, lw=1, label=lab)
        rej = np.array(r["rejects_at"], dtype=int)
        if len(rej):
            ax.plot(rej + 1, h[rej], "x", color=c, ms=4, mew=1.2)
    ax.set_xlabel("step $t$")
    ax.set_ylabel(r"$h_t$")
    ax.set_title(r"(c) repaired: rejects ($\times$) and adaptation")
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout(pad=0.4)
    fig.savefig("figs/fig_controller.pdf")
    plt.close(fig)


def fig_comparison():
    d = json.load(open("fullbatch_n10_extended_results.json"))
    res = d["results"]
    names = list(res.keys())
    accs = np.array([res[n]["mean_test_acc"] for n in names])
    stds = np.array([np.std(res[n]["accs"]) for n in names])
    order = np.argsort(accs)
    names = [names[i] for i in order]
    accs, stds = accs[order], stds[order]

    def color(n):
        if "RK3" in n or "RK4" in n:
            return "C3" if ("Adaptive" in n or "Embedded" in n) else "C1"
        if "Adam" in n:
            return "C0"
        return "0.6"

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    y = np.arange(len(names))
    ax.barh(y, accs, xerr=stds, height=0.65,
            color=[color(n) for n in names], error_kw=dict(lw=0.8))
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("test accuracy (10 seeds, 600 grad evals)")
    lo = max(0.0, accs.min() - 0.05)
    ax.set_xlim(lo, min(1.0, accs.max() + 0.03))
    for yi, (a, s) in enumerate(zip(accs, stds)):
        ax.text(a + s + 0.003, yi, f"{a:.3f}", va="center", fontsize=6)
    fig.savefig("figs/fig_comparison.pdf")
    plt.close(fig)


def fig_fragility():
    d = json.load(open("fixed_controller_results.json"))
    res = d["results"]
    pat = re.compile(r"(PureRK3\(2\)|FixedRK3Adam) h0=([\d.e-]+) rtol=([\d.e-]+)")
    series = {}
    adam = [v["mean_final_loss"] for k, v in res.items() if k.startswith("Adam")]
    for k, v in res.items():
        m = pat.match(k)
        if not m:
            continue
        fam, h0, rtol = m.group(1), float(m.group(2)), float(m.group(3))
        series.setdefault((fam, rtol), []).append(
            (h0, v["mean_final_loss"], np.std(v["losses"])))

    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    markers = {"PureRK3(2)": "o", "FixedRK3Adam": "s"}
    ci = 0
    for (fam, rtol), pts in sorted(series.items()):
        pts.sort()
        h0s = [p[0] for p in pts]
        ls = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(h0s, ls, yerr=es, marker=markers.get(fam, "o"), ms=3.5,
                    lw=1, capsize=2,
                    label=fr"{fam}, $\rho$={rtol:g}", color=f"C{ci}")
        ci += 1
    if adam:
        ax.axhline(min(adam), color="k", ls="--", lw=0.8, label="best Adam")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"initial step size $h_0$")
    ax.set_ylabel("final training loss")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig("figs/fig_fragility.pdf")
    plt.close(fig)


def fig_temperature():
    d = json.load(open("temperature_sweep_results.json"))
    res = d["results"]
    pts = []
    for k, v in res.items():
        if not k.startswith("pSGLD"):
            continue
        T = float(v["T"])
        acc = v.get("mean_test_acc")
        loss = v["mean_final_loss"]
        lstd = v.get("std_final_loss", 0.0)
        astd = np.std(v["accs"]) if "accs" in v else 0.0
        pts.append((T, loss, lstd, acc, astd))
    pts.sort()
    T = np.array([p[0] for p in pts])
    L = np.array([p[1] for p in pts])
    Ls = np.array([p[2] for p in pts])
    have_acc = all(p[3] is not None for p in pts)

    ncol = 2 if have_acc else 1
    fig, axes = plt.subplots(1, ncol, figsize=(3.3 * ncol, 2.0), squeeze=False)
    Tplot = np.where(T == 0, np.min(T[T > 0]) / 10 if (T > 0).any() else 1e-12, T)
    ax = axes[0][0]
    ax.errorbar(Tplot, L, yerr=Ls, marker="o", ms=3.5, lw=1, capsize=2, color="C0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if (T == 0).any():
        i0 = int(np.argmin(T))
        ax.annotate("$T=0$", (Tplot[i0], L[i0]), textcoords="offset points",
                    xytext=(4, 6), fontsize=7)
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel("final training loss")
    if have_acc:
        A = np.array([p[3] for p in pts])
        As = np.array([p[4] for p in pts])
        ax2 = axes[0][1]
        ax2.errorbar(Tplot, A, yerr=As, marker="o", ms=3.5, lw=1, capsize=2, color="C1")
        ax2.set_xscale("log")
        ax2.set_xlabel("temperature $T$")
        ax2.set_ylabel("test accuracy")
    fig.tight_layout(pad=0.4)
    fig.savefig("figs/fig_temperature.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_controller()
    fig_comparison()
    fig_fragility()
    fig_temperature()
    print("Wrote:", sorted(os.listdir("figs")))
