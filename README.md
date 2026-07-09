# Adaptive Runge–Kutta Step Control Buys Training Loss, Not Generalization

**An Honest Compute-Matched Study of RK-Adam Optimizers**

This repository contains the optimizer implementation, experiments, result
data, figures, and full paper for a compute-matched study of Runge–Kutta (RK)
style adaptive-step optimizers for neural-network training.

> **TL;DR — this is a negative / mixed result, reported honestly.**
> Interpreting optimizers as discretizations of gradient flow motivates using
> higher-order Runge–Kutta integrators with adaptive step-size control. We build
> a representative Adam variant driven by a Bogacki–Shampine 3(2) embedded RK
> pair (FSAL reuse + local-error step controller) and evaluate it under a
> **strict compute-matched protocol where every method gets the same number of
> gradient evaluations** — an accounting the RK-optimizer literature typically
> does not enforce. Under that protocol the RK variant **consistently loses to
> plain Adam on training loss**, and its "adaptivity" turns out to be largely
> illusory. Where a repaired controller *does* recover a genuine mechanism, the
> benefit is a warm-up-and-growth step schedule that helps training loss but
> **does not transfer to test accuracy**.

## Key findings

- **Compute-matched, RK loses on training loss.** Given equal gradient-evaluation
  budgets, the RK-Adam variant is beaten by plain Adam on training loss in both
  stochastic minibatch training (MNIST) and full-batch training on an MNIST
  subset — the regime most favorable to RK methods.
- **The "adaptivity" is illusory under noise.** Instrumenting the controller
  shows that under minibatch-gradient noise the truncation-error estimate is
  dominated by sampling noise, so the adaptive step controller does not track
  anything meaningful and effectively reduces to a fixed step.
- **A repaired controller reveals a real but narrow mechanism.** With a
  compute-matched, honestly-accounted controller, the only thing that helps is
  an emergent warm-up-and-growth step-size schedule (`h_max`-driven), which
  lowers training loss but **does not improve generalization** (test accuracy is
  flat vs. Adam at matched compute).
- **Temperature / SGLD framing (pre-registered H1/H0).** A pre-registered
  temperature-sweep test returns the **null (H0) on both runs** — no evidence
  that the RK step control acts as a useful implicit-regularization temperature.
- Bottom line: **step control buys training loss, not generalization.**

## Repository structure

```
RK4Optimizer.py                     Optimizer implementations (incl. self-test)
mnist_experiment.py                 Experiment 1  — minibatch MNIST, compute-matched
fullbatch_experiment.py             Experiment 2  — full-batch MNIST subset
fixed_controller_experiment.py      Experiment 3  — repaired-controller ablation (Table 3)
hmax_control_experiment.py          Experiment 3  — h_max control (Table 4)
temperature_sweep_experiment.py     Experiment 4  — temperature / SGLD sweep (Sec. 6.1)
make_comparison_figure.py           Regenerates the comparison figure

PAPER_DRAFT.md                      Full paper (source)
RK4Optimizer_paper.pdf              Full paper (typeset PDF)
build_pdf.sh                        Markdown -> PDF build script

*_results.json                      Committed result data for every experiment
comparison.png                      Main comparison figure
```

All result JSON files referenced by the paper are committed so every table and
figure is directly checkable. Runs are CPU-only and deterministically seeded.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12 and the exact versions pinned in `requirements.txt`
(`torch==2.2.2`, `torchvision==0.17.2`, `numpy==1.26.4`, `matplotlib==3.8.2`).
Datasets are downloaded automatically by `torchvision` on first run.

## Reproducing the results

```bash
# Sanity check: optimizer self-test
python3 RK4Optimizer.py

# Experiment 1 — minibatch MNIST, compute-matched
python3 mnist_experiment.py --budget 4000

# Experiment 2 — full-batch MNIST subset
python3 fullbatch_experiment.py --budget 600

# Experiment 3 — repaired-controller ablation (Table 3) + h_max control (Table 4)
python3 fixed_controller_experiment.py --budget 600
python3 hmax_control_experiment.py

# Multi-seed chase (Section 4)
python3 fullbatch_experiment.py --budget 600 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --lrs 0.003 --rtols 0.01 --h-max-scales 2.0 8.0 --h0s 1.0

# Experiment 4 — temperature / SGLD sweep (Test 2, n=3)
python3 temperature_sweep_experiment.py --budget 2000 --seeds 0 1 2 \
    --temps 0.0 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 \
    --sgld --out temperature_sweep_results.json

# Depth/overfitting premise (Test 1, n=5) and single-seed T=0 trajectory
python3 temperature_sweep_experiment.py --budget 6000 --seeds 0 1 2 3 4 \
    --temps 0.0 --out temperature_premise_results.json
python3 temperature_sweep_experiment.py --budget 6000 --seeds 0 \
    --temps 0.0 --out temperature_premise_trajectory.json

# Regenerate the comparison figure
python3 make_comparison_figure.py
```

The pre-registered H1/H0 criteria and the automated verdict for Experiment 4
(H0 on both runs) live in the `temperature_sweep_experiment.py` module
docstring / `main()`.

## Paper

- **PDF:** [`RK4Optimizer_paper.pdf`](RK4Optimizer_paper.pdf)
- **Source:** [`PAPER_DRAFT.md`](PAPER_DRAFT.md)
- **Rebuild the PDF:** `./build_pdf.sh` (requires `pandoc` + a LaTeX engine)

## Citation

```bibtex
@misc{gogikar2026rkadam,
  title        = {Adaptive Runge--Kutta Step Control Buys Training Loss, Not
                  Generalization: An Honest Compute-Matched Study of RK-Adam
                  Optimizers},
  author       = {Gogikar, Akhilesh},
  year         = {2026},
  howpublished = {\url{https://github.com/akigogikar/RK4Optimizer}},
  note         = {Compute-matched evaluation of Runge--Kutta step-control optimizers}
}
```

## License

Licensed under the **Apache License, Version 2.0** — see [`LICENSE`](LICENSE)
and [`NOTICE`](NOTICE). Copyright 2026 Akhilesh Gogikar.
