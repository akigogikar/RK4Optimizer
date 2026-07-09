"""Full-batch experiment: the regime where RK methods *should* win.

With a deterministic (full-batch) gradient, the loss landscape defines a true
ODE vector field, so:
  - FSAL is valid again (k1 of step n+1 == k4 of step n, same vector field),
    so an accepted RK3(2) step costs 3 fresh evals (or fewer with FSAL).
  - The embedded error estimate measures real truncation error, so the
    adaptive controller is meaningful (unlike the minibatch case where it
    never rejected a step).

Protocol: identical eval-counting methodology to mnist_experiment.py.
  - MNIST subset (default 1024 train examples) so full-batch is tractable.
  - Every method gets the SAME gradient-evaluation budget.
  - Metrics: final training loss (primary: this tests the *optimizer*, i.e.
    integration of gradient flow) and test accuracy (secondary).
  - >=3 seeds, mean +/- std.

Usage:
  python3 fullbatch_experiment.py --budget 600
"""
import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from RK4Optimizer import AdaptiveEmbeddedRK3Optimizer

DEVICE = "cpu"


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
    ).to(DEVICE)


def get_data(n_train, seed=0):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_ds), generator=g)[:n_train]
    xs = torch.stack([train_ds[i][0] for i in idx.tolist()])
    ys = torch.tensor([train_ds[i][1] for i in idx.tolist()])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)
    return (xs.to(DEVICE), ys.to(DEVICE)), test_loader


def evaluate(model, loader):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            n += y.numel()
    model.train()
    return correct / n


def full_loss(model, x, y):
    return F.cross_entropy(model(x), y)


def run_adam(seed, budget, lr, data, test_loader, weight_decay=0.0):
    (x, y), _ = data, None
    model = make_model(seed)
    opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
           if weight_decay else torch.optim.Adam(model.parameters(), lr=lr))
    evals = 0
    while evals < budget:
        opt.zero_grad()
        loss = full_loss(model, x, y)
        loss.backward()
        opt.step()
        evals += 1
    with torch.no_grad():
        final = full_loss(model, x, y).item()
    return final, evaluate(model, test_loader), evals


def run_torch(seed, budget, opt_ctor, data, test_loader):
    """Generic full-batch runner for any torch.optim optimizer. One gradient
    evaluation == one full-batch backward, so every method is billed 1 eval/step
    and all optimizers share the SAME compute budget."""
    (x, y) = data
    model = make_model(seed)
    opt = opt_ctor(model.parameters())
    evals = 0
    while evals < budget:
        opt.zero_grad()
        full_loss(model, x, y).backward()
        opt.step()
        evals += 1
    with torch.no_grad():
        final = full_loss(model, x, y).item()
    return final, evaluate(model, test_loader), evals


def run_adam(seed, budget, lr, data, test_loader, weight_decay=0.0):
    ctor = (lambda p: torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay)) \
        if weight_decay else (lambda p: torch.optim.Adam(p, lr=lr))
    return run_torch(seed, budget, ctor, data, test_loader)


def run_gd(seed, budget, lr, data, test_loader, momentum=0.0, nesterov=False):
    """Plain/heavy-ball full-batch gradient descent (Euler / momentum on the
    gradient flow) baseline."""
    return run_torch(
        seed, budget,
        lambda p: torch.optim.SGD(p, lr=lr, momentum=momentum, nesterov=nesterov),
        data, test_loader)


# Registry of extra torch baselines: name-suffix -> optimizer constructor(lr).
# These share the identical gradient-eval budget so the comparison stays fair.
EXTRA_OPTIMIZERS = {
    "RMSprop":       lambda lr: (lambda p: torch.optim.RMSprop(p, lr=lr)),
    "Adagrad":       lambda lr: (lambda p: torch.optim.Adagrad(p, lr=lr)),
    "NAdam":         lambda lr: (lambda p: torch.optim.NAdam(p, lr=lr)),
    "RAdam":         lambda lr: (lambda p: torch.optim.RAdam(p, lr=lr)),
    "SGD+mom0.9":    lambda lr: (lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9)),
    "Nesterov0.9":   lambda lr: (lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9,
                                                           nesterov=True)),
}


def run_rk3(seed, budget, lr, data, test_loader, rtol=1e-2,
            h_max_scale=2.0, h0=None):
    """FSAL ENABLED: full-batch gradient is a genuine autonomous vector
    field, so caching k1 across steps is mathematically valid here.

    Returns (final_loss, test_acc, grad_evals, ctrl) where ctrl exposes the
    step-size controller diagnostics so we can see whether rtol/h0 actually
    influenced the run or the step saturated at the h_max ceiling."""
    (x, y) = data
    model = make_model(seed)
    opt = AdaptiveEmbeddedRK3Optimizer(model, lr=lr, rtol=rtol,
                                       h_max_scale=h_max_scale, h0=h0)
    loss_fn = nn.CrossEntropyLoss()
    t = 0
    while opt.grad_evals < budget - 3:
        t += 1
        opt.step(loss_fn, x, y, t)
    with torch.no_grad():
        final = full_loss(model, x, y).item()
    hh = opt.h_history
    ctrl = {
        "final_h": (hh[-1] if hh else None),
        "mean_h": (sum(hh) / len(hh) if hh else None),
        "min_h": (min(hh) if hh else None),
        "max_h": (max(hh) if hh else None),
        "h_max": opt.h_max, "h_min": opt.h_min,
        "frac_at_hmax": (opt.n_saturated_max / len(hh) if hh else None),
        "frac_at_hmin": (opt.n_saturated_min / len(hh) if hh else None),
        "steps": len(hh),
    }
    return final, evaluate(model, test_loader), opt.grad_evals, ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600,
                    help="full-batch gradient evaluations per seed per method")
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--rtols", type=float, nargs="+", default=[1e-2, 1e-1])
    ap.add_argument("--h0s", type=float, nargs="+", default=None,
                    help="initial integration steps to sweep for RK3(2); "
                         "expressed as multiples of lr. Enables the h0 sweep.")
    ap.add_argument("--h-max-scales", type=float, nargs="+", default=[2.0],
                    help="h_max ceiling as a multiple of lr. The 2.0 default "
                         "saturates the controller (see README); larger values "
                         "let rtol/h0 actually take effect.")
    ap.add_argument("--no-extra-optims", action="store_true",
                    help="skip RMSprop/Adagrad/NAdam/RAdam/Nesterov baselines")
    args = ap.parse_args()

    data, test_loader = get_data(args.n_train)
    results = {}

    def record(name, losses, accs, evals, ctrl=None):
        ml = sum(losses) / len(losses)
        sl = (sum((l - ml) ** 2 for l in losses) / max(len(losses) - 1, 1)) ** 0.5
        ma = sum(accs) / len(accs)
        results[name] = {"mean_final_loss": ml, "std_final_loss": sl,
                         "losses": losses, "mean_test_acc": ma,
                         "accs": accs, "evals": evals}
        if ctrl is not None:
            results[name]["controller"] = ctrl
        extra = ""
        if ctrl and ctrl.get("frac_at_hmax") is not None:
            extra = (f"  [h: mean={ctrl['mean_h']:.2e} final={ctrl['final_h']:.2e}"
                     f" @hmax={ctrl['frac_at_hmax']*100:.0f}%"
                     f" @hmin={ctrl['frac_at_hmin']*100:.0f}%]")
        print(f"{name:44s} loss {ml:.6f} +/- {sl:.6f}  "
              f"acc {ma*100:5.2f}%  ({evals} grad evals){extra}", flush=True)

    runners = []
    for lr in args.lrs:
        runners.append((f"GD lr={lr}", lambda s, lr=lr: run_gd(
            s, args.budget, lr, data, test_loader)))
        runners.append((f"Adam lr={lr}", lambda s, lr=lr: run_adam(
            s, args.budget, lr, data, test_loader)))
        runners.append((f"AdamW(wd=1e-2) lr={lr}", lambda s, lr=lr: run_adam(
            s, args.budget, lr, data, test_loader, weight_decay=1e-2)))
        if not args.no_extra_optims:
            for oname, ctor_factory in EXTRA_OPTIMIZERS.items():
                runners.append((f"{oname} lr={lr}",
                                lambda s, lr=lr, cf=ctor_factory: run_torch(
                                    s, args.budget, cf(lr), data, test_loader)))
        # RK3(2): sweep rtol x h_max_scale x (optional) h0 so the controller
        # regime is fully mapped instead of silently pinned at the ceiling.
        h0_mults = args.h0s if args.h0s is not None else [None]
        for hms in args.h_max_scales:
            for rtol in args.rtols:
                for h0m in h0_mults:
                    h0 = (None if h0m is None else h0m * lr)
                    tag = (f"RK3(2)-Adam lr={lr} rtol={rtol} hmax={hms}x"
                           + ("" if h0m is None else f" h0={h0m}x"))
                    runners.append((tag,
                                    lambda s, lr=lr, rtol=rtol, hms=hms, h0=h0:
                                    run_rk3(s, args.budget, lr, data, test_loader,
                                            rtol=rtol, h_max_scale=hms, h0=h0)))

    for name, fn in runners:
        losses, accs, ev, ctrl = [], [], 0, None
        for seed in args.seeds:
            out = fn(seed)
            if len(out) == 4:
                l, a, ev, ctrl = out
            else:
                l, a, ev = out
            losses.append(l); accs.append(a)
        record(name, losses, accs, ev, ctrl)

    with open("fullbatch_results.json", "w") as f:
        json.dump({"budget": args.budget, "n_train": args.n_train,
                   "seeds": args.seeds, "results": results}, f, indent=2)
    print("\nSaved -> fullbatch_results.json")

    best_rk = min((v["mean_final_loss"], k) for k, v in results.items() if "RK3" in k)
    best_bl = min((v["mean_final_loss"], k) for k, v in results.items() if "RK3" not in k)
    print(f"\nBest RK3(2) loss:    {best_rk[1]} @ {best_rk[0]:.6f}")
    print(f"Best baseline loss:  {best_bl[1]} @ {best_bl[0]:.6f}")
    print("Verdict:", "RK3(2) WINS at equal compute (full-batch)"
          if best_rk[0] < best_bl[0] else
          "Baseline wins even full-batch (strengthens negative result)")


if __name__ == "__main__":
    main()
