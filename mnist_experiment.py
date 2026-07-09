"""MNIST compute-matched benchmark: embedded RK3(2) error-controlled Adam
vs. Adam/AdamW at equal gradient-evaluation budgets.

Protocol (addresses the fairness critique in Zhang et al. 2018 reviews and
follows the eval-counting methodology of FlowAdam / IMEX Adam):
  - Every optimizer gets the SAME total gradient-evaluation budget.
  - RK3(2) uses 4 evals/step in the stochastic setting (FSAL is disabled:
    the cached k1 was computed on the previous minibatch, so reusing it
    would silently change the method; we re-evaluate honestly).
  - Adam baselines therefore take 4x more steps at the same budget.
  - >=3 seeds, mean +/- std test accuracy reported.

Usage:
  python3 mnist_experiment.py                 # quick: 1 epoch-equivalent budget
  python3 mnist_experiment.py --budget 8000   # larger budget (grad evals/seed)
"""
import argparse, copy, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from RK4Optimizer import AdaptiveEmbeddedRK3Optimizer

DEVICE = "cpu"  # keep deterministic and simple; MPS optional later


class StochasticRK3Optimizer(AdaptiveEmbeddedRK3Optimizer):
    """Minibatch-honest variant: disables FSAL (previous batch's gradient is
    not the current batch's k1), so every step costs 4 fresh grad evals,
    all computed on the SAME minibatch for a consistent vector field."""

    def step(self, loss_fn, x, y, t):
        self.k1 = None  # invalidate FSAL cache across minibatches
        super().step(loss_fn, x, y, t)


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
    ).to(DEVICE)


def evaluate(model, loader):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            n += y.numel()
    model.train()
    return correct / n


def run_adam(seed, budget, lr, loaders, weight_decay=0.0):
    train_loader, test_loader = loaders
    model = make_model(seed)
    opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
           if weight_decay else torch.optim.Adam(model.parameters(), lr=lr))
    evals = 0
    it = iter(train_loader)
    while evals < budget:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        evals += 1
    return evaluate(model, test_loader), evals


def run_rk3(seed, budget, lr, loaders, rtol=1e-2):
    train_loader, test_loader = loaders
    model = make_model(seed)
    opt = StochasticRK3Optimizer(model, lr=lr, rtol=rtol)
    loss_fn = nn.CrossEntropyLoss()
    t = 0
    it = iter(train_loader)
    while opt.grad_evals < budget - 3:  # each step consumes 4 evals
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        t += 1
        opt.step(loss_fn, x, y, t)
    return evaluate(model, test_loader), opt.grad_evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2000,
                    help="gradient evaluations per seed per method")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--rtols", type=float, nargs="+", default=[1e-2, 1e-1])
    args = ap.parse_args()

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)

    results = {}

    def record(name, accs, evals):
        m = sum(accs) / len(accs)
        s = (sum((a - m) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
        results[name] = {"mean_acc": m, "std_acc": s, "accs": accs, "evals": evals}
        print(f"{name:42s} acc {m*100:6.2f} +/- {s*100:4.2f}%  ({evals} grad evals)")

    for lr in args.lrs:
        for method, fn, kw in [("Adam", run_adam, {}),
                               ("AdamW(wd=1e-2)", run_adam, {"weight_decay": 1e-2})]:
            accs, ev = [], 0
            for seed in args.seeds:
                torch.manual_seed(seed)
                g = torch.Generator().manual_seed(seed)
                tl = torch.utils.data.DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
                a, ev = fn(seed, args.budget, lr, (tl, test_loader), **kw)
                accs.append(a)
            record(f"{method} lr={lr}", accs, ev)

        for rtol in args.rtols:
            accs, ev = [], 0
            for seed in args.seeds:
                torch.manual_seed(seed)
                g = torch.Generator().manual_seed(seed)
                tl = torch.utils.data.DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
                a, ev = run_rk3(seed, args.budget, lr, (tl, test_loader), rtol=rtol)
                accs.append(a)
            record(f"RK3(2)-Adam lr={lr} rtol={rtol}", accs, ev)

    with open("mnist_results.json", "w") as f:
        json.dump({"budget": args.budget, "seeds": args.seeds,
                   "batch_size": args.batch_size, "results": results}, f, indent=2)
    print("\nSaved -> mnist_results.json")

    best_rk = max((v["mean_acc"], k) for k, v in results.items() if "RK3" in k)
    best_ad = max((v["mean_acc"], k) for k, v in results.items() if "RK3" not in k)
    print(f"\nBest RK3(2): {best_rk[1]} @ {best_rk[0]*100:.2f}%")
    print(f"Best Adam*:  {best_ad[1]} @ {best_ad[0]*100:.2f}%")
    print("Verdict:", "RK3(2) WINS at equal compute" if best_rk[0] > best_ad[0]
          else "Adam wins at equal compute (honest negative result)")


if __name__ == "__main__":
    main()
