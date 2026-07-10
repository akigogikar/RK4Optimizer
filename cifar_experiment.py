"""CIFAR-10 / CNN replication of the full-batch compute-matched protocol.

Second dataset+architecture requested for ICML-level generality: replicates
(1) the as-designed RK3(2)-Adam negative result + inert-controller diagnosis
    (frac_at_hmax, rtol-insensitivity),
(2) the repaired-controller (FixedRK3Adam / PureRK3) comparison,
on CIFAR-10 with a small CNN, under the identical eval-billing rules as
fullbatch_experiment.py (1 full-batch backward == 1 eval; RK stages and
rejected steps all billed; FSAL credited only where mathematically valid).

Usage:
  python3 cifar_experiment.py --budget 600 --seeds 0 1 2 3 4 \
      --shard 0 --num-shards 3 --out cifar_results_shard0.json
Shards partition the config list round-robin so shards can run in parallel;
merge with merge_cifar_results.py (trivial dict union).
"""
import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from RK4Optimizer import AdaptiveEmbeddedRK3Optimizer
from fixed_controller_experiment import PureRK3, FixedRK3Adam

DEVICE = "cpu"  # deterministic; keeps parity with the MNIST protocol


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
        nn.Linear(128, 10),
    ).to(DEVICE)


def get_data(n_train, seed=0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))])
    train_ds = datasets.CIFAR10("data", train=True, download=True, transform=tf)
    test_ds = datasets.CIFAR10("data", train=False, download=True, transform=tf)
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


def run_torch(seed, budget, opt_ctor, data, test_loader):
    (x, y) = data
    model = make_model(seed)
    opt = opt_ctor(model.parameters())
    evals = 0
    while evals < budget:
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        evals += 1
    with torch.no_grad():
        final = F.cross_entropy(model(x), y).item()
    return final, evaluate(model, test_loader), evals, None


def run_rk3_asdesigned(seed, budget, lr, rtol, h_max_scale, data, test_loader):
    (x, y) = data
    model = make_model(seed)
    opt = AdaptiveEmbeddedRK3Optimizer(model, lr=lr, rtol=rtol,
                                       h_max_scale=h_max_scale)
    loss_fn = nn.CrossEntropyLoss()
    t = 0
    while opt.grad_evals < budget - 3:
        t += 1
        opt.step(loss_fn, x, y, t)
    with torch.no_grad():
        final = F.cross_entropy(model(x), y).item()
    hh = opt.h_history
    ctrl = {"final_h": hh[-1] if hh else None,
            "mean_h": sum(hh) / len(hh) if hh else None,
            "min_h": min(hh) if hh else None,
            "max_h": max(hh) if hh else None,
            "h_max": opt.h_max, "h_min": opt.h_min,
            "frac_at_hmax": opt.n_saturated_max / len(hh) if hh else None,
            "frac_at_hmin": opt.n_saturated_min / len(hh) if hh else None,
            "steps": len(hh)}
    return final, evaluate(model, test_loader), opt.grad_evals, ctrl


def run_repaired(cls, seed, budget, h0, rtol, data, test_loader):
    (x, y) = data
    model = make_model(seed)
    opt = cls(model, h0=h0, rtol=rtol)
    loss_fn = nn.CrossEntropyLoss()
    t = 0
    while opt.grad_evals < budget - 4:
        t += 1
        opt.step(loss_fn, x, y, t)
    with torch.no_grad():
        final = F.cross_entropy(model(x), y).item()
    ctrl = {"rejects": opt.rejects, "final_h": opt.h}
    return final, evaluate(model, test_loader), opt.grad_evals, ctrl


def build_configs(budget, data, test_loader):
    cfgs = []
    for lr in (1e-3, 3e-3):
        for name, ctor in [
            ("GD",          lambda p, lr=lr: torch.optim.SGD(p, lr=lr)),
            ("SGD+mom0.9",  lambda p, lr=lr: torch.optim.SGD(p, lr=lr, momentum=0.9)),
            ("Adam",        lambda p, lr=lr: torch.optim.Adam(p, lr=lr)),
            ("AdamW(wd=1e-2)", lambda p, lr=lr: torch.optim.AdamW(p, lr=lr, weight_decay=1e-2)),
            ("RMSprop",     lambda p, lr=lr: torch.optim.RMSprop(p, lr=lr)),
            ("NAdam",       lambda p, lr=lr: torch.optim.NAdam(p, lr=lr)),
        ]:
            cfgs.append((f"{name} lr={lr}",
                         lambda s, c=ctor: run_torch(s, budget, c, data, test_loader)))
        for rtol in (0.01, 0.1, 1.0):
            for hms in (2.0, 8.0):
                cfgs.append((f"RK3(2)-Adam lr={lr} rtol={rtol} hmax={hms}x",
                             lambda s, lr=lr, r=rtol, m=hms:
                             run_rk3_asdesigned(s, budget, lr, r, m, data, test_loader)))
    for h0 in (1e-3, 3e-3):
        for rtol in (0.01, 0.1):
            cfgs.append((f"FixedRK3Adam h0={h0} rtol={rtol}",
                         lambda s, h=h0, r=rtol:
                         run_repaired(FixedRK3Adam, s, budget, h, r, data, test_loader)))
    cfgs.append(("PureRK3(2) h0=0.1 rtol=0.01",
                 lambda s: run_repaired(PureRK3, s, budget, 0.1, 0.01, data, test_loader)))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="cifar_results.json")
    args = ap.parse_args()

    data, test_loader = get_data(args.n_train)
    cfgs = build_configs(args.budget, data, test_loader)
    cfgs = [c for i, c in enumerate(cfgs) if i % args.num_shards == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(cfgs)} configs "
          f"x {len(args.seeds)} seeds", flush=True)

    results = {}
    for name, fn in cfgs:
        rows = [fn(s) for s in args.seeds]
        losses = [r[0] for r in rows]; accs = [r[1] for r in rows]
        ml = sum(losses) / len(losses)
        sl = (sum((l - ml) ** 2 for l in losses) / max(len(losses) - 1, 1)) ** 0.5
        entry = {"mean_final_loss": ml, "std_final_loss": sl, "losses": losses,
                 "mean_test_acc": sum(accs) / len(accs), "accs": accs,
                 "evals": rows[-1][2]}
        if rows[-1][3] is not None:
            entry["ctrl"] = rows[-1][3]
        results[name] = entry
        extra = ""
        if rows[-1][3] and rows[-1][3].get("frac_at_hmax") is not None:
            extra = f"  @hmax={rows[-1][3]['frac_at_hmax']*100:.0f}%"
        print(f"{name:48s} loss {ml:.6f} +/- {sl:.6f}  "
              f"acc {entry['mean_test_acc']*100:5.2f}%{extra}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"budget": args.budget, "n_train": args.n_train,
                   "seeds": args.seeds, "shard": args.shard,
                   "num_shards": args.num_shards, "results": results}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
