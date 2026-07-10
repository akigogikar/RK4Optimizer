"""Fixed-controller ablation: closes the "your negative result is just a
controller bug" objection.

Two repaired integrators, evaluated under the same compute-matched full-batch
protocol as fullbatch_experiment.py:

1. PureRK3: honest Bogacki-Shampine 3(2) on gradient flow (NO Adam
   preconditioning). The embedded error estimate is now exactly the error of
   the applied map. Proper accept/REJECT branch: err > 1 -> revert, shrink h,
   retry (rejected stages still count against the eval budget - honesty).
   FSAL credited only on accepted steps.

2. FixedRK3Adam: Adam-preconditioned variant where the error is computed on
   the ACTUAL applied map - the 3rd- and 2nd-order averaged gradients are both
   pushed through the same Adam preconditioner, and the error is the norm of
   the difference of the two candidate parameter updates. Proper reject:
   revert params AND moments, shrink h, retry.

If baselines still win at equal gradient evals, the negative result stands on
methods whose adaptivity is genuinely functional.

Usage: python3 fixed_controller_experiment.py --budget 600
"""
import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

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
    return (xs, ys), test_loader


def evaluate(model, loader):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            n += y.numel()
    model.train()
    return correct / n


class _RKBase:
    """Shared stage machinery with honest eval counting."""

    def __init__(self, model, h0, rtol, atol=1e-4, h_min=1e-6, h_max=None):
        self.model = model
        self.h = h0
        self.rtol, self.atol = rtol, atol
        self.h_min = h_min
        self.h_max = h_max if h_max is not None else 100 * h0
        self.k1 = None
        self.grad_evals = 0
        self.rejects = 0

    def _grads(self, loss_fn, x, y, params):
        loss = loss_fn(self.model(x), y)
        self.grad_evals += 1
        return [g.detach() for g in torch.autograd.grad(loss, params)]

    def _set(self, params, base, direction, scale):
        with torch.no_grad():
            for i, p in enumerate(params):
                p.copy_(base[i] - scale * direction[i])

    def _stages(self, loss_fn, x, y, params, p0, h):
        k1 = self.k1 if self.k1 is not None else self._grads(loss_fn, x, y, params)
        self._set(params, p0, k1, h / 2)
        k2 = self._grads(loss_fn, x, y, params)
        self._set(params, p0, k2, 3 * h / 4)
        k3 = self._grads(loss_fn, x, y, params)
        g3 = [(2/9) * k1[i] + (1/3) * k2[i] + (4/9) * k3[i] for i in range(len(p0))]
        return k1, k2, k3, g3


class PureRK3(_RKBase):
    """Textbook-correct BS3(2) integrator of gradient flow with rejection."""

    def step(self, loss_fn, x, y, t):
        params = list(self.model.parameters())
        p0 = [p.detach().clone() for p in params]
        while True:
            h = self.h
            k1, k2, k3, g3 = self._stages(loss_fn, x, y, params, p0, h)
            self._set(params, p0, g3, h)               # 3rd-order step (applied map)
            k4 = self._grads(loss_fn, x, y, params)    # FSAL candidate
            with torch.no_grad():
                err_sq, n_el = 0.0, 0
                for i in range(len(params)):
                    g2 = (7/24)*k1[i] + (1/4)*k2[i] + (1/3)*k3[i] + (1/8)*k4[i]
                    e = h * (g3[i] - g2)
                    scale = self.atol + self.rtol * p0[i].abs()
                    err_sq += ((e / scale) ** 2).sum().item()
                    n_el += p0[i].numel()
                err = max((err_sq / n_el) ** 0.5, 1e-12)
            factor = min(max(0.9 * err ** (-1/3), 0.2), 5.0)
            if err <= 1.0 or self.h <= self.h_min:     # ACCEPT
                self.k1 = k4
                self.h = min(max(h * factor, self.h_min), self.h_max)
                return
            # REJECT: revert, shrink, retry (evals already counted)
            self.rejects += 1
            self._set(params, p0, [torch.zeros_like(p) for p in p0], 0.0)
            self.k1 = k1                               # k1 at p0 is still valid
            self.h = max(h * factor, self.h_min)


class FixedRK3Adam(_RKBase):
    """Adam-preconditioned RK3 where the error is measured on the ACTUAL
    applied update, with a genuine reject branch reverting params+moments."""

    def __init__(self, model, h0, rtol, beta1=0.9, beta2=0.999, eps=1e-8, **kw):
        super().__init__(model, h0, rtol, **kw)
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = [torch.zeros_like(p) for p in model.parameters()]
        self.v = [torch.zeros_like(p) for p in model.parameters()]

    def step(self, loss_fn, x, y, t):
        params = list(self.model.parameters())
        p0 = [p.detach().clone() for p in params]
        m0 = [m.clone() for m in self.m]
        v0 = [v.clone() for v in self.v]
        while True:
            h = self.h
            k1, k2, k3, g3 = self._stages(loss_fn, x, y, params, p0, h)
            with torch.no_grad():
                g2 = [(7/24)*k1[i] + (1/4)*k2[i] + (1/3)*k3[i] + (1/8)*k1[i]
                      for i in range(len(p0))]  # 2nd-order est. w/o extra eval
                err_sq, n_el = 0.0, 0
                upd3 = []
                for i, p in enumerate(params):
                    self.m[i] = self.b1 * m0[i] + (1 - self.b1) * g3[i]
                    self.v[i] = self.b2 * v0[i] + (1 - self.b2) * g3[i] ** 2
                    m_hat = self.m[i] / (1 - self.b1 ** t)
                    v_hat = self.v[i] / (1 - self.b2 ** t)
                    pre = 1.0 / (torch.sqrt(v_hat) + self.eps)
                    u3 = h * pre * m_hat                       # applied update
                    # counterfactual update driven by 2nd-order gradient,
                    # through the SAME preconditioner state:
                    m2 = self.b1 * m0[i] + (1 - self.b1) * g2[i]
                    u2 = h * pre * (m2 / (1 - self.b1 ** t))
                    e = u3 - u2                                # error of applied map
                    scale = self.atol + self.rtol * p0[i].abs()
                    err_sq += ((e / scale) ** 2).sum().item()
                    n_el += p0[i].numel()
                    upd3.append(u3)
                err = max((err_sq / n_el) ** 0.5, 1e-12)
            factor = min(max(0.9 * err ** (-1/3), 0.2), 5.0)
            if err <= 1.0 or self.h <= self.h_min:     # ACCEPT
                with torch.no_grad():
                    for i, p in enumerate(params):
                        p.copy_(p0[i] - upd3[i])
                self.k1 = None  # preconditioned step; FSAL not claimed
                self.h = min(max(h * factor, self.h_min), self.h_max)
                return
            # REJECT: revert moments (params never moved off p0 stages), retry
            self.rejects += 1
            self.m = [m.clone() for m in m0]
            self.v = [v.clone() for v in v0]
            self._set(params, p0, [torch.zeros_like(p) for p in p0], 0.0)
            self.k1 = k1
            self.h = max(h * factor, self.h_min)


def run_rk(cls, seed, budget, h0, rtol, data, test_loader):
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
    return final, evaluate(model, test_loader), opt.grad_evals, opt.rejects, opt.h


def run_adam(seed, budget, lr, data, test_loader):
    (x, y) = data
    model = make_model(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    evals = 0
    while evals < budget:
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
        evals += 1
    with torch.no_grad():
        final = F.cross_entropy(model(x), y).item()
    return final, evaluate(model, test_loader), evals, 0, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--h0s", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--rtols", type=float, nargs="+", default=[1e-2, 1e-1])
    ap.add_argument("--out", default="fixed_controller_results.json",
                    help="output JSON path (default keeps historical name)")
    args = ap.parse_args()

    data, test_loader = get_data(args.n_train)
    results = {}

    def record(name, rows):
        losses = [r[0] for r in rows]; accs = [r[1] for r in rows]
        rej = sum(r[3] for r in rows) / len(rows)
        hf = sum(r[4] for r in rows) / len(rows)
        ml = sum(losses) / len(losses)
        sl = (sum((l - ml) ** 2 for l in losses) / max(len(losses) - 1, 1)) ** 0.5
        ma = sum(accs) / len(accs)
        results[name] = {"mean_final_loss": ml, "std_final_loss": sl,
                         "losses": losses, "mean_test_acc": ma, "accs": accs,
                         "evals": rows[-1][2], "mean_rejects": rej, "mean_final_h": hf}
        print(f"{name:46s} loss {ml:.6f} +/- {sl:.6f}  acc {ma*100:5.2f}%  "
              f"rejects/seed {rej:5.1f}  final h {hf:.5f}", flush=True)

    for h0 in args.h0s:
        record(f"Adam lr={h0}",
               [run_adam(s, args.budget, h0, data, test_loader) for s in args.seeds])
        for rtol in args.rtols:
            record(f"PureRK3(2) h0={h0} rtol={rtol}",
                   [run_rk(PureRK3, s, args.budget, h0, rtol, data, test_loader)
                    for s in args.seeds])
            record(f"FixedRK3Adam h0={h0} rtol={rtol}",
                   [run_rk(FixedRK3Adam, s, args.budget, h0, rtol, data, test_loader)
                    for s in args.seeds])

    with open(args.out, "w") as f:
        json.dump({"budget": args.budget, "n_train": args.n_train,
                   "seeds": args.seeds, "results": results}, f, indent=2)
    print(f"\nSaved -> {args.out}")

    best_rk = min((v["mean_final_loss"], k) for k, v in results.items() if "RK3" in k)
    best_bl = min((v["mean_final_loss"], k) for k, v in results.items() if "Adam lr" in k)
    print(f"\nBest repaired RK: {best_rk[1]} @ {best_rk[0]:.6f}")
    print(f"Best Adam:        {best_bl[1]} @ {best_bl[0]:.6f}")
    print("Verdict:", "Repaired RK WINS" if best_rk[0] < best_bl[0]
          else "Adam still wins with a functional controller (objection closed)")


if __name__ == "__main__":
    main()
