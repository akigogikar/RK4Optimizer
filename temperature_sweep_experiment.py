"""Temperature-sweep experiment: tests the *tempering reinterpretation* of the
repaired-controller finding (Section 6).

BACKGROUND / CLAIM UNDER TEST
-----------------------------
Section 6 found that the repaired RK controller drives training loss ~34x below
tuned Adam (to ~1e-6) but *does not* improve generalization -- indeed test
accuracy DEGRADES as the minimization deepens (89.0% -> 86.9% across the
fragility sweep), while a moderate minimizer (Adam lr=0.01, train loss 6.8e-5)
attains the study's best test accuracy (90.30%).

The SG-MCMC reading: an optimizer that minimizes the training loss is sampling
the Gibbs distribution p(theta) proportional to exp(-L(theta)/T) at temperature
T->0. Deep minimization == very cold sampling. The *cold-posterior effect*
(Wenzel et al., ICML 2020) says generalization is NON-MONOTONE in T with an
interior optimum T*>0. If our finding is a tempering effect, then the repaired
controller (T->0) has overshot T* onto the cold/overfitting side, and a proper
temperature knob should recover the lost generalization.

We ground the controller's implicit tempering as an EXPLICIT temperature via
preconditioned Stochastic Gradient Langevin Dynamics (pSGLD; Li, Chen, Carlson &
Carin, AAAI 2016), the SG-MCMC kernel closest to our Adam-preconditioned update.
Langevin (Euler-Maruyama) targeting exp(-L/T):
    theta <- theta - lr * M * g + sqrt(2 * lr * T * M) * eta,   eta ~ N(0, I)
with M = diag(1/(sqrt(v_hat)+eps)) the RMSprop preconditioner (the divergence
correction Gamma is dropped, per standard practice; Li et al. show it is
negligible). T=0 recovers a deterministic RMSprop minimizer (our regime); T>0
caps the minimization depth at a noise floor. SGLD (M=I) is included as a
preconditioner-free robustness arm.

PRE-REGISTERED PREDICTIONS (written before the run; do not edit after seeing
results -- report whatever occurs):
  H1 (tempering supported):
     (a) final train loss increases monotonically with T (noise floor), AND
     (b) final TEST ACCURACY is NON-MONOTONE in T: rises from T=0 to an interior
         T*>0 then falls -- i.e. T=0 (pure minimization) is NOT the best, AND
     (c) the within-run test-accuracy trajectory at T=0 peaks then DECLINES
         (classic overfitting), corroborating (b).
  H0 (tempering REFUTED -- must be reported as a negative result if it occurs):
     test accuracy is monotonically non-increasing in T (T=0 is best or tied)
     AND no within-run overfitting peak at T=0 => pure minimization does not
     overfit in this regime => the Section-6 degradation is NOT a tempering
     effect, and the reinterpretation is not supported.

Protocol matches fullbatch_experiment.py / fixed_controller_experiment.py:
1024-example MNIST subset, 784-128-10 MLP, one full-batch backward == one grad
eval, mean +/- std over seeds.

Usage:
  python3 temperature_sweep_experiment.py --smoke          # quick calibration
  python3 temperature_sweep_experiment.py --budget 2000 --seeds 0 1 2 3 4
"""
import argparse, json, math
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


def run_langevin(seed, budget, lr, T, data, test_loader, precondition=True,
                 beta2=0.999, eps=1e-8, probe_every=100):
    """pSGLD (precondition=True) or SGLD (False) at temperature T.

    T=0 is exactly the deterministic minimizer (no noise injected).
    Returns (final_train_loss, final_test_acc, evals, traj) where traj is a list
    of (eval, train_loss, test_acc) probes for the within-run overfitting check.
    """
    (x, y) = data
    model = make_model(seed)
    params = list(model.parameters())
    v = [torch.zeros_like(p) for p in params]
    gen = torch.Generator().manual_seed(1000 * seed + 7)
    evals = 0
    traj = []
    while evals < budget:
        loss = full_loss(model, x, y)
        grads = torch.autograd.grad(loss, params)
        evals += 1
        with torch.no_grad():
            for i, p in enumerate(params):
                g = grads[i]
                if precondition:
                    v[i] = beta2 * v[i] + (1 - beta2) * g * g
                    v_hat = v[i] / (1 - beta2 ** evals)
                    M = 1.0 / (torch.sqrt(v_hat) + eps)
                else:
                    M = torch.ones_like(g)
                step = lr * M * g
                if T > 0:
                    noise = torch.randn(g.shape, generator=gen) * torch.sqrt(
                        torch.clamp(2.0 * lr * T * M, min=0.0))
                    p.copy_(p - step + noise)
                else:
                    p.copy_(p - step)
        if evals % probe_every == 0 or evals == budget:
            with torch.no_grad():
                tl = full_loss(model, x, y).item()
            traj.append((evals, tl, evaluate(model, test_loader)))
    with torch.no_grad():
        final = full_loss(model, x, y).item()
    return final, evaluate(model, test_loader), evals, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-2,
                    help="matches the best-Adam operating point (Table 4)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--temps", type=float, nargs="+", default=None,
                    help="temperatures to sweep; default is a log grid incl. 0")
    ap.add_argument("--sgld", action="store_true",
                    help="also run the preconditioner-free SGLD robustness arm")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed, short budget, coarse grid for calibration")
    ap.add_argument("--out", default="temperature_sweep_results.json")
    args = ap.parse_args()

    if args.smoke:
        args.budget = 800
        args.seeds = [0]
        temps = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    else:
        temps = args.temps if args.temps is not None else [
            0.0, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    data, test_loader = get_data(args.n_train)
    results = {}

    def sweep(precondition, tag):
        for T in temps:
            losses, accs, trajs = [], [], []
            for seed in args.seeds:
                fl, acc, ev, traj = run_langevin(
                    seed, args.budget, args.lr, T, data, test_loader,
                    precondition=precondition)
                losses.append(fl); accs.append(acc); trajs.append(traj)
            ml = sum(losses) / len(losses)
            sl = (sum((l - ml) ** 2 for l in losses) / max(len(losses) - 1, 1)) ** 0.5
            ma = sum(accs) / len(accs)
            sa = (sum((a - ma) ** 2 for a in accs) / max(len(accs) - 1, 1)) ** 0.5
            name = f"{tag} T={T:g}"
            results[name] = {
                "optimizer": tag, "T": T, "lr": args.lr,
                "mean_final_loss": ml, "std_final_loss": sl, "losses": losses,
                "mean_test_acc": ma, "std_test_acc": sa, "accs": accs,
                "evals": ev, "trajectories": trajs}
            print(f"{name:22s} loss {ml:.3e} +/- {sl:.1e}   "
                  f"test {ma*100:6.3f}% +/- {sa*100:.2f}   (n={len(args.seeds)})",
                  flush=True)

    print(f"=== pSGLD temperature sweep (lr={args.lr}, budget={args.budget}, "
          f"seeds={args.seeds}) ===", flush=True)
    sweep(True, "pSGLD")
    if args.sgld:
        print("=== SGLD (preconditioner-free) robustness arm ===", flush=True)
        sweep(False, "SGLD")

    with open(args.out, "w") as f:
        json.dump({"budget": args.budget, "n_train": args.n_train,
                   "lr": args.lr, "seeds": args.seeds, "temps": temps,
                   "results": results}, f, indent=2)
    print(f"\nSaved -> {args.out}", flush=True)

    # Automated pre-registered verdict (pSGLD arm).
    ps = [(v["T"], v["mean_test_acc"], v["mean_final_loss"])
          for k, v in results.items() if v["optimizer"] == "pSGLD"]
    ps.sort()
    T0_acc = ps[0][1]
    best = max(ps, key=lambda r: r[1])
    loss_monotone = all(ps[i][2] <= ps[i + 1][2] * 1.5 for i in range(len(ps) - 1))
    print("\n--- Pre-registered verdict (pSGLD) ---")
    print(f"T=0 test acc:        {T0_acc*100:.3f}%  (loss {ps[0][2]:.2e})")
    print(f"best test acc:       {best[1]*100:.3f}%  at T={best[0]:g} "
          f"(loss {best[2]:.2e})")
    print(f"interior optimum:    {'YES' if best[0] > 0 else 'NO'} "
          f"(best T {'>' if best[0] > 0 else '=='} 0)")
    gain = (best[1] - T0_acc) * 100
    print(f"gen. gain from temp: {gain:+.3f} pts")
    if best[0] > 0 and gain > 0.3:
        print("VERDICT: H1 supported -- interior temperature optimum; "
              "pure minimization (T=0) overshoots (tempering reinterpretation "
              "supported).")
    else:
        print("VERDICT: H0 -- T=0 is best/tied; no interior optimum "
              "(tempering reinterpretation NOT supported; report as negative).")


if __name__ == "__main__":
    main()
