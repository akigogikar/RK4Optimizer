import json, torch, torch.nn.functional as F, torch.nn as nn
from fixed_controller_experiment import (make_model, evaluate, get_data,
                                         FixedRK3Adam, run_adam)

def run_fixed_step(seed, budget, h, data, test_loader):
    x, y = data
    model = make_model(seed)
    opt = FixedRK3Adam(model, h0=h, rtol=1e9, h_min=h, h_max=h)  # pinned step
    loss_fn = nn.CrossEntropyLoss()
    t = 0
    while opt.grad_evals < budget - 4:
        t += 1
        opt.step(loss_fn, x, y, t)
    with torch.no_grad():
        final = F.cross_entropy(model(x), y).item()
    return final, evaluate(model, test_loader), opt.grad_evals, opt.rejects, opt.h

def main():
    budget, seeds = 600, [0, 1, 2]
    data, test_loader = get_data(1024)
    results = []
    for name, fn in [
        ("FixedStep-RK3Adam h=0.1", lambda s: run_fixed_step(s, budget, 0.1, data, test_loader)),
        ("FixedStep-RK3Adam h=0.03", lambda s: run_fixed_step(s, budget, 0.03, data, test_loader)),
        ("Adam lr=0.1", lambda s: run_adam(s, budget, 0.1, data, test_loader)),
        ("Adam lr=0.03", lambda s: run_adam(s, budget, 0.03, data, test_loader)),
        ("Adam lr=0.01", lambda s: run_adam(s, budget, 0.01, data, test_loader)),
    ]:
        losses, accs, rejects = [], [], []
        for s in seeds:
            loss, acc, evals, rej, h = fn(s)
            losses.append(loss); accs.append(acc); rejects.append(rej)
        import statistics as st
        m = st.mean(losses); sd = st.stdev(losses) if len(losses) > 1 else 0.0
        print(f"{name:35s} loss {m:.6f} +/- {sd:.6f}  acc {st.mean(accs)*100:.2f}%  rejects/seed {st.mean(rejects):5.1f}")
        results.append({"name": name, "loss_mean": m, "loss_std": sd,
                        "acc_mean": st.mean(accs), "rejects": st.mean(rejects)})
    json.dump(results, open("hmax_control_results.json", "w"), indent=2)
    print("Saved -> hmax_control_results.json")

if __name__ == "__main__":
    main()
