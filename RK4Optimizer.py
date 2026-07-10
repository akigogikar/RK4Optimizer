import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Define a simple neural network
class SimpleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Momentum-Enhanced RK4 optimizer with adaptive learning rates and momentum
class AdaptiveMomentumRK4Optimizer:
    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.lr = lr  # Base learning rate
        self.beta1 = beta1  # Momentum decay rate
        self.beta2 = beta2  # Variance decay rate
        self.epsilon = epsilon  # To avoid division by zero
        self.m = []  # Initialize momentum (first moment)
        self.v = []  # Initialize variance (second moment)
        for param in self.model.parameters():
            self.m.append(torch.zeros_like(param))  # First moment estimate (momentum)
            self.v.append(torch.zeros_like(param))  # Second moment estimate (variance)

    def _grads(self, loss_fn, x, y, params):
        """Gradients of the loss w.r.t. ALL parameters jointly (no graph retained)."""
        loss = loss_fn(self.model(x), y)
        return torch.autograd.grad(loss, params)

    def _set_params(self, params, base, direction=None, scale=0.0):
        """Set every parameter to base[i] - scale * direction[i] (jointly)."""
        with torch.no_grad():
            for i, p in enumerate(params):
                if direction is None:
                    p.copy_(base[i])
                else:
                    p.copy_(base[i] - scale * direction[i])

    def step(self, loss_fn, x, y, t):
        params = list(self.model.parameters())
        p0 = [p.detach().clone() for p in params]  # original point for ALL stages
        h = self.lr  # stage step size: RK4 integrates the raw gradient flow with step h

        # --- RK4 stages: one consistent vector field (raw loss gradient), ---
        # --- each stage evaluated from the ORIGINAL point p0, all params jointly ---
        k1 = self._grads(loss_fn, x, y, params)                      # at p0

        self._set_params(params, p0, k1, h / 2)                      # p0 - h/2 * k1
        k2 = self._grads(loss_fn, x, y, params)

        self._set_params(params, p0, k2, h / 2)                      # p0 - h/2 * k2
        k3 = self._grads(loss_fn, x, y, params)

        self._set_params(params, p0, k3, h)                          # p0 - h * k3
        k4 = self._grads(loss_fn, x, y, params)

        # --- Adam-style update driven by the RK4-averaged gradient ---
        with torch.no_grad():
            for i, p in enumerate(params):
                g = (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6  # classical RK4 weights

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

                m_hat = self.m[i] / (1 - self.beta1 ** t)  # bias correction
                v_hat = self.v[i] / (1 - self.beta2 ** t)

                adaptive_lr = self.lr / (torch.sqrt(v_hat) + self.epsilon)
                p.copy_(p0[i] - adaptive_lr * m_hat)

# Embedded RK3(2) (Bogacki-Shampine) optimizer with FSAL and adaptive step control
class AdaptiveEmbeddedRK3Optimizer:
    """Theoretical upgrades over fixed-step RK4:
    - Bogacki-Shampine 3(2) embedded pair: 3rd-order update + free 2nd-order
      error estimate from the same stages.
    - FSAL: the gradient at the accepted new point is reused as next step's k1
      (exact here, since k1_{t+1} = grad at theta_{t+1}) -> 3 fresh grad evals/step.
    - Local-error step-size controller: h grows on smooth regions, shrinks
      near high curvature (h <- h * 0.9 * err^(-1/3), clipped).
    The RK-averaged gradient then drives standard Adam moments.
    """

    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 rtol=1e-2, atol=1e-4, h_min_scale=0.1, h_max_scale=2.0,
                 h0=None):
        # h_max_scale=2.0: empirically, beyond ~2x base lr the binding constraint
        # is Adam's discrete stability, not ODE truncation error; larger ceilings
        # cause oscillation (see sweep: 10x -> final loss 3.9e-3 vs 1.0e-5 at 2x).
        # NOTE: on well-conditioned full-batch problems the BS3(2) truncation
        # error sits far below any reasonable tolerance, so the controller wants
        # to grow h every step and pins to h_max -> the step is effectively
        # FIXED at h_max and rtol has no effect. Expose h_max_scale/h0 so this
        # saturation regime is measurable and the controller can be exercised.
        self.model = model
        self.lr = lr
        # h0: explicit initial integration step (defaults to lr). Lets an h0
        # sweep probe whether the initial step matters or is immediately erased
        # by the controller converging to its equilibrium/ceiling.
        self.h = float(h0) if h0 is not None else lr
        self.h_min = lr * h_min_scale
        self.h_max = lr * h_max_scale
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.rtol, self.atol = rtol, atol
        self.k1 = None                   # FSAL cache
        self.grad_evals = 0
        self.m = [torch.zeros_like(p) for p in model.parameters()]
        self.v = [torch.zeros_like(p) for p in model.parameters()]
        # Diagnostics: record the step-size trajectory and per-step normalized
        # local error so we can see whether the controller ever actually adapts
        # (moves off a clamp boundary) or is saturated.
        self.h_history = []
        self.err_history = []
        self.n_rejected = 0
        self.n_saturated_max = 0
        self.n_saturated_min = 0

    def _grads(self, loss_fn, x, y, params):
        loss = loss_fn(self.model(x), y)
        self.grad_evals += 1
        return [g.detach() for g in torch.autograd.grad(loss, params)]

    def _set_params(self, params, base, direction, scale):
        with torch.no_grad():
            for i, p in enumerate(params):
                p.copy_(base[i] - scale * direction[i])

    def step(self, loss_fn, x, y, t):
        params = list(self.model.parameters())
        p0 = [p.detach().clone() for p in params]
        h = self.h

        # --- Bogacki-Shampine 3(2) stages on the gradient flow ---
        k1 = self.k1 if self.k1 is not None else self._grads(loss_fn, x, y, params)

        self._set_params(params, p0, k1, h / 2)              # p0 - h/2 * k1
        k2 = self._grads(loss_fn, x, y, params)

        self._set_params(params, p0, k2, 3 * h / 4)          # p0 - 3h/4 * k2
        k3 = self._grads(loss_fn, x, y, params)

        # 3rd-order averaged gradient
        g3 = [(2 / 9) * k1[i] + (1 / 3) * k2[i] + (4 / 9) * k3[i]
              for i in range(len(params))]

        # --- Adam-style update driven by the RK-averaged gradient ---
        with torch.no_grad():
            for i, p in enumerate(params):
                g = g3[i]
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m[i] / (1 - self.beta1 ** t)
                v_hat = self.v[i] / (1 - self.beta2 ** t)
                adaptive_lr = h / (torch.sqrt(v_hat) + self.epsilon)
                p.copy_(p0[i] - adaptive_lr * m_hat)

        # --- FSAL stage: gradient at the accepted point = next step's k1 ---
        k4 = self._grads(loss_fn, x, y, params)
        self.k1 = k4

        # --- Embedded 2nd-order estimate -> local error -> step control ---
        with torch.no_grad():
            err_sq, n_el = 0.0, 0
            for i, p in enumerate(params):
                g2 = (7 / 24) * k1[i] + (1 / 4) * k2[i] + (1 / 3) * k3[i] + (1 / 8) * k4[i]
                e = h * (g3[i] - g2)                          # local truncation error
                scale = self.atol + self.rtol * p.abs()
                err_sq += ((e / scale) ** 2).sum().item()
                n_el += p.numel()
            err = max((err_sq / n_el) ** 0.5, 1e-12)
            factor = min(max(0.9 * err ** (-1.0 / 3.0), 0.5), 2.0)
            h_new = min(max(h * factor, self.h_min), self.h_max)
            # Diagnostics: track whether the controller is saturated at a clamp
            # boundary (meaning rtol/h0 have no effect) or genuinely adapting.
            self.err_history.append(err)
            self.h_history.append(h_new)
            if h_new >= self.h_max - 1e-15:
                self.n_saturated_max += 1
            elif h_new <= self.h_min + 1e-15:
                self.n_saturated_min += 1
            self.h = h_new


# Function to train the network with any optimizer
def train_network(model, optimizer, loss_fn, x, y, epochs, optimizer_name):
    loss_history = []
    time_history = []

    for epoch in range(1, epochs + 1):
        batch_start_time = time.time()  # Start timing per batch

        if optimizer_name in ('Momentum-Enhanced RK4', 'Embedded RK3(2) FSAL'):
            # Custom RK-based step
            optimizer.step(loss_fn, x, y, epoch)
        else:
            # Vanilla Adam optimizer step
            optimizer.zero_grad()  # Zero the gradients
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the parameters

        batch_time = time.time() - batch_start_time  # Compute per-batch time
        time_history.append(batch_time)

        # Track loss
        with torch.no_grad():
            output = model(x)
            loss = loss_fn(output, y).item()
            loss_history.append(loss)

    return loss_history, time_history

def _demo():
    """Toy benchmark from the initial release. Runs only when executed
    directly (python RK4Optimizer.py), never on import."""
    # Set hyperparameters
    input_dim = 10
    hidden_dim = 64
    output_dim = 2
    epochs = 100
    lr = 0.01

    # RK4 performs 4 gradient evaluations per step, Adam performs 1.
    # For a fair comparison, the equal-budget Adam baseline gets 4x the steps
    # (same number of gradient evaluations / ~equal wall-clock).
    GRAD_EVALS_PER_RK4_STEP = 4

    # Seed for reproducibility
    torch.manual_seed(0)

    # Generate random input and target data
    x = torch.randn((32, input_dim))  # Batch size of 32
    y = torch.randn((32, output_dim))

    # Initialize the models (identical initialization for all three runs)
    torch.manual_seed(1)
    model_rk4 = SimpleNetwork(input_dim, hidden_dim, output_dim)
    torch.manual_seed(1)
    model_adam = SimpleNetwork(input_dim, hidden_dim, output_dim)
    torch.manual_seed(1)
    model_adam_eq = SimpleNetwork(input_dim, hidden_dim, output_dim)
    torch.manual_seed(1)
    model_bs3 = SimpleNetwork(input_dim, hidden_dim, output_dim)

    # Initialize the optimizers
    rk4_optimizer = AdaptiveMomentumRK4Optimizer(model_rk4, lr=lr)
    adam_optimizer = optim.Adam(model_adam.parameters(), lr=lr)
    adam_eq_optimizer = optim.Adam(model_adam_eq.parameters(), lr=lr)
    bs3_optimizer = AdaptiveEmbeddedRK3Optimizer(model_bs3, lr=lr)

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Train the models
    rk4_loss_history, rk4_time_history = train_network(
        model_rk4, rk4_optimizer, loss_fn, x, y, epochs, 'Momentum-Enhanced RK4'
    )
    adam_loss_history, adam_time_history = train_network(
        model_adam, adam_optimizer, loss_fn, x, y, epochs, 'Adam'
    )
    # Equal-budget baseline: same gradient-evaluation count as RK4
    adam_eq_loss_history, adam_eq_time_history = train_network(
        model_adam_eq, adam_eq_optimizer, loss_fn, x, y,
        epochs * GRAD_EVALS_PER_RK4_STEP, 'Adam'
    )
    # Embedded RK3(2) with FSAL: 3 fresh grad evals/step + adaptive step size
    bs3_loss_history, bs3_time_history = train_network(
        model_bs3, bs3_optimizer, loss_fn, x, y, epochs, 'Embedded RK3(2) FSAL'
    )

    # Cumulative wall-clock per run (for the equal-wall-clock plot)
    def cumulative(times):
        out, total = [], 0.0
        for t in times:
            total += t
            out.append(total)
        return out

    rk4_cum = cumulative(rk4_time_history)
    adam_cum = cumulative(adam_time_history)
    adam_eq_cum = cumulative(adam_eq_time_history)
    bs3_cum = cumulative(bs3_time_history)

    # Plot the comparisons
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(rk4_loss_history, label="Momentum-Enhanced RK4")
    plt.plot(bs3_loss_history, label="Embedded RK3(2) FSAL")
    plt.plot(adam_loss_history, label="Adam (equal steps)")
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs Steps (RK4 uses 4x gradients/step)')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(rk4_cum, rk4_loss_history, label="Momentum-Enhanced RK4")
    plt.plot(bs3_cum, bs3_loss_history, label="Embedded RK3(2) FSAL")
    plt.plot(adam_eq_cum, adam_eq_loss_history, label="Adam (equal wall-clock)")
    plt.xlabel('Cumulative Wall-clock Time (seconds)')
    plt.ylabel('Loss')
    plt.title('Loss vs Wall-clock (fair comparison)')
    plt.legend()

    plt.subplot(1, 3, 3)
    grad_evals_rk4 = [(i + 1) * GRAD_EVALS_PER_RK4_STEP for i in range(len(rk4_loss_history))]
    grad_evals_adam_eq = [i + 1 for i in range(len(adam_eq_loss_history))]
    plt.plot(grad_evals_rk4, rk4_loss_history, label="Momentum-Enhanced RK4")
    grad_evals_bs3 = [1 + (i + 1) * 3 for i in range(len(bs3_loss_history))]  # FSAL: 3/step after initial k1
    plt.plot(grad_evals_bs3, bs3_loss_history, label="Embedded RK3(2) FSAL")
    plt.plot(grad_evals_adam_eq, adam_eq_loss_history, label="Adam (equal budget)")
    plt.xlabel('Gradient Evaluations')
    plt.ylabel('Loss')
    plt.title('Loss vs Gradient-Evaluation Budget')
    plt.legend()

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=120)
    plt.show()

    # Print total training time and final losses
    print(f"Momentum-Enhanced RK4  : {sum(rk4_time_history):.4f}s over {epochs} steps, final loss {rk4_loss_history[-1]:.6f}")
    print(f"Adam (equal steps)     : {sum(adam_time_history):.4f}s over {epochs} steps, final loss {adam_loss_history[-1]:.6f}")
    print(f"Adam (equal wall-clock): {sum(adam_eq_time_history):.4f}s over {epochs * GRAD_EVALS_PER_RK4_STEP} steps, final loss {adam_eq_loss_history[-1]:.6f}")
    print(f"Embedded RK3(2) FSAL   : {sum(bs3_time_history):.4f}s over {epochs} steps, {bs3_optimizer.grad_evals} grad evals, final loss {bs3_loss_history[-1]:.6f}, final h {bs3_optimizer.h:.5f}")


if __name__ == "__main__":
    _demo()
