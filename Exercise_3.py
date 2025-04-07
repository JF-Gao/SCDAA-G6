# ================================
# Exercise 3.1: Critic-only Learning (Soft LQR)
# Learn the value function v(t, x) using supervised regression from a fixed soft policy
# ================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

# === System Parameters (same as Part 2, with D = Identity matrix as required in Exercise 3) ===
H = torch.tensor([[0.5, 0.5], [0.0, 0.5]])           # Drift matrix for state
M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])           # Control influence matrix
sigma = 0.5 * torch.eye(2)                           # Diffusion coefficient (noise strength)
C = torch.tensor([[1.0, 0.1], [0.1, 1.0]])           # Running cost for state
D = torch.eye(2)                                     # Running cost for control (set to Identity)
R = 10.0 * torch.tensor([[1.0, 0.3], [0.3, 1.0]])    # Terminal cost

# === Time Discretization Parameters ===
T = 0.5                    # Terminal time
N = 100                    # Number of time steps
dt = T / N                 # Time step size
tau = 0.5                  # Entropy regularization parameter
gamma = 1.0                # Control noise variance
time_grid = torch.linspace(0, T, N + 1)  # Time discretization grid

# === Solve the Riccati ODE for exact value function and soft policy ===
def riccati_rhs(t, S_flat):
    S = torch.tensor(S_flat.reshape(2, 2))
    dSdt = S @ M @ torch.linalg.inv(D) @ M.T @ S - H.T @ S - S @ H - C
    return dSdt.numpy().flatten()

# Solve backward in time (T → 0) with final condition S(T) = R
sol = solve_ivp(riccati_rhs, [T, 0], R.numpy().flatten(),
                t_eval=np.linspace(T, 0, N + 1),
                method='BDF', rtol=1e-8, atol=1e-10)

# Store Riccati solution as torch tensor (shape: [N+1, 2, 2])
S_list = torch.tensor(sol.y.T.reshape(-1, 2, 2)[::-1].copy())

# Interpolate S(t) from precomputed list
def get_S(t):
    idx = min(np.searchsorted(time_grid.numpy(), t, side='right') - 1, N)
    return S_list[idx]

# === Fixed policy π(t,x): optimal soft LQR policy ===
def mean_control(t, x):
    return -torch.linalg.inv(D) @ M.T @ get_S(t) @ x

def sample_control(t, x):
    # Sample action from Gaussian π(·|t,x) = N(mean, τγ I)
    mu = mean_control(t, x)
    return mu + torch.randn_like(mu) * np.sqrt(tau * gamma)

# === Exact value function (for evaluation only) ===
def exact_value_function(t, x):
    S_t = get_S(t)
    trace_vals = torch.tensor([
        torch.trace(sigma @ sigma.T @ S) for S in S_list[min(np.searchsorted(time_grid, t), N):]
    ])
    integral = torch.trapz(trace_vals, time_grid[min(np.searchsorted(time_grid, t), N):])
    return x @ S_t @ x + integral

# === Critic Neural Network v(t,x): 3-layer ReLU MLP with width 512 ===
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, t, x):
        tx = torch.cat([t.unsqueeze(-1), x], dim=-1)  # Concatenate time and state
        return self.net(tx).squeeze(-1)               # Output scalar value v(t,x)

# === Sample a trajectory from the fixed policy π(t,x) ===
def generate_trajectory():
    x = torch.zeros(N + 1, 2)
    x[0] = torch.randn(2)  # Initial state sampled from ρ
    a_list, f_list = [], []
    W = torch.randn(N, 2) * np.sqrt(dt)  # Brownian noise

    for n in range(N):
        a = sample_control(time_grid[n].item(), x[n])
        x[n + 1] = x[n] + dt * (H @ x[n] + M @ a) + sigma @ W[n]
        a_list.append(a)
        f_list.append(x[n] @ C @ x[n] + a @ D @ a)

    return time_grid[:-1], x[:-1], torch.stack(a_list), torch.tensor(f_list), x[N] @ R @ x[N]

# === Train critic using supervised regression from fixed policy (Algorithm 2) ===
model = Critic()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

steps = 2000
batch_size = 32
loss_log = []
best_loss = float('inf')
best_model = None

for step in range(steps):
    loss = 0.0
    for _ in range(batch_size):
        t_seq, x_seq, a_seq, f_seq, gT = generate_trajectory()
        for n in range(N):
            v_pred = model(t_seq[n:n+1], x_seq[n:n+1])   # v(t_n, x_n)
            logp = torch.sum(a_seq[n:]**2, dim=1)        # Approximation of log π
            target = torch.sum(f_seq[n:] + tau * logp) * dt + gT
            loss += N * (v_pred - target) ** 2           # Sum-based loss scaled by N

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_log.append(loss.item())

    # Save best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model = model.state_dict()

    if step % 100 == 0:
        print(f"Step {step}/{steps}, Loss = {loss.item():.4e}")

# === Plot smoothed training loss (log scale) ===
def smooth(y, alpha=0.95):
    avg, s = [], y[0]
    for v in y:
        s = alpha * s + (1 - alpha) * v
        avg.append(s)
    return avg

plt.figure()
plt.plot(smooth(loss_log, alpha=0.95))
plt.yscale("log")
plt.title("Critic Training Loss (log scale, smoothed)")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Save images to the left folder for review
plt.savefig("plot_1-Exercise 3.png")

# === Evaluate maximum value function error over grid (for reporting) ===
def evaluate_max_error(model):
    t_test = torch.tensor([0.0, 1/6, 1/3, 0.5])  # 4 evaluation time points
    x_lin = torch.linspace(-3, 3, 10)            # 10x10 spatial grid
    max_err = 0.0
    for t in t_test:
        t_batch = t.repeat(x_lin.numel()**2)
        x_mesh = torch.cartesian_prod(x_lin, x_lin)
        v_pred = model(t_batch, x_mesh)
        v_true = torch.tensor([exact_value_function(t, x) for x in x_mesh])
        err = torch.abs(v_pred - v_true)
        max_err = max(max_err, err.max().item())
    return max_err

# Load best critic model and evaluate final max error
model.load_state_dict(best_model)
max_error = evaluate_max_error(model)
print(f"\n✅ Max value function error on grid: {max_error:.4f}")