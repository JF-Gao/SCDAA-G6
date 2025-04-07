import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, TensorDataset

# === 1. LQR system parameters ===
T = 0.5  # Time horizon
N = 1000  # Time discretization steps
H = torch.tensor([[0.5, 0.5], [0.0, 0.5]], dtype=torch.float64)  # Drift matrix
M = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)  # Control matrix
sigma = 0.5 * torch.eye(2, dtype=torch.float64)  # Diffusion matrix
C = torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float64)  # State cost
D = 0.1 * torch.tensor([[1.0, 0.1], [0.1, 1.0]], dtype=torch.float64)  # Control cost
R = 10.0 * torch.tensor([[1.0, 0.3], [0.3, 1.0]], dtype=torch.float64)  # Terminal cost
time_grid = torch.linspace(0, T, N + 1)  # Time grid for simulation

# === 2. Solve Riccati equation backward in time ===
def riccati_ode(t, S_flat):
    S = torch.tensor(S_flat.reshape(2, 2), dtype=torch.float64)
    D_inv = torch.linalg.inv(D)
    dSdt = S @ M @ D_inv @ M.T @ S - H.T @ S - S @ H - C
    return dSdt.numpy().flatten()

def solve_riccati():
    S_T = R.numpy().flatten()
    sol = solve_ivp(riccati_ode, [T, 0], S_T, method='BDF',
                    t_eval=np.linspace(T, 0, N + 1),
                    rtol=1e-8, atol=1e-10)
    S_values = sol.y.T.reshape(-1, 2, 2)[::-1]
    return torch.tensor(np.ascontiguousarray(S_values), dtype=torch.float64)

S_list = solve_riccati()

def get_S(t):
    """Get Riccati matrix S(t) at given time t"""
    idx = min(np.searchsorted(time_grid.numpy(), t, side='right') - 1, len(S_list) - 1)
    return S_list[idx]

# === 3. Value function V(t, x) = x^T S(t) x + ∫ Tr[σᵀ S(t) σ] dt ===
def value_function(t, x):
    S_t = get_S(t)
    term1 = x @ S_t @ x
    trace_integral = torch.tensor([torch.trace(sigma @ sigma.T @ S) for S in S_list])
    return term1 + torch.trapz(trace_integral, time_grid)

# === 4. Analytic optimal control: a*(t, x) = -D⁻¹ Mᵀ S(t) x ===
def strict_control(t, x):
    S_t = get_S(t)
    return -torch.linalg.inv(D) @ M.T @ S_t @ x

# === 5. Actor neural network: 2-layer ReLU MLP ===
class ActorPolicy(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, tx):
        return self.net(tx)

# === 6. Supervised actor trainer using sum loss ===
class SupervisedActor:
    def __init__(self, policy_net, optimizer, scheduler):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_history = []

    def supervised_train_actor(self, epochs=2000, batch_size=1024):
        # Sample random (t, x) pairs
        t = torch.rand(batch_size, dtype=torch.float64) * T
        x = torch.randn(batch_size, 2, dtype=torch.float64) * 2.0
        inputs = torch.cat([t.view(-1, 1), x], dim=1)

        # Supervised targets from analytic control
        targets = torch.stack([
            strict_control(ti.item(), xi) for ti, xi in zip(t, x)
        ])

        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        loss_fn = nn.MSELoss(reduction='sum')  # Required: sum loss for stability

        for ep in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                pred = self.policy_net(xb)
                loss = loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.scheduler.step()
            self.loss_history.append(epoch_loss)
            if ep % 100 == 0:
                print(f"Epoch {ep}/{epochs}, Loss = {epoch_loss:.2e}")

    def plot_loss(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.loss_history)
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (log scale)")
        plt.title("Supervised Actor Loss over Epochs (sum loss)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === 7. Compare strict vs actor trajectories from 4 x₀ ===
def plot_strict_vs_actor_trajectories(actor_net):
    x0_list = [
        torch.tensor([2.0, 2.0], dtype=torch.float64),
        torch.tensor([2.0, -2.0], dtype=torch.float64),
        torch.tensor([-2.0, -2.0], dtype=torch.float64),
        torch.tensor([-2.0, 2.0], dtype=torch.float64),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    dt = T / N

    for i, x0 in enumerate(x0_list):
        torch.manual_seed(42)  # Shared Brownian motion
        W = torch.randn((N, 2), dtype=torch.float64) * np.sqrt(dt)

        # Strict LQR trajectory
        x_strict = torch.zeros((N + 1, 2), dtype=torch.float64)
        x_strict[0] = x0.clone()
        for k in range(N):
            t_k = time_grid[k].item()
            a = strict_control(t_k, x_strict[k])
            drift = H @ x_strict[k] + M @ a
            x_strict[k + 1] = x_strict[k] + dt * drift + sigma @ W[k]

        # Actor trajectory
        x_actor = torch.zeros((N + 1, 2), dtype=torch.float64)
        x_actor[0] = x0.clone()
        for k in range(N):
            t_k = time_grid[k].item()
            tx = torch.cat([torch.tensor([[t_k]]), x_actor[k].view(1, -1)], dim=1)
            a = actor_net(tx).view(-1)
            drift = H @ x_actor[k] + M @ a
            x_actor[k + 1] = x_actor[k] + dt * drift + sigma @ W[k]

        # Plot both trajectories
        axs[i].plot(x_strict[:, 0].detach().numpy(), x_strict[:, 1].detach().numpy(), label="Strict LQR", color="blue")
        axs[i].plot(x_actor[:, 0].detach().numpy(), x_actor[:, 1].detach().numpy(), label="Actor", color="orange", linestyle="--")
        axs[i].scatter(*x0.detach().numpy(), color="black")
        axs[i].set_title(f"Trajectory from x0 = {x0.tolist()}")
        axs[i].set_xlabel("x[0]")
        axs[i].set_ylabel("x[1]")
        axs[i].legend()
        axs[i].grid(True)

    plt.suptitle("Strict vs Actor Trajectories")
    plt.tight_layout()
    plt.show()

# === 8. Print value function and controls at t=0 ===
def print_value_and_controls(actor_net):
    x0_list = [
        torch.tensor([2.0, 2.0], dtype=torch.float64),
        torch.tensor([2.0, -2.0], dtype=torch.float64),
        torch.tensor([-2.0, -2.0], dtype=torch.float64),
        torch.tensor([-2.0, 2.0], dtype=torch.float64),
    ]
    print("\n===== Value Function and Controls at t=0 =====")
    for x0 in x0_list:
        t0 = 0.0
        val = value_function(t0, x0)
        strict_a = strict_control(t0, x0)
        actor_tx = torch.cat([torch.tensor([[t0]]), x0.view(1, -1)], dim=1)
        actor_a = actor_net(actor_tx).view(-1)
        print(f"\nx0 = {x0.tolist()}")
        print(f"Value V(0,x) = {val.item():.4f}")
        print(f"Strict control: {strict_a}")
        print(f"Actor  control: {actor_a}")

# === 9. Run everything ===
if __name__ == '__main__':
    actor_net = ActorPolicy()
    actor_net.double()

    optimizer = optim.Adam(actor_net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

    trainer = SupervisedActor(actor_net, optimizer, scheduler)
    trainer.supervised_train_actor(epochs=2000, batch_size=1024)
    trainer.plot_loss()

    plot_strict_vs_actor_trajectories(actor_net)
    print_value_and_controls(actor_net)