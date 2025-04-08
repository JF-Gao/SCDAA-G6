import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)

# === Parameters from Figure 1 ===
H = torch.tensor([[0.5, 0.5], [0.0, 0.5]])
M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
sigma = 0.5 * torch.eye(2)
C = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
D = 0.1 * torch.tensor([[1.0, 0.1], [0.1, 1.0]])
R = 10.0 * torch.tensor([[1.0, 0.3], [0.3, 1.0]])

T = 0.5
N = 1000
dt = T / N
tau = 0.5
gamma = 1.0
time_grid = torch.linspace(0, T, N + 1)

# === Riccati ===
def riccati_rhs(t, S_flat):
    S = torch.tensor(S_flat.reshape(2, 2))
    dSdt = S @ M @ torch.linalg.inv(D) @ M.T @ S - H.T @ S - S @ H - C
    return dSdt.numpy().flatten()

sol = solve_ivp(riccati_rhs, [T, 0], R.numpy().flatten(),
                t_eval=np.linspace(T, 0, N + 1),
                method='BDF', rtol=1e-8, atol=1e-10)
S_list = torch.tensor(sol.y.T.reshape(-1, 2, 2)[::-1].copy())

def get_S(t):
    idx = min(np.searchsorted(time_grid.numpy(), t, side='right') - 1, N)
    return S_list[idx]

def mean_control(t, x):
    return -torch.linalg.inv(D) @ M.T @ get_S(t) @ x

def exact_value_function(t, x):
    S_t = get_S(t)
    tr_vals = torch.tensor([torch.trace(sigma @ sigma.T @ S) for S in S_list[min(np.searchsorted(time_grid, t), N):]])
    return x @ S_t @ x + torch.trapz(tr_vals, time_grid[min(np.searchsorted(time_grid, t), N):])

# === Networks ===
class ActorPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, t, x):
        tx = torch.cat([t.view(-1, 1), x], dim=1)
        return self.net(tx)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, t, x):
        tx = torch.cat([t.view(-1, 1), x], dim=1)
        return self.net(tx).squeeze(-1)

# === Trajectory sampling ===
def generate_trajectory(actor_net):
    x = torch.randn(2)
    t_list, x_list, a_list, logp_list, cost_list = [], [], [], [], []
    W = torch.randn(N, 2) * np.sqrt(dt)
    for n in range(N):
        t = time_grid[n].view(1)
        x_input = x.view(1, -1)
        with torch.no_grad():
            a = actor_net(t, x_input).view(-1)
            mean = mean_control(t.item(), x)
            logp = -0.5 * torch.sum((a - mean) ** 2) / (tau * gamma)
            logp = torch.clamp(logp, min=-50.0, max=0.0)
        cost = x @ C @ x + a @ D @ a
        drift = H @ x + M @ a
        x = x + dt * drift + sigma @ W[n]
        t_list.append(t)
        x_list.append(x.clone())
        a_list.append(a)
        logp_list.append(logp)
        cost_list.append(cost)
    terminal_cost = x @ R @ x
    return t_list, x_list, a_list, logp_list, cost_list, terminal_cost

# === Training loop ===
critic = Critic()
actor = ActorPolicy()
opt_critic = optim.Adam(critic.parameters(), lr=1e-3)
opt_actor = optim.Adam(actor.parameters(), lr=1e-3)

loss_log = []
batch_size = 32

for step in range(2000):
    total_critic_loss = 0.0
    total_actor_loss = 0.0

    for _ in range(batch_size):
        t_list, x_list, a_list, logp_list, cost_list, terminal_cost = generate_trajectory(actor)
        t_tensor = torch.cat(t_list)
        x_tensor = torch.stack(x_list)
        cost_tensor = torch.stack(cost_list)

        returns = []
        cum = terminal_cost
        for i in reversed(range(N)):
            cum = cum + dt * (cost_tensor[i] + tau * logp_list[i])
            returns.insert(0, cum)
        returns = torch.stack(returns).detach()

        v_pred = critic(t_tensor, x_tensor)
        critic_loss = torch.sum((v_pred - returns) ** 2) / batch_size
        total_critic_loss += critic_loss

        logp_recomputed = []
        for t_i, x_i, a_true in zip(t_list, x_list, a_list):
            t_in = t_i.view(1)
            x_in = x_i.view(1, -1)
            a_pred = actor(t_in, x_in).view(-1)
            logp = -0.5 * torch.sum((a_pred - a_true.detach()) ** 2) / (tau * gamma)
            logp = torch.clamp(logp, min=-50.0, max=0.0)
            logp_recomputed.append(logp)
        logp_tensor = torch.stack(logp_recomputed)
        advantage = (returns - v_pred.detach())
        advantage = torch.clamp(advantage, -1000.0, 1000.0)
        actor_loss = torch.sum(logp_tensor * advantage) / batch_size
        total_actor_loss += actor_loss

    opt_critic.zero_grad()
    total_critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    opt_critic.step()

    opt_actor.zero_grad()
    total_actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    opt_actor.step()

    loss_log.append(total_critic_loss.item())
    if step % 200 == 0:
        print(f"Step {step}, Critic Loss = {total_critic_loss.item():.4e}")

# === Loss plot ===
def smooth(y, alpha=0.99):
    out = []
    s = y[0]
    for v in y:
        s = alpha * s + (1 - alpha) * v
        out.append(s)
    return out

plt.figure()
plt.plot(smooth(loss_log), label="Critic Loss")
plt.yscale("log")
plt.xlabel("Step")
plt.ylabel("Loss (log scale)")
plt.title("Actor-Critic Critic Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Trajectory plot ===
def plot_strict_vs_actor_trajectories(actor_net):
    x0_list = [
        torch.tensor([2.0, 2.0]),
        torch.tensor([2.0, -2.0]),
        torch.tensor([-2.0, -2.0]),
        torch.tensor([-2.0, 2.0]),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    for i, x0 in enumerate(x0_list):
        torch.manual_seed(42)
        W = torch.randn(N, 2) * np.sqrt(dt)
        x_strict = torch.zeros(N + 1, 2)
        x_actor = torch.zeros(N + 1, 2)
        x_strict[0] = x0.clone()
        x_actor[0] = x0.clone()
        for k in range(N):
            a_s = mean_control(time_grid[k].item(), x_strict[k])
            drift_s = H @ x_strict[k] + M @ a_s
            x_strict[k + 1] = x_strict[k] + dt * drift_s + sigma @ W[k]
            t_k = time_grid[k].view(1)
            x_k = x_actor[k].view(1, -1)
            a_a = actor_net(t_k, x_k).view(-1)
            drift_a = H @ x_actor[k] + M @ a_a
            x_actor[k + 1] = x_actor[k] + dt * drift_a + sigma @ W[k]
        axs[i].plot(x_strict[:, 0].detach().numpy(), x_strict[:, 1].detach().numpy(),
                    label="Strict LQR", color="blue")
        axs[i].plot(x_actor[:, 0].detach().numpy(), x_actor[:, 1].detach().numpy(),
                    label="Actor", linestyle="--", color="orange")
        axs[i].scatter(x0[0].item(), x0[1].item(), color="black", s=40, label="Start" if i == 0 else "")
        axs[i].set_title(f"Trajectory from x0 = {x0.tolist()}")
        axs[i].set_xlabel("x[0]")
        axs[i].set_ylabel("x[1]")
        axs[i].legend()
        axs[i].grid(True)
    plt.suptitle("Strict vs Actor Trajectories")
    plt.tight_layout()
    plt.show()

plot_strict_vs_actor_trajectories(actor)

# === 输出 t=0 时的控制比较 ===
print("\n=== Value & Control at t=0 ===")
x0_list = [
    torch.tensor([2.0, 2.0]),
    torch.tensor([2.0, -2.0]),
    torch.tensor([-2.0, -2.0]),
    torch.tensor([-2.0, 2.0]),
]
t0 = torch.tensor([0.0])
for x0 in x0_list:
    v = exact_value_function(0.0, x0).item()
    mean = mean_control(0.0, x0)
    actor_u = actor(t0, x0.view(1, -1)).view(-1)
    print(f"x0 = {x0.tolist()}:\n  Value(0,x) = {v:.4f}\n  Mean control = {mean.tolist()}\n  Actor control = {actor_u.tolist()}\n")
