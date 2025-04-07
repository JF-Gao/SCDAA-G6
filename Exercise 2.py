import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SoftLQR:
    def __init__(self, H, M, sigma, C, D, R, T, N, tau, gamma):
        # Define dynamics and cost matrices
        self.H = torch.tensor(H, dtype=torch.float64)
        self.M = torch.tensor(M, dtype=torch.float64)
        self.sigma = torch.tensor(sigma, dtype=torch.float64)
        self.C = torch.tensor(C, dtype=torch.float64)
        self.D = torch.tensor(D, dtype=torch.float64)
        self.R = torch.tensor(R, dtype=torch.float64)
        self.T = T
        self.N = N
        self.tau = tau
        self.gamma = gamma

        self.time_grid = torch.linspace(0, T, N + 1)
        self.S_t = self.solve_riccati()  # Solve Riccati equation once on init

    def riccati_ode(self, t, S_flat):
        # Right-hand side of Riccati ODE for strict LQR
        S = torch.tensor(S_flat.reshape(2, 2), dtype=torch.float64)
        D_inv = torch.linalg.inv(self.D)
        dSdt = S @ self.M @ D_inv @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        return dSdt.numpy().flatten()

    def solve_riccati(self):
        # Solve the Riccati equation backward in time using SciPy
        S_T = self.R.numpy().flatten()
        sol = solve_ivp(
            self.riccati_ode, [self.T, 0], S_T,
            method='BDF', t_eval=np.linspace(self.T, 0, self.N + 1),
            rtol=1e-8, atol=1e-10
        )
        S_values = sol.y.T.reshape(-1, 2, 2)[::-1]  # reverse time
        return torch.tensor(np.ascontiguousarray(S_values), dtype=torch.float64)

    def get_S(self, t):
        # Get S(t) from precomputed grid
        idx = min(np.searchsorted(self.time_grid.numpy(), t, side='right') - 1, len(self.S_t) - 1)
        return self.S_t[idx]

    def mean_control(self, t, x):
        # Deterministic mean control (strict LQR)
        S_t = self.get_S(t)
        D_inv = torch.linalg.inv(self.D)
        return -D_inv @ self.M.T @ S_t @ x

    def sample_control(self, t, x):
        # Sample stochastic soft control using Gaussian perturbation
        mean = self.mean_control(t, x)
        noise = torch.randn_like(mean) * np.sqrt(self.tau * self.gamma)
        return mean + noise

    def value_function(self, t, x):
        # Compute soft LQR value function as quadratic form plus noise trace integral
        S_t = self.get_S(t)
        term1 = x @ S_t @ x
        trace_integral = torch.tensor([
            torch.trace(self.sigma @ self.sigma.T @ S) for S in self.S_t
        ])
        return term1 + torch.trapz(trace_integral, self.time_grid)

    def simulate_trajectory(self, x0, use_soft=False, seed=None):
        # Simulate one trajectory under strict or soft LQR
        if seed is not None:
            torch.manual_seed(seed)

        dt = self.T / self.N
        x = torch.zeros((self.N + 1, 2), dtype=torch.float64)
        x[0] = x0
        W = torch.randn((self.N, 2), dtype=torch.float64) * np.sqrt(dt)

        for k in range(self.N):
            t_k = self.time_grid[k].item()
            a = self.sample_control(t_k, x[k]) if use_soft else self.mean_control(t_k, x[k])
            drift = self.H @ x[k] + self.M @ a
            diffusion = self.sigma @ W[k]
            x[k + 1] = x[k] + dt * drift + diffusion

        return x

    def plot_trajectories(self, x0_list):
        # Plot both strict and soft LQR trajectories for multiple initial states
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()

        for i, x0 in enumerate(x0_list):
            strict_traj = self.simulate_trajectory(x0, use_soft=False, seed=42)
            soft_traj = self.simulate_trajectory(x0, use_soft=True, seed=42)

            axs[i].plot(strict_traj[:, 0], strict_traj[:, 1], label="Strict LQR", color="blue")
            axs[i].plot(soft_traj[:, 0], soft_traj[:, 1], label="Soft LQR", color="orange", linestyle="--")
            axs[i].scatter(*x0.numpy(), color="black")
            axs[i].set_title(f"Trajectory from x0 = {x0.tolist()}")
            axs[i].set_xlabel("x[0]")
            axs[i].set_ylabel("x[1]")
            axs[i].legend()
            axs[i].grid(True)

        plt.suptitle("Strict vs Soft LQR Trajectories")
        plt.tight_layout()
        plt.show()

    def print_value_and_controls(self, x0_list):
        # For each x0, print value function, mean and sampled controls
        print("\n===== Value Function and Controls at t=0 =====")
        for x0 in x0_list:
            t0 = 0.0
            value = self.value_function(t0, x0)
            mean_control = self.mean_control(t0, x0)
            sampled_control = self.sample_control(t0, x0)

            print(f"\nx0 = {x0.tolist()}")
            print(f"Value v(0,x) = {value.item():.4f}")
            print(f"Mean control: {mean_control}")
            print(f"Sampled control: {sampled_control}")


if __name__ == '__main__':
    # Define problem parameters
    H = [[0.5, 0.5], [0.0, 0.5]]
    M = [[1.0, 1.0], [0.0, 1.0]]
    sigma = np.eye(2) * 0.5
    C = [[1.0, 0.1], [0.1, 1.0]]
    D = (np.array([[1.0, 0.1], [0.1, 1.0]]) * 0.1).tolist()
    R = (np.array([[1.0, 0.3], [0.3, 1.0]]) * 10.0).tolist()
    T = 0.5
    N = 100
    tau = 0.1
    gamma = 10.0

    soft_lqr = SoftLQR(H, M, sigma, C, D, R, T, N, tau, gamma)

    x0_list = [
        torch.tensor([2.0, 2.0], dtype=torch.float64),
        torch.tensor([2.0, -2.0], dtype=torch.float64),
        torch.tensor([-2.0, -2.0], dtype=torch.float64),
        torch.tensor([-2.0, 2.0], dtype=torch.float64),
    ]

    soft_lqr.plot_trajectories(x0_list)
    soft_lqr.print_value_and_controls(x0_list)