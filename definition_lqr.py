import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Define problem parameters from project spec
H = [[0.5, 0.5], [0.0, 0.5]]      # Drift matrix for state
M = [[1.0, 1.0], [0.0, 1.0]]      # Control influence matrix
sigma = np.eye(2) * 0.5           # Diffusion coefficient (noise strength)
C = [[1.0, 0.1], [0.1, 1.0]]      # Running cost for state
D = [[1.0, 0.1], [0.1, 1.0]]
D = (np.array(D) * 0.1).tolist()  # Running cost for control (set to Identity)
R = [[1.0, 0.3], [0.3, 1.0]]
R = (np.array(R) * 10.0).tolist() # Terminal cost
T = 0.5         # Terminal time
N = 1000        # Number of time steps
tau = 0.1       # Entropy regularization parameter
gamma = 10.0    # Control noise variance

### Exercise 1.1 

class StrictLQR:
    def __init__(self, H, M, sigma, C, D, R, T, N):
        """
        Initialize the LQR problem with system and cost matrices.
        Solves the Riccati equation on a fixed time grid [0, T].

        Args:
            H, M, sigma, C, D, R: LQR system and cost matrices (list or numpy array)
            T: Time horizon (float)
            N: Number of time grid steps (int)
        """
        self.H = torch.tensor(H, dtype=torch.float64)
        self.M = torch.tensor(M, dtype=torch.float64)
        self.sigma = torch.tensor(sigma, dtype=torch.float64)
        self.C = torch.tensor(C, dtype=torch.float64)
        self.D = torch.tensor(D, dtype=torch.float64)
        self.R = torch.tensor(R, dtype=torch.float64)
        self.T = T
        self.N = N
        self.time_grid = torch.linspace(0, T, N)  # Create uniform time grid from 0 to T
        self.S_t = self.solve_riccati()  # Solve Riccati ODE on this grid

    def riccati_ode(self, t, S_flat):
        """
        Right-hand side of the Riccati ODE, reshaped into 2x2 matrix.
        S'(t) = SMD^{-1}M^T S - H^T S - S H - C

        Args:
            t: Time (not used explicitly since coefficients are constant)
            S_flat: Flattened matrix S(t) of shape (4,)

        Returns:
            dSdt_flat: Flattened derivative dS/dt of shape (4,)
        """
        S = torch.tensor(S_flat.reshape(2, 2), dtype=torch.float64)
        D_inv = torch.linalg.inv(self.D)
        dSdt = S @ self.M @ D_inv @ self.M.T @ S - self.H.T @ S - S @ self.H - self.C
        return dSdt.numpy().flatten()

    def solve_riccati(self):
        """
        Solve the Riccati differential equation backward in time from T to 0.

        Returns:
            Tensor of shape (N, 2, 2) where each slice is S(t_n)
        """
        S_T = self.R.numpy().flatten()
        sol = solve_ivp(
            self.riccati_ode, [self.T, 0], S_T,
            method='BDF', t_eval=np.linspace(self.T, 0, self.N),
            rtol=1e-8, atol=1e-10
        )
        # Convert solution to tensor, reverse in time, reshape to (N, 2, 2)
        S_values = sol.y.T[:, :4].reshape(-1, 2, 2)
        S_values = torch.tensor(np.ascontiguousarray(S_values[::-1]), dtype=torch.float64)
        return S_values

    def value_function(self, t, x):
        """
        Compute the value function v(t, x) = x^T S(t) x + \int_t^T tr(sigma sigma^T S(r)) dr

        Args:
            t: Scalar time in [0, T]
            x: State vector (2D torch tensor)

        Returns:
            Scalar value v(t, x)
        """
        idx = min(np.searchsorted(self.time_grid.numpy(), t, side='right') - 1, len(self.S_t) - 1)
        S_t = self.S_t[idx]

        # Compute trace term for integral part from t to T
        traces = torch.tensor([
            torch.trace(self.sigma @ self.sigma.T @ S) for S in self.S_t[idx:]
        ], dtype=torch.float64)
        time_subgrid = self.time_grid[idx:]
        integral_term = torch.trapz(traces, time_subgrid)

        return (x @ S_t @ x + integral_term).item()

    def optimal_control(self, t, x):
        """
        Compute the optimal control a(t, x) = -D^{-1} M^T S(t) x

        Args:
            t: Scalar time
            x: State vector

        Returns:
            Control vector (torch tensor shape (2,))
        """
        idx = min(np.searchsorted(self.time_grid.numpy(), t, side='right') - 1, len(self.S_t) - 1)
        S_t = self.S_t[idx]
        D_inv = torch.linalg.inv(self.D)
        return -D_inv @ self.M.T @ S_t @ x

    def value_function_batch(self, t_tensor, x_tensor):
        """
        Batched evaluation of value function v(t, x) over many (t, x) pairs.

        Args:
            t_tensor: Tensor of shape (T,) of time points
            x_tensor: Tensor of shape (T, 2) of state vectors

        Returns:
            Tensor of shape (T,) of v(t, x)
        """
        result = []
        for t, x in zip(t_tensor, x_tensor):
            result.append(self.value_function(t.item(), x))
        return torch.tensor(result, dtype=torch.float64)

    def optimal_control_batch(self, t_tensor, x_tensor):
        """
        Batched evaluation of control a(t, x) over many (t, x) pairs.

        Args:
            t_tensor: Tensor of shape (T,) of time points
            x_tensor: Tensor of shape (T, 2) of state vectors

        Returns:
            Tensor of shape (T, 2) of optimal controls
        """
        result = []
        for t, x in zip(t_tensor, x_tensor):
            result.append(self.optimal_control(t.item(), x))
        return torch.stack(result)
    

### Exercise 1.2

class LQRMonteCarlo(StrictLQR):
    def simulate(self, x0, num_samples, time_steps):
        """
        Run Monte Carlo simulation of SDE under optimal control to estimate expected cost.

        Args:
            x0: initial state (2D torch tensor)
            num_samples: number of Monte Carlo samples
            time_steps: number of time discretization steps

        Returns:
            Mean absolute error between MC estimate and exact value function
        """
        tau = self.T / time_steps
        time_grid = torch.linspace(0, self.T, time_steps + 1)
        errors = []

        final_costs = []
        for _ in range(num_samples):
            X = torch.zeros((time_steps + 1, 2), dtype=torch.float64)
            X[0] = x0.clone()
            W = torch.randn((time_steps, 2), dtype=torch.float64) * np.sqrt(tau)
            cost = 0.0

            for k in range(time_steps):
                t_k = time_grid[k].item()
                a_k = self.optimal_control(t_k, X[k])
                drift = self.H @ X[k] + self.M @ a_k
                diffusion = self.sigma @ W[k]
                X[k + 1] = X[k] + tau * drift + diffusion

                running_cost = X[k] @ self.C @ X[k] + a_k @ self.D @ a_k
                cost += running_cost * tau

            terminal_cost = X[-1] @ self.R @ X[-1]
            final_costs.append(cost + terminal_cost)

        mc_estimate = torch.tensor(final_costs).mean().item()
        exact_value = self.value_function(0.0, x0)
        return abs(mc_estimate - exact_value)

    def run_experiment_1(self, num_samples, time_steps_list):
        """
        Experiment 1: Fix number of samples, vary time steps
        """
        errors = []
        for N in time_steps_list:
            err = self.simulate(torch.tensor([1.0, 1.0], dtype=torch.float64), num_samples, N)
            print(f"Time steps: {N}, Error: {err}")
            errors.append(err)

        plt.figure()
        plt.loglog(time_steps_list, errors, 'o-', label="Empirical Error")
        ref_x = np.array(time_steps_list)
        ref_y = errors[0] * (ref_x / ref_x[0])**-1
        plt.loglog(ref_x, ref_y, 'y--', label="Reference slope -1")
        plt.xlabel("Time Steps")
        plt.ylabel("Error")
        plt.title("LQR Error vs Time Steps")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

    def run_experiment_2(self, fixed_time_steps, num_samples_list):
        """
        Experiment 2: Fix time step, vary number of Monte Carlo samples
        """
        errors = []
        for M in num_samples_list:
            err = self.simulate(torch.tensor([1.0, 1.0], dtype=torch.float64), M, fixed_time_steps)
            print(f"Monte Carlo Samples: {M}, Error: {err}")
            errors.append(err)

        plt.figure()
        plt.loglog(num_samples_list, errors, 'o-', label="Empirical Error")
        ref_x = np.array(num_samples_list)
        ref_y = errors[0] * (ref_x / ref_x[0])**-0.5
        plt.loglog(ref_x, ref_y, 'y--', label="Reference slope -0.5")
        plt.xlabel("Monte Carlo Samples")
        plt.ylabel("Error")
        plt.title("LQR Error vs Monte Carlo Samples")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()


### Exercise 2

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

