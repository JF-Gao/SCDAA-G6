"""
Exercis 1.1
"""
import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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


# Example test
if __name__ == "__main__":
    # Define problem parameters from project spec
    H = [[0.5, 0.5], [0.0, 0.5]]
    M = [[1.0, 1.0], [0.0, 1.0]]
    sigma = np.eye(2) * 0.5
    C = [[1.0, 0.1], [0.1, 1.0]]
    D = [[1.0, 0.1], [0.1, 1.0]]
    D = (np.array(D) * 0.1).tolist()
    R = [[1.0, 0.3], [0.3, 1.0]]
    R = (np.array(R) * 10.0).tolist()
    T = 0.5
    N = 1000

    lqr = StrictLQR(H, M, sigma, C, D, R, T, N)
    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    y = torch.tensor([2.0, 2.0], dtype=torch.float64)

    print("Value function v(0, x):", lqr.value_function(0, x))
    print("Optimal control a(0, x):", lqr.optimal_control(0, x))
    print("Value function v(0, y):", lqr.value_function(0, y))
    print("Optimal control a(0, y):", lqr.optimal_control(0, y))

    # Batch evaluation examples
    t_list = torch.tensor([0.0, 0.1], dtype=torch.float64)
    x_list = torch.stack([x, y])
    print("Batch value v(t, x):", lqr.value_function_batch(t_list, x_list))
    print("Batch control a(t, x):", lqr.optimal_control_batch(t_list, x_list))

"""
Exericse 1.2
"""

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


# Example usage for Exercise 1.2
if __name__ == '__main__':
    # Problem setup
    H = [[0.5, 0.5], [0.0, 0.5]]
    M = [[1.0, 1.0], [0.0, 1.0]]
    sigma = np.eye(2) * 0.5
    C = [[1.0, 0.1], [0.1, 1.0]]
    D = (np.array([[1.0, 0.1], [0.1, 1.0]]) * 0.1).tolist()
    R = (np.array([[1.0, 0.3], [0.3, 1.0]]) * 10.0).tolist()
    T = 0.5
    N = 1000

    lqr_mc = LQRMonteCarlo(H, M, sigma, C, D, R, T, N)

    # Experiment 1: Fix samples, vary time steps
    time_steps_list = [2**i for i in range(1, 12)]
    lqr_mc.run_experiment_1(num_samples=10000, time_steps_list=time_steps_list)

    # Experiment 2: Fix time step, vary number of samples
    num_samples_list = [2 * 4**i for i in range(6)]  # 2, 8, 32, ..., 2048
    lqr_mc.run_experiment_2(fixed_time_steps=10000, num_samples_list=num_samples_list)


"""
Experiment 1 (Fix samples, vary time steps):
We observed an initial error decay at the expected rate of O(1/N), corresponding to the Euler–Maruyama method. 
Beyond 𝑁≈256, the error plateaued due to Monte Carlo variance dominating.

Experiment 2 (Fix time steps, vary samples):
The observed error decayed approximately as O(1/ sqrt(M)), in line with Monte Carlo theory. 
Some fluctuations appeared for small 𝑀, but the overall convergence trend is clear.

These results confirm both the theoretical convergence rates and the correctness of our implementation.
"""
