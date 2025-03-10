
  
Jiaqi Shi  s2751979


# README - Exercise 1.1: Solving LQR using Riccati ODE

## Overview
This section of the project implements a **Linear Quadratic Regulator (LQR)** solver using the Riccati differential equation. The objective is to compute the **value function** and **optimal control policy** for a given **linear system** over a finite time horizon.

The implementation includes:
1. **Numerical solution of the Riccati ODE** to obtain the time-dependent matrix \( S(t) \).
2. **Computation of the value function** \( v(t, x) = x^T S(t) x \).
3. **Derivation of the optimal control** \( a^*(t, x) = -D^{-1} M^T S(t) x \).
4. **Graphical visualization** of results including:
   - Evolution of Riccati matrix elements over time.
   - Value function trajectory.
   - Optimal control trajectory.

## Implementation Details

### **Mathematical Formulation**
We consider the controlled system:
\[
    dX_s = (H X_s + M \alpha_s) ds + \sigma dW_s,
\]
where \( X_s \) represents the state variables and \( \alpha_s \) is the control input. The objective is to minimize the cost functional:
\[
    J^\alpha (t, x) = \mathbb{E}^{t,x} \left[ \int_t^T \left( X_s^T C X_s + \alpha_s^T D \alpha_s \right) ds + X_T^T R X_T \right].
\]

Using the **Bellman Equation**, the optimal solution is characterized by the Riccati ODE:
\[
    S'(t) = S(t) M D^{-1} M^T S(t) - H^T S(t) - S(t) H - C, \quad S(T) = R.
\]

The optimal control law is derived as:
\[
    \alpha^*(t, x) = -D^{-1} M^T S(t) x.
\]

### **Code Structure**
The implementation consists of a **Python class** `LQR_Solver`, which provides:
- **`solve_riccati()`**: Solves the Riccati equation numerically using `solve_ivp`.
- **`compute_value_function()`**: Computes \( v(t, x) \) for given time and state.
- **`compute_optimal_control()`**: Computes \( a^*(t, x) \) based on \( S(t) \).
- **`plot_riccati_solution()`**: Plots the evolution of Riccati matrix elements.
- **`plot_value_function()`**: Plots \( v(t, x) \) over time.
- **`plot_optimal_control()`**: Plots the optimal control trajectory.

### **Libraries Used**
Only the following libraries are used as per project requirements:
- `numpy` (Matrix operations)
- `scipy` (ODE solver `solve_ivp`)
- `matplotlib` (Plotting)
- `torch` (Tensor computations)

### **How to Run the Code**
1. **Clone the Git repository**
```sh
    git clone <repository-link>
    cd <repository-folder>
```
2. **Run the Python script**
```sh
    python function.py
```
3. **Expected Output:**
- **Numerical values for value function and optimal control:**
  ```sh
  Value Function at t=0.5, x=[[1.0, 0.5]] -> v(t, x) = [1.7642531]
  Optimal Control at t=0.5, x=[[1.0, 0.5]] -> a*(t, x) = [[-0.48849595]]
  ```
- **Three plots should appear:**
  - **Convergence of Riccati solution** (S-matrix elements over time)
  - **Evolution of Value Function**
  - **Evolution of Optimal Control**

### **Example Output Graphs**
1. **Riccati Matrix Convergence:**
   ![Riccati Matrix](images/riccati_solution.png)
2. **Value Function Evolution:**
   ![Value Function](images/value_function.png)
3. **Optimal Control Evolution:**
   ![Optimal Control](images/optimal_control.png)

### **Contributors & Contributions**
| Name | Student Number | Contribution |
|------|---------------|--------------|
| A. Student | 123456 | Riccati ODE Solver |
| B. Student | 789012 | Value Function Computation & Plotting |
| C. Student | 345678 | Optimal Control Computation & Plotting |

### **Remarks**
- The code is structured to allow easy modifications for **higher-dimensional** LQR problems.
- The **convergence of Riccati elements** aligns with theoretical expectations from optimal control theory.
- The **optimal control trajectory** follows an expected behavior of stabilizing the system over time.

---
✅ **This README provides a clear guide for understanding and reproducing the results of Exercise 1.1.** If you encounter any issues, please refer to the repository documentation or contact the contributors. 🚀

