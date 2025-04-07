# README

## Overview
This repository contains an end-to-end implementation of an actor-critic reinforcement learning algorithm for solving a continuous-time **soft LQR** problem, based on the coursework specification. 

The project is divided into five main parts, progressing from solving the strict LQR via Riccati equations to implementing and validating the actor-critic learning loop.

Each part of the code is run locally via pycharm. Specific code names and partial output images can be seen in the folder on the left.

### **File Structure**
- Exercise_1.py        # Strict LQR: Riccati solution + Monte Carlo (Ex. 1.1, 1.2)
- Exercise_2.py        # Soft LQR: Entropy-regularized control & trajectory sim (Ex. 2.1)
- Exercise_3.py        # Critic-only learning using optimal policy (Ex. 3.1)
- Exercise_4.py        # Actor-only supervised training using value function (Ex. 4.1)
- Exercise_5.py        # Full actor-critic algorithm (Ex. 5.1)
- plot                 # Each plot corresponds to a code output image of the exercise.

### **Libraries Used**
Only the following libraries are used as per project requirements:
- `numpy` (Matrix operations)
- `scipy` (ODE solver `solve_ivp`)
- `matplotlib` (Plotting)
- `torch` (Tensor computations)


## **How to Run the Code**
1. **Clone the Git repository**
```sh
    git clone <repository-link>
    cd <repository-folder>
```
2. **Run the Python script**
```sh
    python Exercise_1.py   # Strict LQR
    python Exercise_2.py   # Soft LQR
    python Exercise_3.py   # Critic-only
    python Exercise_4.py   # Actor-only
    python Exercise_5.py   # Actor-Critic
```
3. **Expected Output——Example for Exercise 2:**
- **Numerical values for value function and optimal control:**
```sh
    (Part of the answers)
    x0 = [2.0, 2.0]
    Value v(0,x) = 2.3822
    Mean control: tensor([ -2.5540, -10.3992], dtype=torch.float64)
    Sampled control: tensor([ -3.0248, -10.3863], dtype=torch.float64)
```
- **Output Graphs**
  - See plot_1-Exercise 2 in the left-hand folder
4. **Conclusion**
  - All you need to do is either click the Run button directly in codespace to get the values and images, or save the code locally and run it directly to get the results.


## **Contributors & Contributions**
|    Name     | Student Number | Contribution |
|-------------|----------------|--------------|
|  Jiaqi Shi  |    s2751979    |     1/3      |
| Jinfeng Gao |    s2694132    |     1/3      |
|  Junda Lu   |    s2656307    |     1/3      |

All members were involved in writing all the code for the five exercises as well as the reports, so the contributions are equal.
