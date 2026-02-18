# Physics-Informed Neural Network (PINN) for Burgers' Equation

This repository contains a **Physics-Informed Neural Network (PINN)** implemented in TensorFlow to solve the 1D viscous Burgers' equation.

## 🏗 Why This Project?

Traditional structural analysis often relies on Finite Element Analysis (FEA). This project explores an alternative: using deep learning to solve the underlying partial differential equations (PDEs) directly by incorporating physics into the loss function.

## 🚀 The Physics

The model is trained to satisfy the **viscous Burgers' equation**:
$$u_t + u u_x - \nu u_{xx} = 0$$

- **Viscosity ($\nu$):** 0.01
- **Domain:** $x \in [-1, 1]$, $t \in [0, 1]$
- **Initial Condition:** $u(x, 0) = -\sin(\pi x)$

## 🛠 Tech Stack & Architecture

- **Framework:** TensorFlow 2.x / Keras
- **Architecture:** Fully Connected Neural Network (MLP)
  - 4 Hidden Layers (50 neurons each)
  - Activation: `tanh` (Chosen for its infinite differentiability)
- **Loss Function:** $$\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{Boundary} + \mathcal{L}_{Initial}$$
- **Optimization:** Adam Optimizer with Automatic Differentiation via `tf.GradientTape`.

## 📈 Performance & Results

The model successfully captures the non-linear "shock wave" behavior characteristic of the Burgers' equation as $t \to 1$.

## 💻 How to Run

1. Clone the repo:
   ```bash
   git clone [https://github.com/M-Mohammadifar/pinn-burgers-solver.git](https://github.com/M-Mohammadifar/pinn-burgers-solver.git)
   ```
