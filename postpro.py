import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
data = pd.read_csv('pinn_burgers_predictions.csv')
x = data['x'].values
t = data['t'].values
u = data['u'].values

# Reshape data for heatmap
x_unique = np.unique(x)
t_unique = np.unique(t)
u_grid = u.reshape(len(t_unique), len(x_unique))

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(u_grid, extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()], origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='u(x, t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('PINN Prediction of u(x, t) for the 1D Burgers\' Equation')
plt.show()
