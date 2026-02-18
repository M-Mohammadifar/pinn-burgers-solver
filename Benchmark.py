import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'benchmark_data.csv' contains columns 'x', 't', 'u_benchmark'
benchmark_data = pd.read_csv('benchmark_data.csv')
u_benchmark = benchmark_data['u_benchmark'].values
error = np.abs(u - u_benchmark)

# Reshape error data for heatmap
error_grid = error.reshape(len(t_unique), len(x_unique))

# Plot error heatmap
plt.figure(figsize=(8, 6))
plt.imshow(error_grid, extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()], origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(label='Absolute Error |u_pinn - u_benchmark|')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Error Heatmap between PINN and Benchmark Solution')
plt.show()
