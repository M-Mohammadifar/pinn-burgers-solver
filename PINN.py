import tensorflow as tf
import numpy as np

# Define the viscosity coefficient
nu = 0.01

# Define the neural network model
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(50, activation='tanh') for _ in range(4)
        ]
        self.output_layer = tf.keras.layers.Dense(1)  # Single output for u(x, t)

    def call(self, inputs):
        x, t = inputs[:, 0:1], inputs[:, 1:2]
        u = tf.concat([x, t], 1)  # Concatenate x and t as input
        for layer in self.hidden_layers:
            u = layer(u)
        return self.output_layer(u)  # Output u(x, t)

# Instantiate the model
model = PINN()

# Define a function for the PDE residuals (Burgers' equation)
def burgers_residual(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(tf.concat([x, t], axis=1))
        u_x = tape.gradient(u, x)
    u_t = tape.gradient(u, t)
    u_xx = tape.gradient(u_x, x)
    return u_t + u * u_x - nu * u_xx  # Burgers' equation residual

# Custom loss function
def custom_loss(model, x_in, t_in, x_boundary, t_boundary, u_boundary, x_init, t_init, u_init):
    # PDE residuals
    residuals = burgers_residual(model, x_in, t_in)
    residual_loss = tf.reduce_mean(tf.square(residuals))

    # Boundary loss
    u_b = model(tf.concat([x_boundary, t_boundary], axis=1))
    boundary_loss = tf.reduce_mean(tf.square(u_b - u_boundary))

    # Initial condition loss
    u_i = model(tf.concat([x_init, t_init], axis=1))
    init_loss = tf.reduce_mean(tf.square(u_i - u_init))

    # Combine losses
    total_loss = residual_loss + boundary_loss + init_loss
    return total_loss

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Training step function
def train_step(model, x_in, t_in, x_boundary, t_boundary, u_boundary, x_init, t_init, u_init):
    with tf.GradientTape() as tape:
        loss = custom_loss(model, x_in, t_in, x_boundary, t_boundary, u_boundary, x_init, t_init, u_init)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Generate training data
x_in = tf.random.uniform((100, 1), -1, 1)
t_in = tf.random.uniform((100, 1), 0, 1)

x_boundary = tf.concat([tf.constant(-1.0, shape=(50, 1)), tf.constant(1.0, shape=(50, 1))], axis=0)
t_boundary = tf.random.uniform((100, 1), 0, 1)
u_boundary = tf.zeros((100, 1))

x_init = tf.random.uniform((100, 1), -1, 1)
t_init = tf.zeros((100, 1))
u_init = -tf.sin(np.pi * x_init)

# Training loop
epochs = 5000
for epoch in range(epochs):
    loss = train_step(model, x_in, t_in, x_boundary, t_boundary, u_boundary, x_init, t_init, u_init)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
# Save the model after training
model.save('pinn_burgers_model')

# Generate a grid of (x, t) points for predictions
x_pred = np.linspace(-1, 1, 100).reshape(-1, 1)
t_pred = np.linspace(0, 1, 100).reshape(-1, 1)
x_grid, t_grid = np.meshgrid(x_pred, t_pred)
x_flat = x_grid.flatten().reshape(-1, 1)
t_flat = t_grid.flatten().reshape(-1, 1)
input_points = np.hstack((x_flat, t_flat))

# Predict using the trained model
predictions = model.predict(input_points)

# Save predictions to a file
np.savetxt('pinn_burgers_predictions.csv', np.hstack((input_points, predictions)), delimiter=',', header='x,t,u', comments='')