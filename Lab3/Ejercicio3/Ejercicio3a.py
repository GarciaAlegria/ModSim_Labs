import numpy as np
import matplotlib.pyplot as plt

# Define the derivatives based on the image
def dx_dt(x, y):
    return 0.5*x - 0.001*x**2 - x*y

def dy_dt(x, y):
    return -0.2*y + 0.1*x*y

# Create the grid for plotting
x_values = np.linspace(0, 10, 20)  # Adjusted range for x
y_values = np.linspace(0, 5, 20)   # Adjusted range for y
X, Y = np.meshgrid(x_values, y_values)

# Calculate the derivatives at each point
U = dx_dt(X, Y)
V = dy_dt(X, Y)

# Plot the vector field (phase plane)
plt.quiver(X, Y, U, V)
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Campo Vectorial (Plano de Fase)')
plt.grid(True)
plt.xlim(0, 10)   # Set x-axis limits
plt.ylim(0, 5)    # Set y-axis limits
plt.show()