import numpy as np
import matplotlib.pyplot as plt

def dx_dt(x, y):
    return 0.2 * x - 0.005 * x * y

def dy_dt(x, y):
    return -0.5 * y + 0.01 * x * y

x_values = np.linspace(0, 200, 20)
y_values = np.linspace(0, 200, 20)
X, Y = np.meshgrid(x_values, y_values)

U = dx_dt(X, Y)
V = dy_dt(X, Y)

plt.quiver(X, Y, U, V)
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Campo Vectorial del Sistema de EDOs')
plt.grid(True)
plt.show()
