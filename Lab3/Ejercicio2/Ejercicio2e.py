import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4_system(f_system, y0, x0, h, n):
    """
    Implementación del método de Runge-Kutta de orden 4 para un sistema de EDOs.
    
    :param f_system: Lista de funciones que definen el sistema de EDOs [f1, f2, ..., fm]
    :param y0: Lista con los valores iniciales de [y1, y2, ..., ym]
    :param x0: Valor inicial de x
    :param h: Tamaño del paso
    :param n: Número de pasos
    :return: Lista de valores de x y matriz de valores de y
    """
    m = len(y0)
    x_values = [x0]
    y_values = [y0]
    
    for i in range(n):
        k1 = np.array([h * f(x0, *y0) for f in f_system])
        k2 = np.array([h * f(x0 + 0.5 * h, *(y0 + 0.5 * k1)) for f in f_system])
        k3 = np.array([h * f(x0 + 0.5 * h, *(y0 + 0.5 * k2)) for f in f_system])
        k4 = np.array([h * f(x0 + h, *(y0 + k3)) for f in f_system])
        
        y0 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x0 = x0 + h
        
        x_values.append(x0)
        y_values.append(y0)
    
    return x_values, y_values

# Definir las funciones del sistema de EDOs:
# x'(t) = 0.2 * x - 0.005 * x * y
# y'(t) = -0.5 * y + 0.01 * x * y

def f1(t, x, y):
    return 0.2 * x - 0.005 * x * y

def f2(t, x, y):
    return -0.5 * y + 0.01 * x * y

# Parámetros
x0_values = [70, 100]  # Valores iniciales de x(t)
y0_values = [30, 10]   # Valores iniciales de y(t)
h = 0.1   # Tamaño de paso (1 mes)
n = int(5 * 12 / h)  # 5 años en pasos de h meses

# Graficar campo vectorial
X, Y = np.meshgrid(np.linspace(0, 160, 20), np.linspace(0, 160, 20))
U = 0.2 * X - 0.005 * X * Y
V = -0.5 * Y + 0.01 * X * Y

plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U, V, color='gray', alpha=0.6)

# Solución para ambas condiciones iniciales
for i in range(2):
    x0 = x0_values[i]
    y0 = y0_values[i]
    
    t_values, populations = runge_kutta_4_system([f1, f2], [x0, y0], 0, h, n)
    populations = np.array(populations)
    x_values = populations[:, 0]
    y_values = populations[:, 1]
    
    # Graficar la trayectoria
    plt.plot(x_values, y_values, label=f'Trayectoria (x0={x0}, y0={y0})')
    
    # Marcar el punto inicial
    plt.plot(x0, y0, 'o', label=f'Inicial (x0={x0}, y0={y0})')
    
    # Marcar el punto final
    plt.plot(x_values[-1], y_values[-1], 's', label=f'Final (x={x_values[-1]:.2f}, y={y_values[-1]:.2f})')

plt.xlabel('Población de x')
plt.ylabel('Población de y')
plt.title('Plano de fase de las poblaciones x(t) e y(t)')
plt.legend()
plt.grid(True)
plt.show()
