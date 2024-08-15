import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
x0 = 100  # Valor inicial de x(t)
y0 = 10   # Valor inicial de y(t)
h = 0.1   # Tamaño de paso (1 mes)
n = int(5 * 12 / h)  # 5 años en pasos de h meses

# Solución
t_values, populations = runge_kutta_4_system([f1, f2], [x0, y0], 0, h, n)
populations = np.array(populations)
x_values = populations[:, 0]
y_values = populations[:, 1]

# Graficar las soluciones
plt.figure(figsize=(10, 6))
plt.plot(t_values, x_values, label='Población de x(t)')
plt.plot(t_values, y_values, label='Población de y(t)')
plt.xlabel('Tiempo (meses)')
plt.ylabel('Población')
plt.title('Evolución de las poblaciones x(t) e y(t) en el tiempo')
plt.legend()
plt.grid(True)
plt.show()

# Poblaciones después de 5 años
print(f"Población de x después de 5 años: {x_values[-1]:.2f}")
print(f"Población de y después de 5 años: {y_values[-1]:.2f}")

# Estimar el período o ciclo de repetición
# Identificar puntos máximos de las poblaciones para estimar el período
peaks_x, _ = find_peaks(x_values)
peaks_y, _ = find_peaks(y_values)

if len(peaks_x) > 1:
    period_x = (peaks_x[1] - peaks_x[0]) * h
    print(f"Período aproximado de la población x(t): {period_x:.2f} meses")
else:
    print("No se pudo estimar el período de x(t).")

if len(peaks_y) > 1:
    period_y = (peaks_y[1] - peaks_y[0]) * h
    print(f"Período aproximado de la población y(t): {period_y:.2f} meses")
else:
    print("No se pudo estimar el período de y(t).")
