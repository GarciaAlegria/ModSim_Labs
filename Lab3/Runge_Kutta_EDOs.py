import numpy as np

def runge_kutta_4_system(f_system, y0, x0, h, n):
    """
    Implementacion del metodo de Runge-Kutta de orden 4 para un sistema de EDOs.
    
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

# Ejemplo de un sistema de EDOs:
# y1' = -2 * x * y1
# y2' = x^2 * y2

def f1(x, y1, y2):
    return -2 * x * y1

def f2(x, y1, y2):
    return x**2 * y2

# Parámetros
y0 = [1, 1]
x0 = 0
h = 0.1
n = 10

# Solución
x_values, y_values = runge_kutta_4_system([f1, f2], y0, x0, h, n)
print("Valor para x1 y x2 ", x_values)
print("")
print("Valor para y1 y y2", y_values)
