def runge_kutta_4_single(f, y0, x0, h, n):
    """
    Implementacion del metodo de Runge-Kutta de orden 4 para una EDO.
    
    :param f: Función que define la EDO, f(x, y)
    :param y0: Valor inicial de y
    :param x0: Valor inicial de x
    :param h: Tamaño del paso
    :param n: Número de pasos
    :return: Listas de valores de x y y
    """
    x_values = [x0]
    y_values = [y0]
    
    for i in range(n):
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + 0.5 * h, y0 + 0.5 * k1)
        k3 = h * f(x0 + 0.5 * h, y0 + 0.5 * k2)
        k4 = h * f(x0 + h, y0 + k3)
        
        y0 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x0 = x0 + h
        
        x_values.append(x0)
        y_values.append(y0)
    
    return x_values, y_values

# Ejemplo de una EDO: y' = -2 * x * y, con y(0) = 1
def f(x, y):
    return -2 * x * y

# Parámetros
y0 = 1
x0 = 0
h = 0.1
n = 10

# Solución
x_values, y_values = runge_kutta_4_single(f, y0, x0, h, n)
print("Valor para x ", x_values)
print("Valor para y ", y_values)
