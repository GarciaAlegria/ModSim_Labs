import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def criterio_paro(x_viejo, x_nuevo):
    return np.linalg.norm(x_nuevo - x_viejo)

def descenso_maximo_naive(f, df, x0, alpha, maxIter, tol, criterio_paro):
    """
    Implementa el método de descenso máximo naive.

    Args:
        f: Función objetivo.
        df: Gradiente de la función objetivo.
        x0: Punto inicial.
        alpha: Tamaño de paso.
        maxIter: Número máximo de iteraciones.
        tol: Tolerancia.
        criterio_paro: Función que define el criterio de paro.

    Returns:
        x: Mejor solución encontrada.
        xx: Secuencia de iteraciones.
        fx: Secuencia de valores de f(x).
        errores: Secuencia de errores.
        num_iter: Número de iteraciones.
        convergencia: Booleano que indica si se alcanzó la convergencia.
    """
    x = x0
    xx = [x]
    fx = [f(x)]
    errores = []
    num_iter = 0
    convergencia = False

    for _ in range(maxIter):
        direccion = -df(x)
        x_nuevo = x + alpha * direccion
        xx.append(x_nuevo)
        fx.append(f(x_nuevo))
        error = criterio_paro(x, x_nuevo)
        errores.append(error)
        x = x_nuevo
        num_iter += 1
        if error < tol:
            convergencia = True
            break

    return x, xx, fx, errores, num_iter, convergencia

# Función y gradiente para el inciso (a)
def f_a(x):
    return x[0]**4 + x[1]**4 - 4*x[0]*x[1] + (1/2)*x[1] + 1

def df_a(x):
    df_x = 4*x[0]**3 - 4*x[1]
    df_y = 4*x[1]**3 - 4*x[0] + (1/2)
    return np.array([df_x, df_y])

# Función y gradiente para el inciso (b) (Rosenbrock 2D)
def f_b(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def df_b(x):
    df_x = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    df_y = 200 * (x[1] - x[0]**2)
    return np.array([df_x, df_y])

# Función y gradiente para el inciso (c) (Rosenbrock 10D)
def f_c(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

def df_c(x):
    grad = np.zeros_like(x)
    for i in range(len(x) - 1):
        grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        grad[i+1] += 200 * (x[i+1] - x[i]**2)
    return grad

# Parámetros comunes
alpha = 0.001
maxIter = 10000
tol = 1e-6

# ----- Función (a) -----
x0_a = np.array([-3.1, -3.1])
x_a, xx_a, fx_a, errores_a, num_iter_a, convergencia_a = descenso_maximo_naive(f_a, df_a, x0_a, alpha, maxIter, tol, criterio_paro)

print(f"Solución (a): {x_a}, iteraciones: {num_iter_a}, convergencia: {convergencia_a}")

# ----- Función (b) -----
x0_b = np.array([-1.2, 1])
x_b, xx_b, fx_b, errores_b, num_iter_b, convergencia_b = descenso_maximo_naive(f_b, df_b, x0_b, alpha, maxIter, tol, criterio_paro)

print(f"Solución (b): {x_b}, iteraciones: {num_iter_b}, convergencia: {convergencia_b}")

# ----- Función (c) -----
x0_c = np.array([-1.2] * 10)
alpha_c = 0.0001  # Reducir tamaño de paso para la función en 10D
x_c, xx_c, fx_c, errores_c, num_iter_c, convergencia_c = descenso_maximo_naive(f_c, df_c, x0_c, alpha_c, maxIter, tol, criterio_paro)

print(f"Solución (c): {x_c}, iteraciones: {num_iter_c}, convergencia: {convergencia_c}")

# ----- Gráficas -----

# Gráfica de error de aproximación para el inciso (a)
plt.plot(errores_a)
plt.title("Error de Aproximación (a)")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.show()

# Gráfica de error de aproximación para el inciso (b)
plt.plot(errores_b)
plt.title("Error de Aproximación (b)")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.show()

# Gráfica de error de aproximación para el inciso (c)
plt.plot(errores_c)
plt.title("Error de Aproximación (c)")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.show()

# Gráfica de trayectoria en el plano para la función (a)
xx_a_np = np.array(xx_a)
plt.plot(xx_a_np[:, 0], xx_a_np[:, 1], marker='o')
plt.title("Trayectoria en (a)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Gráfica de trayectoria en el plano para la función (b)
xx_b_np = np.array(xx_b)
plt.plot(xx_b_np[:, 0], xx_b_np[:, 1], marker='o')
plt.title("Trayectoria en (b)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# NOTA: La función (c) está en 10 dimensiones, por lo que no es posible visualizar la trayectoria fácilmente.

# ----- Tabla de Resultados -----
def crear_tabla(xx, df, x_optimo, errores):
    primeras_3 = xx[:3]
    ultimas_3 = xx[-3:]
    aproximaciones = primeras_3 + ultimas_3
    errores_aprox = [np.linalg.norm(x - x_optimo) for x in aproximaciones]
    normas_gradiente = [np.linalg.norm(df(x)) for x in aproximaciones]
    iteraciones = list(range(1, 4)) + list(range(len(xx) - 2, len(xx) + 1))
    
    tabla = pd.DataFrame({
        'Iteración': iteraciones,
        'Aproximación': aproximaciones,
        'Error de Aproximación': errores_aprox,
        'Norma del Gradiente': normas_gradiente
    })
    
    return tabla

# Crear tablas para cada función
tabla_a = crear_tabla(xx_a, df_a, x_a, errores_a)
tabla_b = crear_tabla(xx_b, df_b, x_b, errores_b)
tabla_c = crear_tabla(xx_c, df_c, x_c, errores_c)

print("\nTabla de resultados (a):")
print(tabla_a)

print("\nTabla de resultados (b):")
print(tabla_b)

print("\nTabla de resultados (c):")
print(tabla_c)