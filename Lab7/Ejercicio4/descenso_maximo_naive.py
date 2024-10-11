import numpy as np

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

# Ejemplo de uso
def f(x):
  return x[0]**2 + x[1]**2

def df(x):
  return np.array([2*x[0], 2*x[1]])

def criterio_paro(x_viejo, x_nuevo):
  return np.linalg.norm(x_nuevo - x_viejo)

x0 = np.array([1, 1])
alpha = 0.1
maxIter = 100
tol = 1e-6

x, xx, fx, errores, num_iter, convergencia = descenso_maximo_naive(f, df, x0, alpha, maxIter, tol, criterio_paro)

print("Descenso máximo naive:")
print("Solución encontrada:", x)
print("Secuencia de iteraciones:", xx)
print("Secuencia de valores de f(x):", fx)
print("Secuencia de errores:", errores)
print("Número de iteraciones:", num_iter)
print("Convergencia:", convergencia)