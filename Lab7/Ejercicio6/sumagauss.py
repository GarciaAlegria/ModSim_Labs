import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def descenso_newton_exacto(f, df, ddf, x0, alpha, maxIter, tol, criterio_paro):
  """
  Implementa el método de descenso de gradiente de Newton con Hessiano exacto.

  Args:
    f: Función objetivo.
    df: Gradiente de la función objetivo.
    ddf: Hessiano de la función objetivo.
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
    direccion = -np.linalg.solve(ddf(x), df(x))
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

def suma_gaussianas(x, puntos, a):
  """
  Calcula la suma de gaussianas en 2D.

  Args:
    x: Punto en el que se evalúa la función (array de tamaño 2).
    puntos: Puntos aleatorios (array de tamaño (k, 2)).
    a: Parámetro de escala.

  Returns:
    Valor de la función en el punto x.
  """
  k = puntos.shape[0]
  suma = 0
  for i in range(k):
    suma += np.exp(-(1/a**2) * np.linalg.norm(x - puntos[i])**2)
  return -suma  # Negativo para encontrar mínimos

def gradiente_suma_gaussianas(x, puntos, a):
  """
  Calcula el gradiente de la suma de gaussianas en 2D.

  Args:
    x: Punto en el que se evalúa el gradiente (array de tamaño 2).
    puntos: Puntos aleatorios (array de tamaño (k, 2)).
    a: Parámetro de escala.

  Returns:
    Gradiente de la función en el punto x (array de tamaño 2).
  """
  k = puntos.shape[0]
  gradiente = np.zeros(2)
  for i in range(k):
    gradiente += (2/a**2) * (x - puntos[i]) * np.exp(-(1/a**2) * np.linalg.norm(x - puntos[i])**2)
  return -gradiente  # Negativo para encontrar mínimos

def hessiano_suma_gaussianas(x, puntos, a):
  """
  Calcula el Hessiano de la suma de gaussianas en 2D.

  Args:
    x: Punto en el que se evalúa el Hessiano (array de tamaño 2).
    puntos: Puntos aleatorios (array de tamaño (k, 2)).
    a: Parámetro de escala.

  Returns:
    Hessiano de la función en el punto x (array de tamaño (2, 2)).
  """
  k = puntos.shape[0]
  hessiano = np.zeros((2, 2))
  for i in range(k):
    term1 = (2/a**2) * np.exp(-(1/a**2) * np.linalg.norm(x - puntos[i])**2)
    term2 = (4/a**4) * np.outer(x - puntos[i], x - puntos[i]) * np.exp(-(1/a**2) * np.linalg.norm(x - puntos[i])**2)
    hessiano += term1 * np.eye(2) - term2
  return -hessiano  # Negativo para encontrar mínimos

# Generar puntos aleatorios
np.random.seed(0)  # Para reproducibilidad
k = 8
puntos = np.random.rand(k, 2) * 0.8  # Puntos en [0, 0.8] x [0, 0.8]
a = 0.1  # Parámetro de escala

# Parámetros del método de descenso
x0 = np.array([0.1, 0.1])  # Punto inicial
alpha = 0.1
maxIter = 100
tol = 1e-6

# Criterio de paro (norma euclidiana de la diferencia)
criterio_paro = lambda x_viejo, x_nuevo: np.linalg.norm(x_nuevo - x_viejo)

# Ejecutar el método de descenso de gradiente de Newton
x, xx, fx, errores, num_iter, convergencia = descenso_newton_exacto(
    lambda x: suma_gaussianas(x, puntos, a),
    lambda x: gradiente_suma_gaussianas(x, puntos, a),
    lambda x: hessiano_suma_gaussianas(x, puntos, a),
    x0, alpha, maxIter, tol, criterio_paro
)

# Crear la malla para la gráfica de contorno
x1 = np.linspace(0, 0.8, 50)
x2 = np.linspace(0, 0.8, 50)
X1, X2 = np.meshgrid(x1, x2)
Z = np.zeros_like(X1)
for i in range(X1.shape[0]):
  for j in range(X1.shape[1]):
    Z[i, j] = suma_gaussianas(np.array([X1[i, j], X2[i, j]]), puntos, a)

# Gráfica de contorno
plt.figure(figsize=(8, 6))
CS = plt.contour(X1, X2, Z, 20)
plt.clabel(CS, inline=1, fontsize=10)

# Graficar la secuencia de iteraciones
plt.plot([x[0] for x in xx], [x[1] for x in xx], '-o', color='red', label='Iteraciones')
plt.plot(x[0], x[1], 'g*', markersize=10, label='Solución')

# Mostrar los puntos aleatorios
plt.scatter(puntos[:, 0], puntos[:, 1], color='black', marker='x', label='Puntos')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Descenso de gradiente de Newton - Suma de gaussianas')
plt.legend()
plt.show()