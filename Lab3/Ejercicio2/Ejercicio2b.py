import numpy as np
from scipy.optimize import fsolve

# Definir la función que describe el sistema de EDO
def sistema(puntos):
    x, y = puntos
    return [0.2 * x - 0.005 * x * y, -0.5 * y + 0.01 * x * y]

# Encontrar los puntos de equilibrio utilizando fsolve
puntos_iniciales = [1, 1]
equilibrio = fsolve(sistema, puntos_iniciales)
print(f"Punto de equilibrio: {equilibrio}")

# Definir la función que calcula la matriz Jacobiana
def jacobiano(punto):
    x, y = punto
    return np.array([[0.2 - 0.005 * y, -0.005 * x],
                     [0.01 * y, -0.5 + 0.01 * x]])

# Calcular la matriz Jacobiana en el punto de equilibrio
J = jacobiano(equilibrio)
eigenvalores = np.linalg.eigvals(J)
print(f"Valores propios de la matriz Jacobiana: {eigenvalores}")

# Clasificación del punto de equilibrio
if np.all(np.real(eigenvalores) < 0):
    print("El punto de equilibrio es un nodo atractivo (estable).")
elif np.all(np.real(eigenvalores) > 0):
    print("El punto de equilibrio es un nodo repulsivo (inestable).")
else:
    if np.all(np.iscomplex(eigenvalores)):
        print("El punto de equilibrio es un foco.")
    else:
        print("El punto de equilibrio es una silla.")
