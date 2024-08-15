import numpy as np
from scipy.optimize import fsolve

# Definir la función que describe el sistema de EDO (adaptada de la imagen)
def sistema(puntos):
    x, y = puntos
    return [0.5*x - 0.001*x**2 - x*y, -0.2*y + 0.1*x*y]

# Encontrar los puntos de equilibrio utilizando fsolve
puntos_iniciales = [1, 1]  # Puedes ajustar estos valores iniciales si es necesario
equilibrio = fsolve(sistema, puntos_iniciales)
print(f"Punto de equilibrio: {equilibrio}")

# Definir la función que calcula la matriz Jacobiana (adaptada de la imagen)
def jacobiano(punto):
    x, y = punto
    return np.array([[0.5 - 0.002*x - y, -x],
                     [0.1*y, -0.2 + 0.1*x]])

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