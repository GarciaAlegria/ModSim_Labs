import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros (ajustados según la imagen)
N, M = 50, 50  # Tamaño de la grilla
beta = 0.1     # Probabilidad de contagio
gamma = 0.25   # Probabilidad de recuperación
r = 1          # Radio de interacción
tiempo_total = 100
I0 = 2          # Número inicial de infectados

# Número de experimentos
Nexp = 10

# Generar posiciones iniciales aleatorias (una sola vez)
posiciones_iniciales = [(np.random.randint(N), np.random.randint(M)) for _ in range(I0)]

# Lista para almacenar los historiales de todos los experimentos
historiales_grillas = []

# Función para actualizar la grilla en cada paso de tiempo
def actualizar_grilla(grilla):
    nueva_grilla = grilla.copy()
    for i in range(N):
        for j in range(M):
            if grilla[i, j] == 1:  # Infectado
                for x in range(max(0, i-r), min(N, i+r+1)):
                    for y in range(max(0, j-r), min(M, j+r+1)):
                        if grilla[x, y] == 0:  # Susceptible
                            if np.random.rand() < beta:
                                nueva_grilla[x, y] = 1
                if np.random.rand() < gamma:
                    nueva_grilla[i, j] = 2  # Recuperado
    return nueva_grilla

# Bucle para realizar múltiples experimentos
for _ in range(Nexp):
    # Inicializar la grilla
    grilla = np.zeros((N, M))
    for i, j in posiciones_iniciales:
        grilla[i, j] = 1

    # Listas para almacenar el historial de S, I y R de este experimento
    S, I, R = [np.sum(grilla == 0)], [np.sum(grilla == 1)], [np.sum(grilla == 2)]

    # Simulación (sin animación en este caso, solo cálculo)
    for _ in range(tiempo_total):
        grilla = actualizar_grilla(grilla)
        S.append(np.sum(grilla == 0))
        I.append(np.sum(grilla == 1))
        R.append(np.sum(grilla == 2))

    # Almacenar el historial de este experimento
    historiales_grillas.append(np.array([S, I, R]))

# Convertir la lista de historiales en un array NumPy
historiales_grillas = np.array(historiales_grillas)

# Calcular el promedio a lo largo del eje de los experimentos
promedio_grillas = np.mean(historiales_grillas, axis=0)

# Graficar el promedio de S, I y R
plt.figure()
plt.plot(promedio_grillas[0], label='Susceptibles (Promedio)')
plt.plot(promedio_grillas[1], label='Infectados (Promedio)')
plt.plot(promedio_grillas[2], label='Recuperados (Promedio)')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Dinámica del Modelo SIR (Promedio de {} Experimentos)'.format(Nexp))
plt.show()