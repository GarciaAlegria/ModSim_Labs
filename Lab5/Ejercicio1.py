import numpy as np
import matplotlib.pyplot as plt

# Parámetros de entrada (ajusta según tus necesidades)
M = 60  # Número de filas del grid
N = 60  # Número de columnas del grid
T = 100  # Límite temporal
K = 0.25  # Parámetro de difusión

# Distribución inicial de probabilidad (ajusta según la Figura 1)
u_inicial = np.zeros((M, N))
u_inicial[1, 2] = 1
u_inicial[2, 1] = 1
u_inicial[2, 3] = 1
u_inicial[3, 2] = 1

# Definir la región R (ajusta según la Figura 1)
R = set([(1, 2), (2, 1), (2, 3), (3, 2)])

# Tipo de vecindad (Von Neumann)
vecindad = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Crear el grid y el historial
grid = u_inicial.copy()
historial = [grid.copy()]

# Simulación
for t in range(T):
    nuevo_grid = grid.copy()
    for i in range(M):
        for j in range(N):
            if (i, j) in R:
                suma_vecinos = 0
                for vecino in vecindad:
                    ni, nj = i + vecino[0], j + vecino[1]
                    if 0 <= ni < M and 0 <= nj < N and (ni, nj) in R:
                        suma_vecinos += grid[ni][nj]
                nuevo_grid[i][j] = (1 - K) * grid[i][j] + (K / 4) * suma_vecinos  # Usamos 4 vecinos en Von Neumann

    grid = nuevo_grid
    historial.append(grid.copy())

# Visualización (mostrar algunas imágenes clave)
tiempos_a_mostrar = [0, 1, 25, 50, 100]  # Ajusta según tus necesidades
for t in tiempos_a_mostrar:
    plt.imshow(historial[t], cmap='hot', interpolation='nearest')
    plt.title(f'Tiempo t = {t}')
    plt.colorbar()
    plt.show()