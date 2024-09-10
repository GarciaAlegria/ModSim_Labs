import numpy as np
import matplotlib.pyplot as plt

def get_neighbors(i, j, M, N, neigh):
    """ Devuelve la lista de vecinos según el tipo de vecindad. """
    if neigh == 4:  # Vecindad de 4 (arriba, abajo, izquierda, derecha)
        return [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
    elif neigh == 8:  # Vecindad de 8 (incluye diagonales)
        return [(i+1, j), (i-1, j), (i, j+1), (i, j-1), 
                (i+1, j+1), (i-1, j-1), (i+1, j-1), (i-1, j+1)]

def diffusion_simulation(M, N, T, u0, K, neigh):
    # Inicialización de la simulación
    u = np.copy(u0)
    history = [u0]
    
    for t in range(T):
        u_new = np.copy(u)
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                neighbors = get_neighbors(i, j, M, N, neigh)
                total = sum(u[ni, nj] for ni, nj in neighbors if 0 <= ni < M and 0 <= nj < N)
                u_new[i, j] = (1 - K) * u[i, j] + (K / len(neighbors)) * total
        
        u = np.copy(u_new)
        history.append(u)
    
    return history

def plot_diffusion(history, T):
    fig, ax = plt.subplots(1, T//25 + 1, figsize=(15, 5))
    
    for t in range(0, T+1, 25):
        ax[t//25].imshow(history[t], cmap='hot', interpolation='nearest')
        ax[t//25].set_title(f'Time {t}')
    
    plt.show()

# Parámetros iniciales
M, N = 50, 50
T = 100
K = 0.2
neigh = 8  # Vecindad de 8 vecinos
u0 = np.zeros((M, N))
u0[M//2, N//2] = 1  # Concentración inicial

history = diffusion_simulation(M, N, T, u0, K, neigh)
plot_diffusion(history, T)
