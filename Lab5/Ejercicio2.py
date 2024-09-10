import numpy as np
import matplotlib.pyplot as plt
import random

def get_particle_neighbors(x, y, M, N, neigh):
    """ Devuelve los vecinos posibles para una partícula. """
    if neigh == 4:  # Vecindad de 4 (arriba, abajo, izquierda, derecha)
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    elif neigh == 8:  # Vecindad de 8 (incluye diagonales)
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1), 
                (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]

def particle_diffusion(M, N, T, P, K, Nexp, neigh):
    history = np.zeros((M, N, T))  # Mantener historial de cada repetición
    
    for exp in range(Nexp):
        grid = np.zeros((M, N))  # Grid para una simulación
        particles = [(random.randint(0, M-1), random.randint(0, N-1)) for _ in range(P)]
        
        for t in range(T):
            new_particles = []
            for (x, y) in particles:
                if random.random() < K:
                    neighbors = get_particle_neighbors(x, y, M, N, neigh)
                    move = random.choice(neighbors)
                    x_new = move[0] % M
                    y_new = move[1] % N
                else:
                    x_new, y_new = x, y
                
                new_particles.append((x_new, y_new))
                grid[x_new, y_new] += 1
            
            particles = new_particles
            history[:, :, t] += grid  # Acumular resultados en la historia
    
    return history / Nexp  # Promedio de las simulaciones

def normalize_grid(grid):
    """ Normaliza el grid dividiendo por la suma total de partículas. """
    total = np.sum(grid)
    if total > 0:
        return grid / total
    return grid

def plot_average_diffusion(history, T):
    """ Graficar el promedio espacial en diferentes instantes de tiempo. """
    for t in range(0, T, 25):
        norm_grid = normalize_grid(history[:, :, t])
        plt.imshow(norm_grid, cmap='hot', interpolation='nearest')
        plt.title(f'Time {t}')
        plt.colorbar()
        plt.show()

# Parámetros iniciales
M, N = 50, 50
T = 100
P = 50
K = 0.2
Nexp = 5 # Número de experimentos
neigh = 8  # Vecindad de 8 vecinos

# Simulación de difusión usando partículas con promedio
history = particle_diffusion(M, N, T, P, K, Nexp, neigh)

# Graficar el promedio espacial
plot_average_diffusion(history, T)