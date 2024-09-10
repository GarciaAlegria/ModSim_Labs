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

def custom_initial_distribution(M, N, P, mode="random"):
    """ Define una distribución inicial específica de partículas """
    particles = []
    
    if mode == "random":
        particles = [(random.randint(0, M-1), random.randint(0, N-1)) for _ in range(P)]
    
    elif mode == "center":
        # Coloca las partículas cerca del centro del grid
        center_x, center_y = M // 2, N // 2
        particles = [(random.randint(center_x - 5, center_x + 5), 
                      random.randint(center_y - 5, center_y + 5)) for _ in range(P)]
    
    elif mode == "quadrant":
        # Coloca las partículas en el primer cuadrante
        particles = [(random.randint(0, M//2 - 1), random.randint(0, N//2 - 1)) for _ in range(P)]
    
    return particles

def particle_diffusion_custom(M, N, T, P, K, Nexp, neigh, initial_mode):
    history = np.zeros((M, N, T))  # Mantener historial de cada repetición
    
    for exp in range(Nexp):
        grid = np.zeros((M, N))  # Grid para una simulación
        particles = custom_initial_distribution(M, N, P, initial_mode)
        
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

def compare_diffusion_for_K(M, N, T, P, Nexp, neigh, K_values, initial_mode):
    """ Compara la difusión para diferentes valores de K """
    fig, axs = plt.subplots(1, len(K_values), figsize=(15, 5))
    
    for i, K in enumerate(K_values):
        history = particle_diffusion_custom(M, N, T, P, K, Nexp, neigh, initial_mode)
        norm_grid = normalize_grid(history[:, :, T-1])  # Estado final
        
        axs[i].imshow(norm_grid, cmap='hot', interpolation='nearest')
        axs[i].set_title(f'K = {K}')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    plt.show()

def compare_diffusion_for_initial_conditions(M, N, T, P, K, Nexp, neigh, initial_modes):
    """ Compara la difusión para diferentes distribuciones iniciales """
    fig, axs = plt.subplots(1, len(initial_modes), figsize=(15, 5))
    
    for i, mode in enumerate(initial_modes):
        history = particle_diffusion_custom(M, N, T, P, K, Nexp, neigh, mode)
        norm_grid = normalize_grid(history[:, :, T-1])  # Estado final
        
        axs[i].imshow(norm_grid, cmap='hot', interpolation='nearest')
        axs[i].set_title(f'Initial: {mode}')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    plt.show()

# Parámetros iniciales
M, N = 50, 50
T = 100
P = 50
Nexp = 5
neigh = 8  # Vecindad de 8 vecinos

# Parte (a) - Variar K para la misma distribución inicial (aleatoria)
K_values = [0.1, 0.5, 0.9]  # Diferentes probabilidades de movimiento
compare_diffusion_for_K(M, N, T, P, Nexp, neigh, K_values, initial_mode="random")

# Parte (b) - Variar la distribución inicial con un mismo K
initial_modes = ["random", "center", "quadrant"]
K_fixed = 0.5  # Mantener K constante
compare_diffusion_for_initial_conditions(M, N, T, P, K_fixed, Nexp, neigh, initial_modes)
