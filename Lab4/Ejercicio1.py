import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros (ajustados según la imagen)
N, M = 50, 50  # Tamaño de la grilla
beta = 0.1     # Probabilidad de contagio
gamma = 0.25   # Probabilidad de recuperación
r = 1          # Radio de interacción
tiempo_total = 100
I0 = 2         # Número inicial de infectados

# Inicializar la grilla
grilla = np.zeros((N, M))
infectados_iniciales = np.random.choice(N*M, I0, replace=False)
grilla.ravel()[infectados_iniciales] = 1 

# Listas para almacenar el historial de S, I y R
S, I, R = [np.sum(grilla == 0)], [np.sum(grilla == 1)], [np.sum(grilla == 2)]

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

# Simulación y animación
fig, ax = plt.subplots()
imagen = ax.imshow(grilla, cmap='viridis', animated=True)

def actualizar(frame):
    global grilla
    grilla = actualizar_grilla(grilla)
    imagen.set_array(grilla)
    S.append(np.sum(grilla == 0))
    I.append(np.sum(grilla == 1))
    R.append(np.sum(grilla == 2))
    return imagen,

animacion = FuncAnimation(fig, actualizar, frames=tiempo_total, interval=200, blit=True)
plt.show()

# Graficar S, I y R
plt.figure()
plt.plot(S, label='Susceptibles')
plt.plot(I, label='Infectados')
plt.plot(R, label='Recuperados')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Dinámica del Modelo SIR')
plt.show()
