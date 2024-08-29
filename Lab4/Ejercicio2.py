import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros base
N, M = 50, 50  # Tamaño de la grilla
tiempo_total = 100
I0 = 2  # Número inicial de infectados

# Número de repeticiones de la simulación
N_exp = 10

# Valores de parámetros a probar
betas = [0.05, 0.1, 0.2]
gammas = [0.1, 0.25, 0.5]
radios = [1, 2, 3]

# Almacenar resultados para diferentes parámetros
resultados = {}

for beta in betas:
    for gamma in gammas:
        for r in radios:
            # Almacenar resultados de todas las simulaciones para esta combinación de parámetros
            all_S, all_I, all_R = [], [], []

            for _ in range(N_exp):
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
                            if grilla[i, j] == 1 and np.random.rand() < gamma:
                                nueva_grilla[i, j] = 2  # Recuperado
                    return nueva_grilla

                # Simulación (sin animación en este bucle)
                for _ in range(tiempo_total):
                    grilla = actualizar_grilla(grilla)
                    S.append(np.sum(grilla == 0))
                    I.append(np.sum(grilla == 1))
                    R.append(np.sum(grilla == 2))

                all_S.append(S)
                all_I.append(I)
                all_R.append(R)

            # Calcular promedios
            avg_S = np.mean(all_S, axis=0)
            avg_I = np.mean(all_I, axis=0)
            avg_R = np.mean(all_R, axis=0)

            resultados[(beta, gamma, r)] = (avg_S, avg_I, avg_R)

# Graficar resultados para diferentes parámetros
plt.figure()
for (beta, gamma, r), (S, I, R) in resultados.items():
    plt.plot(I, label=f'Infectados (β={beta}, γ={gamma}, r={r})')
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Influencia de β, γ y r en la dinámica de infectados')
plt.show()