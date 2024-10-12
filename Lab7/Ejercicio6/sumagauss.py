import numpy as np
import matplotlib.pyplot as plt

def suma_gaussianas(x, puntos, sigma):
    k = puntos.shape[0]
    suma = 0
    for i in range(k):
        suma += np.exp(-np.linalg.norm(x - puntos[i])**2 / (2 * sigma**2))
    return suma

def gradiente_suma_gaussianas(x, puntos, sigma):
    k = puntos.shape[0]
    gradiente = np.zeros(2)
    for i in range(k):
        gradiente += -(x - puntos[i]) * np.exp(-np.linalg.norm(x - puntos[i])**2 / (2 * sigma**2)) / sigma**2
    return gradiente

def descenso_gradiente(x0, puntos, sigma, alpha=0.1, max_iter=100, tol=1e-6):
    x = x0
    trayectoria = [x]
    for _ in range(max_iter):
        x_prev = x
        x = x - alpha * gradiente_suma_gaussianas(x, puntos, sigma)
        trayectoria.append(x)
        if np.linalg.norm(x - x_prev) < tol:
            break
    return trayectoria

# Parámetros
k = 8
sigma = 1  # Puedes ajustar este valor
puntos = np.random.rand(k, 2) * np.array([8, 8])  # Puntos en el rectángulo [0, 8] x [0, 8]

# Visualización de la función
x = np.linspace(0, 8, 100)
y = np.linspace(0, 8, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = suma_gaussianas(np.array([X[i, j], Y[i, j]]), puntos, sigma)

# Diccionario para almacenar las trayectorias
trayectorias_minimos = {}

# Optimización con diferentes puntos iniciales
num_inicializaciones = 50  # Aumentamos el número de inicializaciones
for _ in range(num_inicializaciones):
    x0 = np.random.rand(2) * np.array([8, 8])
    trayectoria = descenso_gradiente(x0, puntos, sigma)
    minimo_local = tuple(trayectoria[-1].round(2))  # Redondeamos para usar como clave
    if minimo_local not in trayectorias_minimos:
        trayectorias_minimos[minimo_local] = []
    trayectorias_minimos[minimo_local].append(trayectoria)

# Seleccionar aleatoriamente 10 trayectorias para visualizar
import random
trayectorias_seleccionadas = random.sample(list(trayectorias_minimos.items()), min(10, len(trayectorias_minimos)))

# Crear gráficos separados para cada mínimo local seleccionado
for minimo_local, trayectorias in trayectorias_seleccionadas:
    plt.figure()  # Crea una nueva figura para cada mínimo local
    plt.contour(X, Y, Z, 20)
    for trayectoria in trayectorias[:5]:  # Mostrar solo las primeras 5 trayectorias
        plt.plot([p[0] for p in trayectoria], [p[1] for p in trayectoria], '-x', label='Iterates')
        plt.plot(trayectoria[-1][0], trayectoria[-1][1], 'g*', markersize=10, label='Solution')
    plt.title(f'Mínimo local en {minimo_local}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()