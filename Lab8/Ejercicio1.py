import random

# Define the fitness function
def fitness(individuo, v, w, K):
    valor = sum(v[i] for i in range(len(individuo)) if individuo[i] == 1)
    peso = sum(w[i] for i in range(len(individuo)) if individuo[i] == 1)
    if peso > K:
        return 0  # Penalización si el peso excede la capacidad
    return valor

# Define the function to generate the initial population
def generar_poblacion(N, length):
    return [[random.randint(0, 1) for _ in range(length)] for _ in range(N)]

# Define the function to select parents
def seleccionar_padres(poblacion, fitness_valores, s):
    if not fitness_valores:
        raise ValueError("fitness_valores is empty")
    if len(fitness_valores) != len(poblacion):
        raise ValueError("Length of fitness_valores does not match length of poblacion")
    padres = random.choices(poblacion, weights=fitness_valores, k=int(len(poblacion) * s))
    return padres

# Define the function to perform crossover
def cruzar(padre1, padre2):
    punto = random.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2

# Define the function to mutate an individual
def mutar(individuo, m):
    for i in range(len(individuo)):
        if random.random() < m:
            individuo[i] = 1 - individuo[i]  # Cambia el bit
    return individuo

# Parámetros del algoritmo
N = 50  # Tamaño de la población
s = 0.5  # Porcentaje de selección
c = 0.8  # Porcentaje de cruce
m = 0.1  # Porcentaje de mutación
maxI = 100  # Número máximo de iteraciones

# Datos del problema
v = [10, 5, 15, 7, 6]
w = [2, 3, 5, 7, 1]
K = 10

# Algoritmo genético
poblacion = generar_poblacion(N, len(v))
for _ in range(maxI):
    fitness_valores = [fitness(s, v, w, K) for s in poblacion]
    if not fitness_valores:
        print("Error: fitness_valores is empty")
        break
    padres = seleccionar_padres(poblacion, fitness_valores, s)
    nueva_poblacion = []
    for i in range(0, len(padres) - 1, 2):
        hijo1, hijo2 = cruzar(padres[i], padres[i + 1])
        nueva_poblacion.extend([mutar(hijo1, m), mutar(hijo2, m)])
    if len(nueva_poblacion) < N:
        # Rellenar la población si es necesario
        while len(nueva_poblacion) < N:
            nueva_poblacion.append(random.choice(poblacion))
    poblacion = nueva_poblacion

poblacion = nueva_poblacion

# Verificar que la población no esté vacía antes de encontrar la mejor solución
if poblacion:
    # Imprimir los valores de fitness de toda la población
    print("Valores de fitness de la población:")
    for idx, solucion in enumerate(poblacion):
        fitness_valor = fitness(solucion, v, w, K)
        print(f"Solución {idx + 1}: {solucion}, Fitness: {fitness_valor}")

    mejor_solucion = max(poblacion, key=lambda s: fitness(s, v, w, K))
    mejor_fitness = fitness(mejor_solucion, v, w, K)

    print("\nMejor solución:", mejor_solucion)
    print("Valor total:", mejor_fitness)
else:
    print("Error: La población está vacía.")