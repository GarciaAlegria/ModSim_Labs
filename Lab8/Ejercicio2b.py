import random
from deap import base, creator, tools, algorithms

# Definir los datos del problema (de nuevo)
valores = [10, 12, 8, 5, 8, 5, 6, 7, 6, 12, 8, 8, 10, 9, 8, 3, 7, 8, 5, 6]
pesos = [6, 7, 7, 3, 5, 2, 4, 5, 3, 9, 8, 7, 8, 6, 5, 2, 3, 5, 4, 6]
K = 50

# Definir el problema para maximizar la recompensa total
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Crear un individuo aleatorio (lista binaria)
def crear_individuo():
    return [random.randint(0, 1) for _ in range(len(valores))]

# Definir la función de evaluación
def evaluar_individuo(individuo):
    valor_total = sum(valores[i] * individuo[i] for i in range(len(individuo)))
    peso_total = sum(pesos[i] * individuo[i] for i in range(len(individuo)))
    if peso_total > K:  # Penalización si se excede el peso máximo
        return 0,
    return valor_total,

# Inicializar la herramienta base de DEAP
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, crear_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluar_individuo)

# Algoritmo genético
def algoritmo_genetico():
    random.seed(42)

    # Crear la población inicial
    poblacion = toolbox.population(n=100)

    # Aplicar el algoritmo evolutivo
    resultado = algorithms.eaSimple(poblacion, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=False)

    # Obtener el mejor individuo
    mejor_individuo = tools.selBest(poblacion, k=1)[0]
    
    valor_total = sum(valores[i] * mejor_individuo[i] for i in range(len(mejor_individuo)))
    peso_total = sum(pesos[i] * mejor_individuo[i] for i in range(len(mejor_individuo)))
    
    print("Selección óptima con algoritmo genético:")
    for i in range(len(mejor_individuo)):
        if mejor_individuo[i] == 1:
            print(f"Item {i+1} - Valor: {valores[i]}, Peso: {pesos[i]}")
    
    print(f"Valor total: {valor_total}")
    print(f"Peso total: {peso_total}")

# Ejecutar el algoritmo genético
algoritmo_genetico()
