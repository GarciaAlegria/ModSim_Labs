import numpy as np
import random
import math

# Función para leer el archivo ch150.tsp
def read_tsp_file(filename):
    cities = [] # Lista de ciudades (id, x, y) 
    with open(filename, 'r') as file: # Abrir el archivo en modo lectura
        lines = file.readlines() # Leer todas las líneas del archivo
        coord_section = False # Variable para indicar si se está en la sección de coordenadas
        for line in lines: # Iterar sobre las líneas del archivo
            if "NODE_COORD_SECTION" in line: # Si se encuentra la sección de coordenadas
                coord_section = True # Cambiar la variable a True
                continue # Continuar con la siguiente iteración
            if "EOF" in line: # Si se encuentra el final del archivo
                break
            if coord_section: # Si se está en la sección de coordenadas
                parts = line.split() # Separar la línea por espacios
                city_id = int(parts[0]) # Obtener el id de la ciudad
                x = float(parts[1]) # Obtener la coordenada x
                y = float(parts[2]) # Obtener la coordenada y
                cities.append((city_id, x, y)) # Agregar la ciudad a la lista
    return cities # Retornar la lista de ciudades

# Función de distancia euclidiana entre dos ciudades
def euclidean_distance(city1, city2): # Recibe dos ciudades (id, x, y)
    return math.sqrt((city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2) # Calcular la distancia euclidiana

# Función para calcular la distancia total de un recorrido
def total_distance(tour, cities):
    distance = 0 # Inicializar la distancia en 0
    for i in range(len(tour)): # Iterar sobre el recorrido
        city1 = cities[tour[i]] # Obtener la ciudad actual
        city2 = cities[tour[(i + 1) % len(tour)]] # Obtener la siguiente ciudad
        distance += euclidean_distance(city1, city2) # Sumar la distancia entre las dos ciudades
    return distance

# Función de fitness (inversa de la distancia total)
def fitness(tour, cities): # Recibe un recorrido y la lista de ciudades
    return 1 / total_distance(tour, cities) # Retornar la inversa de la distancia total

# Crear un recorrido aleatorio
def create_random_tour(num_cities): # Recibe el número de ciudades
    tour = list(range(num_cities)) # Crear una lista con los índices de las ciudades
    random.shuffle(tour) # Mezclar aleatoriamente los índices
    return tour # Retornar el recorrido aleatorio

# Crear la población inicial
def create_initial_population(population_size, num_cities): # Recibe el tamaño de la población y el número de ciudades
    return [create_random_tour(num_cities) for _ in range(population_size)] # Crear una lista de recorridos aleatorios

# Selección por torneo
def tournament_selection(population, cities, k=3): # Recibe la población, la lista de ciudades y el tamaño del torneo
    selected = random.sample(population, k) # Seleccionar k individuos aleatorios
    selected_fitness = [fitness(tour, cities) for tour in selected] # Calcular el fitness de los individuos seleccionados
    return selected[np.argmax(selected_fitness)] # Retornar el individuo con el mejor fitness

# Cruce de orden (Order Crossover, OX)
def order_crossover(parent1, parent2): # Recibe dos padres
    size = len(parent1) # Obtener el tamaño de los padres
    start, end = sorted(random.sample(range(size), 2)) # Seleccionar dos puntos de corte aleatorios
    child = [None] * size # Crear un hijo con el mismo tamaño que los padres
    child[start:end] = parent1[start:end] # Copiar la sección de los padres al hijo

    pointer = end # Inicializar el puntero en el punto
    for gene in parent2: # Iterar sobre los genes del segundo padre
        if gene not in child: # Si el gen no está en el hijo
            if pointer >= size: # Si el puntero llega al final
                pointer = 0 # Reiniciar el puntero
            child[pointer] = gene # Agregar el gen al hijo
            pointer += 1 # Mover el puntero al siguiente índice
    return child # Retornar el hijo

def inversion_mutation(tour): # Recibe un recorrido (padre) para mutar y retornar un recorrido (hijo)
    i, j = sorted(random.sample(range(len(tour)), 2)) # Seleccionar dos índices aleatorios
    tour[i:j] = reversed(tour[i:j]) # Invertir la sección del recorrido
    return tour # Retornar el recorrido mutado


# Mutación por intercambio (Swap Mutation)
def swap_mutation(tour): # Recibe un recorrido (padre) para mutar y retornar un recorrido (hijo)
    i, j = random.sample(range(len(tour)), 2) # Seleccionar dos índices aleatorios
    tour[i], tour[j] = tour[j], tour[i] # Intercambiar los genes en los índices seleccionados
    return tour  # Retornar el recorrido mutado

def two_opt(tour, cities): # Recibe un recorrido y la lista de ciudades
    best_distance = total_distance(tour, cities) # Calcular la distancia del recorrido actual
    best_tour = tour[:] # Copiar el recorrido actual
    for i in range(1, len(tour) - 1): # Iterar sobre los índices del recorrido
        for j in range(i + 1, len(tour)): # Iterar sobre los índices del recorrido
            new_tour = tour[:i] + tour[i:j][::-1] + tour[j:] # Aplicar la operación 2-opt
            new_distance = total_distance(new_tour, cities) # Calcular la distancia del nuevo recorrido
            if new_distance < best_distance: # Si la nueva distancia es mejor
                best_distance = new_distance # Actualizar la mejor distancia
                best_tour = new_tour # Actualizar el mejor recorrido
    return best_tour # Retornar el mejor recorrido


# Algoritmo Genético Mejorado
def genetic_algorithm(cities, population_size=500, generations=1000, mutation_rate=0.05, elite_size=5): # Recibe la lista de ciudades, el tamaño de la población, el número de generaciones, la tasa de mutación y el tamaño de la élite
    population = create_initial_population(population_size, len(cities)) # Crear la población inicial
    
    for generation in range(generations): # Iterar sobre las generaciones
        new_population = [] # Crear una nueva población
        
        # Añadir la élite directamente a la nueva población
        sorted_population = sorted(population, key=lambda t: total_distance(t, cities)) # Ordenar la población por distancia
        new_population.extend(sorted_population[:elite_size]) # Añadir los mejores individuos a la nueva población
        
        while len(new_population) < population_size: # Mientras la nueva población no alcance el tamaño deseado
            parent1 = tournament_selection(population, cities) # Seleccionar el primer padre
            parent2 = tournament_selection(population, cities) # Seleccionar el segundo padre
            
            child1 = order_crossover(parent1, parent2) # Aplicar el cruce de orden
            child2 = order_crossover(parent2, parent1) # Aplicar el cruce de orden
            
            if random.random() < mutation_rate: # Si se cumple la tasa de mutación
                child1 = inversion_mutation(child1)  # Usando la mutación de inversión
            if random.random() < mutation_rate: # Si se cumple la tasa de mutación
                child2 = swap_mutation(child2)  # Alternando entre mutación por intercambio
            
            new_population.append(child1) # Añadir el primer hijo a la nueva población
            new_population.append(child2) # Añadir el segundo hijo a la nueva población
        
        population = new_population[:population_size] # Actualizar la población con los mejores individuos
        
        # Aplicar búsqueda local 2-opt en los mejores individuos
        best_tour = min(population, key=lambda t: total_distance(t, cities))
        best_tour = two_opt(best_tour, cities)  # Aplicar 2-opt para refinar
        best_distance = total_distance(best_tour, cities) # Calcular la distancia del mejor recorrido
        
        print(f"Generación {generation+1}, Mejor distancia: {best_distance:.2f}") # Mostrar la generación y la mejor distancia
    
    return best_tour, best_distance # Retornar el mejor recorrido y su distancia



# Ejecutar el algoritmo genético
if __name__ == "__main__":
    # Cargar las ciudades desde el archivo
    cities = read_tsp_file('ch150.tsp')
    
    # Ejecutar el algoritmo genético
    best_tour, best_distance = genetic_algorithm(cities)
    
    print("Mejor recorrido encontrado:", best_tour)
    print("Distancia del mejor recorrido:", best_distance)
