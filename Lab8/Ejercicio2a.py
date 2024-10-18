import pulp

# Definir los datos del problema
valores = [10, 12, 8, 5, 8, 5, 6, 7, 6, 12, 8, 8, 10, 9, 8, 3, 7, 8, 5, 6]
pesos = [6, 7, 7, 3, 5, 2, 4, 5, 3, 9, 8, 7, 8, 6, 5, 2, 3, 5, 4, 6]
K = 50  # Capacidad máxima de la mochila

# Crear el problema de maximización
problema = pulp.LpProblem("Problema_de_la_Mochila", pulp.LpMaximize)

# Crear variables binarias para cada ítem
x = [pulp.LpVariable(f'x{i}', cat='Binary') for i in range(len(valores))]

# Función objetivo: maximizar el valor total de los ítems seleccionados
problema += pulp.lpSum(valores[i] * x[i] for i in range(len(valores))), "Valor total"

# Restricción: el peso total no debe exceder la capacidad de la mochila
problema += pulp.lpSum(pesos[i] * x[i] for i in range(len(pesos))) <= K, "Restriccion de capacidad"

# Resolver el problema
problema.solve()

# Mostrar los resultados
print("Estado:", pulp.LpStatus[problema.status])
print("Selección óptima:")
for i in range(len(valores)):
    if x[i].varValue == 1:
        print(f"Item {i+1} - Valor: {valores[i]}, Peso: {pesos[i]}")

valor_total = sum(valores[i] for i in range(len(valores)) if x[i].varValue == 1)
peso_total = sum(pesos[i] for i in range(len(pesos)) if x[i].varValue == 1)

print(f"Valor total: {valor_total}")
print(f"Peso total: {peso_total}")
