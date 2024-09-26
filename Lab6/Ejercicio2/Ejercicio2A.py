import pulp

# Definir el problema
prob = pulp.LpProblem("Producción de Ventanas Acme", pulp.LpMinimize)

# Definir los periodos
periodos = range(1, 7)

# Definir la demanda en cada periodo
demanda = {1: 180, 2: 250, 3: 190, 4: 140, 5: 220, 6: 250}

# Definir el costo de producción en cada periodo
costo_produccion = {1: 50, 2: 45, 3: 55, 4: 52, 5: 48, 6: 50}

# Definir el costo de almacenamiento en cada periodo
costo_almacenamiento = {1: 8, 2: 10, 3: 10, 4: 10, 5: 8, 6: 8}

# Definir las variables de decisión
# x_ij: cantidad de ventanas producidas en el periodo i para satisfacer la demanda del periodo j
x = pulp.LpVariable.dicts("x", [(i, j) for i in periodos for j in periodos if i <= j], lowBound=0, cat='Integer')

# Función objetivo: minimizar el costo total de producción y almacenamiento
prob += pulp.lpSum(costo_produccion[i] * x[(i, j)] for i in periodos for j in periodos if i <= j) + \
        pulp.lpSum(costo_almacenamiento[i] * pulp.lpSum(x[(k, j)] for k in periodos if k <= i) for i in periodos for j in periodos if i < j)

# Restricciones
# 1. Satisfacer la demanda en cada periodo
for j in periodos:
    prob += pulp.lpSum(x[(i, j)] for i in periodos if i <= j) == demanda[j]

# 2. Capacidad máxima de producción en cada periodo
for i in periodos:
    prob += pulp.lpSum(x[(i, j)] for j in periodos if i <= j) <= 225

# Resolver el problema
prob.solve()

# Imprimir el estado de la solución
print("Estado:", pulp.LpStatus[prob.status])

# Imprimir los resultados
print("Costo total óptimo:", pulp.value(prob.objective))
print("Plan de producción óptimo:")
for i in periodos:
    for j in periodos:
        if i <= j and x[(i, j)].varValue > 0:
            print(f"Producir {x[(i, j)].varValue} ventanas en el periodo {i} para el periodo {j}")