from pulp import *

# Definir los costos
costos = [
    [3, 9, 2, 2, 7],
    [6, 1, 5, 6, 6],
    [9, 4, 7, 10, 3],
    [2, 5, 4, 2, 1],
    [9, 6, 2, 4, 6]
]

# Crear el problema de asignación
problema = LpProblem("Problema de Asignación", LpMinimize)

# Crear variables de decisión
trabajadores = range(len(costos))
tareas = range(len(costos[0]))
x = LpVariable.dicts("Asignación", (trabajadores, tareas), 0, 1, LpInteger)

# Función objetivo: minimizar el costo total
problema += lpSum(costos[i][j] * x[i][j] for i in trabajadores for j in tareas)

# Restricciones:
# Cada trabajador se asigna a una sola tarea
for i in trabajadores:
    problema += lpSum(x[i][j] for j in tareas) == 1

# Cada tarea se asigna a un solo trabajador
for j in tareas:
    problema += lpSum(x[i][j] for i in trabajadores) == 1

# Resolver el problema
problema.solve()

# Imprimir los resultados
print("Estado:", LpStatus[problema.status])
print("Costo total:", value(problema.objective))

for i in trabajadores:
    for j in tareas:
        if x[i][j].varValue == 1:
            print(f"Trabajador {i+1} asignado a la tarea {j+1}")