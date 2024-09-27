from pulp import *

# Crear el problema de minimización
prob = LpProblem("Asignacion_de_Autobuses", LpMinimize)

# Definir las variables (no negativas)
x1 = LpVariable("x1", 0)
x2 = LpVariable("x2", 0)
x3 = LpVariable("x3", 0)
x4 = LpVariable("x4", 0)
x5 = LpVariable("x5", 0)
x6 = LpVariable("x6", 0)

# Definir la función objetivo: minimizar el número total de autobuses
prob += x1 + x2 + x3 + x4 + x5 + x6, "Total_de_autobuses"

# Definir las restricciones de demanda
prob += x1 + x6 >= 4, "Demanda_intervalo_1"
prob += x1 + x2 >= 8, "Demanda_intervalo_2"
prob += x2 + x3 >= 10, "Demanda_intervalo_3"
prob += x3 + x4 >= 7, "Demanda_intervalo_4"
prob += x4 + x5 >= 12, "Demanda_intervalo_5"
prob += x5 + x6 >= 4, "Demanda_intervalo_6"

# Resolver el problema
prob.solve()

# Imprimir el estado de la solución
print("Estado:", LpStatus[prob.status])

# Imprimir los valores de las variables
for v in prob.variables():
    print(v.name, "=", v.varValue)

# Imprimir el valor óptimo de la función objetivo
print("Valor óptimo de la función objetivo:", value(prob.objective))
