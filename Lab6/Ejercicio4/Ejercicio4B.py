import pulp

# Crear el problema de maximización
modelo = pulp.LpProblem("RenovacionUrbana", pulp.LpMaximize)

# Variables de decisión
x1 = pulp.LpVariable("x1", lowBound=0)  # Casas unifamiliares
x2 = pulp.LpVariable("x2", lowBound=0)  # Casas dobles
x3 = pulp.LpVariable("x3", lowBound=0)  # Casas triples
x4 = pulp.LpVariable("x4", lowBound=0)  # Casas cuádruples

# Función objetivo
modelo += 1000*x1 + 1900*x2 + 2700*x3 + 3400*x4, "Recaudacion de Impuestos"

# Restricciones
modelo += 0.85 * (300 * 0.25) >= 0.18*x1 + 0.28*x2 + 0.4*x3 + 0.5*x4
modelo += 50000*x1 + 70000*x2 + 130000*x3 + 160000*x4 <= 15000000
modelo += x3 + x4 >= 0.25*(x1 + x2 + x3 + x4)
modelo += x1 >= 0.20*(x1 + x2 + x3 + x4)
modelo += x2 >= 0.10*(x1 + x2 + x3 + x4)

# Resolver
modelo.solve()

# Mostrar resultados
print("Solución óptima (variables continuas):")
print(f"x1 = {x1.varValue}")
print(f"x2 = {x2.varValue}")
print(f"x3 = {x3.varValue}")
print(f"x4 = {x4.varValue}")
print(f"Recaudación máxima de impuestos = ${pulp.value(modelo.objective)}")