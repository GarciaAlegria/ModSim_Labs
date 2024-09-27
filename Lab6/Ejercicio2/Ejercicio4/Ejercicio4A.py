from scipy.optimize import linprog

# Definir los coeficientes de la función objetivo (impuestos por unidad)
c = [-1000, -1900, -2700, -3400]  # Se multiplica por -1 para maximizar

# Definir los coeficientes de las restricciones
A = [
    [18, 28, 40, 50],  # Restricción de acres por unidad
    [50000, 70000, 130000, 160000],  # Restricción de costo de construcción
    [-1, -1, -1, -1],  # Restricción de unidades totales (para el 25% de triples y cuádruples)
    [1, 1, 1, 1],  # Restricción de unidades totales (para el 20% de sencillas)
    [1, 1, 1, 1],  # Restricción de unidades totales (para el 10% de dobles)
]

# Definir los límites de las restricciones
b = [
    300 * 25 * 0.85,  # 85% del área total disponible después de la demolición
    15000000,  # Límite de financiamiento
    -0.25,  # Al menos 25% de triples y cuádruples
    0.20,  # Al menos 20% de sencillas
    0.10,  # Al menos 10% de dobles
]

# Definir los límites de las variables (no negatividad y número entero)
x_bounds = [(0, None), (0, None), (0, None), (0, None)]

# Resolver el problema de programación lineal
res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds, integrality=1)

# Imprimir los resultados
print("Unidades sencillas:", int(res.x[0]))
print("Unidades dobles:", int(res.x[1]))
print("Unidades triples:", int(res.x[2]))
print("Unidades cuádruples:", int(res.x[3]))
print("Recaudación máxima de impuestos:", int(-res.fun))  # Se vuelve a multiplicar por -1