import pulp

# Datos del problema
costos = [
    [100 * 25, 150 * 25, 200 * 25, 140 * 25, 35 * 25],
    [50 * 25, 70 * 25, 60 * 25, 65 * 25, 80 * 25],
    [40 * 25, 90 * 25, 100 * 25, 150 * 25, 130 * 25]
]
oferta = [400, 200, 150]
# Modificar la demanda del concesionario 5
demanda = [100, 200, 150, 160, 200]  

# Crear el modelo
modelo = pulp.LpProblem("Problema_de_Transporte", pulp.LpMinimize)

# Variables de decisión
x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Integer')
     for i in range(len(oferta)) for j in range(len(demanda))}

# Función objetivo
modelo += pulp.lpSum(costos[i][j] * x[i, j]
                    for i in range(len(oferta))
                    for j in range(len(demanda)))

# Restricciones de oferta
for i in range(len(oferta)):
    modelo += pulp.lpSum(x[i, j] for j in range(len(demanda))) <= oferta[i]

# Restricciones de demanda
for j in range(len(demanda)):
    modelo += pulp.lpSum(x[i, j] for i in range(len(oferta))) >= demanda[j]

# Resolver el modelo
modelo.solve()

# Imprimir la solución
print("Distribución óptima:")
for i in range(len(oferta)):
    for j in range(len(demanda)):
        if pulp.value(x[i, j]) > 0:
            print(f"Enviar {pulp.value(x[i, j])} automóviles del centro {i + 1} al concesionario {j + 1}")

print(f"\nCosto total de transporte: ${pulp.value(modelo.objective)}")