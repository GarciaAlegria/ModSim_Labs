import pulp as pl
import matplotlib.pyplot as plt

# Datos del problema
demanda = [180, 250, 190, 140, 220, 250]
costo_produccion = [50, 45, 55, 52, 48, 50]
costo_inventario = [8, 10, 10, 10, 8, 8]
capacidad_maxima = 225

# Crear modelo
modelo = pl.LpProblem("Problema_de_Produccion", pl.LpMinimize)

# Variables de decisión
produccion = pl.LpVariable.dicts("Produccion", range(1, 7), 0, capacidad_maxima)
inventario = pl.LpVariable.dicts("Inventario", range(7), 0)

# Restricciones
for i in range(6):
    # Balance de inventario
    if i == 0:
        modelo += produccion[1] == demanda[0] + inventario[1]
    else:
        modelo += produccion[i+1] + inventario[i] == demanda[i] + inventario[i+1]

# Función objetivo
costo_total = pl.lpSum(costo_produccion[i] * produccion[i+1] for i in range(6)) + pl.lpSum(costo_inventario[i] * inventario[i+1] for i in range(6))
modelo += costo_total

# Resolver el modelo
modelo.solve()

# Obtener la solución óptima
produccion_optima = [pl.value(produccion[i+1]) for i in range(6)]
inventario_optimo = [pl.value(inventario[i]) for i in range(7)]

# Imprimir la solución óptima
print("Producción óptima:", produccion_optima)
print("Inventario óptimo:", inventario_optimo)

# Calcular el costo total de la solución óptima
costo_total_optimo = pl.value(costo_total)

# Calcular la producción no óptima (igual a la demanda)
produccion_no_optima = demanda
inventario_no_optimo = [0] * 7

# Calcular el costo total de la producción no óptima
costo_total_no_optimo = sum(costo_produccion[i] * produccion_no_optima[i] for i in range(6)) + sum(costo_inventario[i] * inventario_no_optimo[i+1] for i in range(6))

# Imprimir los costos totales
print("Costo total óptimo:", costo_total_optimo)
print("Costo total no óptimo:", costo_total_no_optimo)

# Calcular el ahorro
ahorro = costo_total_no_optimo - costo_total_optimo
print("Ahorro:", ahorro)

# Crear una tabla con los costos
print("\nTabla de costos:")
print("Mes | Producción | Costo Prod. | Inventario | Costo Inv. | Costo Total | Ahorro")
print("-" * 67)
for i in range(6):
    costo_prod = costo_produccion[i] * produccion_optima[i]
    costo_inv = costo_inventario[i] * inventario_optimo[i+1]
    costo_total_mes = costo_prod + costo_inv
    ahorro_mes = costo_produccion[i] * produccion_no_optima[i] - costo_prod
    print(f"{i+1}   | {produccion_optima[i]:.0f}        | ${costo_prod:.2f}     | {inventario_optimo[i+1]:.0f}         | ${costo_inv:.2f}      | ${costo_total_mes:.2f}    | ${ahorro_mes:.2f}")

# Crear un diagrama de la producción óptima
plt.figure(figsize=(10, 5))
plt.bar(range(1, 7), produccion_optima, label="Producción")
plt.plot(range(1, 7), inventario_optimo[1:], color="red", marker="o", label="Inventario")
plt.xticks(range(1, 7))
plt.xlabel("Mes")
plt.ylabel("Unidades")
plt.title("Producción e Inventario Óptimos")
plt.legend()
plt.show()