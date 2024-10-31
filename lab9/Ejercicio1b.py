import scipy.stats as stats
import numpy as np

# Generar dos muestras de datos
muestra1 = np.random.normal(0, 1, 100)  # Muestra de una distribución normal
muestra2 = np.random.uniform(0, 1, 100) # Muestra de una distribución uniforme

# Realizar la prueba de Kolmogorov-Smirnov
ks_stat, p_valor = stats.ks_2samp(muestra1, muestra2)

# Imprimir los resultados
print(f"Estadístico KS: {ks_stat}")
print(f"Valor p: {p_valor}")