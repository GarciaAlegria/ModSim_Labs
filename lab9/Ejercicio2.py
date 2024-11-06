import numpy as np
from scipy.stats import geom, chisquare, kstest

# Definir parámetros
p = 0.3  # Probabilidad de éxito en la distribución geométrica
N = 1000  # Tamaño de la muestra

# Generar muestras

# Muestra teórica usando scipy.stats
muestra_teorica = geom.rvs(p, size=N)

# Muestra empírica usando la transformada integral
def transformada_integral(p, N):
    """
    Genera una muestra de una distribución geométrica utilizando la 
    transformada integral.

    Args:
        p: Probabilidad de éxito.
        N: Tamaño de la muestra.

    Returns:
        Una lista de números que representan la muestra.
    """
    muestra = []
    for _ in range(N):
        u = np.random.rand()
        x = int(np.ceil(np.log(1 - u) / np.log(1 - p)))
        muestra.append(x)
    return muestra

muestra_empirica = transformada_integral(p, N)

# Prueba de Chi-Cuadrado

# Calcular las frecuencias observadas en la muestra empírica
frecuencias_observadas = np.bincount(muestra_empirica)

# Calcular las frecuencias esperadas según la distribución geométrica
frecuencias_esperadas = geom.pmf(np.arange(len(frecuencias_observadas)), p) * N

# Filtrar las frecuencias para eliminar ceros
nonzero_indices = frecuencias_esperadas > 0
frecuencias_observadas = frecuencias_observadas[nonzero_indices]
frecuencias_esperadas = frecuencias_esperadas[nonzero_indices]

# Normalizar las frecuencias observadas y esperadas para que sumen al mismo valor
frecuencias_observadas = frecuencias_observadas * (frecuencias_esperadas.sum() / frecuencias_observadas.sum())

# Realizar la prueba de Chi-Cuadrado
chi2_stat, chi2_p_valor = chisquare(frecuencias_observadas, f_exp=frecuencias_esperadas)

# Imprimir resultados
print(f"Chi-Cuadrado estadístico: {chi2_stat}")
print(f"Chi-Cuadrado p-valor: {chi2_p_valor}")

# Comparar el p-valor con el nivel de significancia
alpha = 0.05
if chi2_p_valor > alpha:
    print("No se rechaza la hipótesis nula (Chi-Cuadrado): Las muestras provienen de la misma distribución.")
else:
    print("Se rechaza la hipótesis nula (Chi-Cuadrado): Las muestras NO provienen de la misma distribución.")

# Prueba de Kolmogorov-Smirnov

# Realizar la prueba de Kolmogorov-Smirnov
ks_stat, ks_p_valor = kstest(muestra_empirica, 'geom', args=(p,))

# Imprimir resultados
print(f"Kolmogorov-Smirnov estadístico: {ks_stat}")
print(f"Kolmogorov-Smirnov p-valor: {ks_p_valor}")

# Comparar el p-valor con el nivel de significancia
if ks_p_valor > alpha:
    print("No se rechaza la hipótesis nula (Kolmogorov-Smirnov): Las muestras provienen de la misma distribución.")
else:
    print("Se rechaza la hipótesis nula (Kolmogorov-Smirnov): Las muestras NO provienen de la misma distribución.")