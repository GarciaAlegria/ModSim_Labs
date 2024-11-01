import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2_contingency, kstest
import seaborn as sns

# Parámetros de la distribución normal
mu = 0  # media
sigma = 1  # desviación estándar
N = 500  # tamaño de la muestra

# Generación de la muestra teórica
muestra_teorica = norm.rvs(loc=mu, scale=sigma, size=N)

# Generación de la muestra empírica usando el método de transformación integral
u = np.random.rand(N)
muestra_empirica = norm.ppf(u, loc=mu, scale=sigma)

# Dibujar histogramas
plt.figure(figsize=(12, 6))

# Histograma de la muestra teórica con línea de densidad
plt.subplot(1, 2, 1)
sns.histplot(muestra_teorica, bins=20, kde=True, color='g', label='Teórica')
plt.title('Histograma de la muestra teórica')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')

# Histograma de la muestra empírica con línea de densidad
plt.subplot(1, 2, 2)
sns.histplot(muestra_empirica, bins=20, kde=True, color='b', label='Empírica')
plt.title('Histograma de la muestra empírica')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Prueba de chi-cuadrado
observed_freq, _ = np.histogram(muestra_empirica, bins=20)
expected_freq, _ = np.histogram(muestra_teorica, bins=20)

# Evitar frecuencias esperadas cero
expected_freq = expected_freq + 1e-10

# Prueba de Chi-Cuadrado para comparar las frecuencias observadas y esperadas 
chi2_stat, p_val_chi2 = chi2_contingency([observed_freq, expected_freq])[:2]

print(f'Prueba de Chi-Cuadrado:')
print(f'Estadístico chi-cuadrado: {chi2_stat}')
print(f'Valor p: {p_val_chi2}')

# Nivel de significancia
alpha = 0.05

# Determinar si se rechaza la hipótesis nula
if p_val_chi2 < alpha:
    print("Se rechaza la hipótesis nula en la prueba de Chi-Cuadrado.")
else:
    print("No se rechaza la hipótesis nula en la prueba de Chi-Cuadrado.")

# Prueba de Kolmogorov-Smirnov
ks_stat, p_val_ks = kstest(muestra_empirica, 'norm', args=(mu, sigma))

print(f'\nPrueba de Kolmogorov-Smirnov:')
print(f'Estadístico KS: {ks_stat}')
print(f'Valor p: {p_val_ks}')

# Determinar si se rechaza la hipótesis nula
if p_val_ks < alpha:
    print("Se rechaza la hipótesis nula en la prueba de Kolmogorov-Smirnov.")
else:
    print("No se rechaza la hipótesis nula en la prueba de Kolmogorov-Smirnov.")