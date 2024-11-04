import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, chisquare

# Función del Generador Lineal Congruencial (GLC)
def glc(m, a, c, seed, size):
    x = seed
    random_numbers = []
    for _ in range(size):
        x = (a * x + c) % m
        random_numbers.append(x / m)  # Normalización a [0,1]
    return np.array(random_numbers)

# Parámetros para el generador teórico (GLC aleatorio)
m_teorico = 2**31 - 1
a_teorico = 1664525
c_teorico = 1013904223
seed_teorico = 42
N = 1000  # Tamaño de la muestra

# Parámetros para los generadores prácticos (GLC)
parametros_practicos = [
    (2**31 - 1, 1103515245, 12345),
    (2**16, 75, 74),
    (2**24, 214013, 2531011)
]

# Generación y comparación de muestras
for i, (m_practico, a_practico, c_practico) in enumerate(parametros_practicos):
    # Generar muestra teórica (GLC aleatorio)
    muestra_teorica = glc(m_teorico, a_teorico, c_teorico, seed_teorico, N)
    
    # Generar muestra práctica con parámetros específicos
    muestra_practica = glc(m_practico, a_practico, c_practico, seed_teorico, N)
    
    # Pruebas de Hipótesis
    # Prueba de Kolmogorov-Smirnov para comparar con una distribución uniforme
    ks_stat_teorico, p_val_teorico = kstest(muestra_teorica, 'uniform')
    ks_stat_practico, p_val_practico = kstest(muestra_practica, 'uniform')
    
    # Prueba de Chi-Cuadrado
    bins = np.linspace(0, 1, 11)  # Dividimos el rango en 10 intervalos para Chi-Cuadrado
    hist_teorico, _ = np.histogram(muestra_teorica, bins=bins)
    hist_practico, _ = np.histogram(muestra_practica, bins=bins)
    chi2_teorico, chi2_pval_teorico = chisquare(hist_teorico)
    chi2_practico, chi2_pval_practico = chisquare(hist_practico)
    
    # Resultados
    print(f"Parámetros del conjunto {i+1}: m={m_practico}, a={a_practico}, c={c_practico}")
    print(f"\nPrueba de Kolmogorov-Smirnov (Teórico): estadístico = {ks_stat_teorico:.4f}, valor p = {p_val_teorico:.4f}")
    print(f"Prueba de Chi-Cuadrado (Teórico): estadístico = {chi2_teorico:.4f}, valor p = {chi2_pval_teorico:.4f}")
    
    print(f"\nPrueba de Kolmogorov-Smirnov (Práctico): estadístico = {ks_stat_practico:.4f}, valor p = {p_val_practico:.4f}")
    print(f"Prueba de Chi-Cuadrado (Práctico): estadístico = {chi2_practico:.4f}, valor p = {chi2_pval_practico:.4f}")
    
    if p_val_teorico > 0.05:
        print("Muestra teórica: No se rechaza la hipótesis nula. La muestra generada puede considerarse como uniforme.\n")
    else:
        print("Muestra teórica: Se rechaza la hipótesis nula. La muestra generada no puede considerarse como uniforme.\n")
    
    if p_val_practico > 0.05:
        print("Muestra práctica: No se rechaza la hipótesis nula. La muestra generada puede considerarse como uniforme.\n")
    else:
        print("Muestra práctica: Se rechaza la hipótesis nula. La muestra generada no puede considerarse como uniforme.\n")
    
    # Graficar histogramas de la muestra teórica y práctica
    plt.figure(figsize=(12, 5))
    
    # Histograma de la muestra teórica
    plt.subplot(1, 2, 1)
    plt.hist(muestra_teorica, bins=15, density=True, alpha=0.6, color='green', edgecolor='black')
    plt.title(f'Histograma de Muestra Teórica (Conjunto {i+1})')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    
    # Histograma de la muestra práctica
    plt.subplot(1, 2, 2)
    plt.hist(muestra_practica, bins=15, density=True, alpha=0.6, color='blue', edgecolor='black')
    plt.title(f'Histograma de Muestra Práctica (Conjunto {i+1})')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    
    plt.show()
