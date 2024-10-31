import scipy.stats as stats

# Datos observados
observados = [15, 25, 20, 40] 

# Datos esperados (siguiendo una distribución uniforme, por ejemplo)
esperados = [25, 25, 25, 25] 

# Realizar la prueba de Chi-Cuadrado
chi2_stat, p_valor = stats.chisquare(observados, f_exp=esperados)

# Imprimir los resultados
print(f"Estadístico Chi-Cuadrado: {chi2_stat}")
print(f"Valor p: {p_valor}")