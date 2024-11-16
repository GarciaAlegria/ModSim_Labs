import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation, PillowWriter

# Datos de población y años
years = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2024]) # Años de los datos de población obtenidos de https://datacommons.org/place/country/GTM?utm_medium=explore&mprop=count&popt=Person&hl=es
population = np.array([3000000, 4128880, 5455197, 6890346, 9050115, 11589761, 14259687, 18602431]) # Población de Guatemala en cada año obtenida de https://datacommons.org/place/country/GTM?utm_medium=explore&mprop=count&popt=Person&hl=es

# Convertir años a "años desde 1950"
t = years - 1950

# Modelo exponencial
def exponential_model(t, P0, r): # P0: población inicial, r: tasa de crecimiento, t: tiempo
    return P0 * np.exp(r * t) # P(t) = P0 * e^(rt)

# Modelo logístico
def logistic_model(t, P0, r, K): # P0: población inicial, r: tasa de crecimiento, K: capacidad de carga, t: tiempo
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t)) # P(t) = K / (1 + ((K - P0) / P0) * e^(-rt))

# Ajustar el modelo exponencial
params_exp, _ = curve_fit(exponential_model, t, population, p0=[3e6, 0.02]) # P0 = 3e6, r = 0.02
P0_exp, r_exp = params_exp # Población inicial y tasa de crecimiento

# Ajustar el modelo logístico
params_log, _ = curve_fit(logistic_model, t, population, p0=[3e6, 0.02, 2e7]) # P0 = 3e6, r = 0.02, K = 2e7
P0_log, r_log, K_log = params_log # Población inicial, tasa de crecimiento y capacidad de carga

# Imprimir constantes
print("Modelo exponencial:")
print(f"P0: {P0_exp:.2f}, r: {r_exp:.6f}") # Imprimir con 2 decimales y 6 decim
print("\nModelo logístico:")
print(f"P0: {P0_log:.2f}, r: {r_log:.6f}, K: {K_log:.2f}") # Imprimir con 2 decimales y 6 decim

# Predicción para 2025-2050
years_future = np.arange(2025, 2051) # Años futuros para proyección de población (2025-2050)
t_future = years_future - 1950 # Convertir años futuros a "años desde 1950"

pop_exp_future = exponential_model(t_future, P0_exp, r_exp) # Proyección de población con modelo exponencial
pop_log_future = logistic_model(t_future, P0_log, r_log, K_log) # Proyección de población con modelo logístico

# Crear GIF
fig, ax = plt.subplots() # Crear figura y ejes
ax.set_xlim(2025, 2050) # Rango X para años futuros
ax.set_ylim(17e6, 35e6)  # Ajuste del rango Y para mayor claridad
ax.set_xlabel("Año") # Etiqueta del eje X
ax.set_ylabel("Población") # Etiqueta del eje Y
ax.set_title("Simulación de crecimiento poblacional") # Título del gráfico

line_exp, = ax.plot([], [], label="Modelo Exponencial", color="blue") # Línea para modelo exponencial
line_log, = ax.plot([], [], label="Modelo Logístico", color="green") # Línea para modelo logístico
ax.legend() # Mostrar leyenda

def update(frame): # Función de actualización para la animación
    current_years = years_future[:frame] # Años actuales para el cuadro actual de la animación
    current_pop_exp = pop_exp_future[:frame] # Población proyectada con modelo exponencial
    current_pop_log = pop_log_future[:frame] # Población proyectada con modelo logístico

    line_exp.set_data(current_years, current_pop_exp) # Actualizar datos de la línea exponencial
    line_log.set_data(current_years, current_pop_log) # Actualizar datos de la línea logística

    return line_exp, line_log # Devolver líneas actualizadas

ani = FuncAnimation(fig, update, frames=len(years_future), interval=200, blit=True) # Crear animación con FuncAnimation

# Guardar el GIF
writer = PillowWriter(fps=5) # Configurar el escritor de video
ani.save("population_simulation.gif", writer=writer) # Guardar la animación como un GIF

plt.show() # Mostrar la animación

# Crear GIF con puntos dispersos iniciales
fig, ax = plt.subplots() # Crear figura y ejes
ax.set_xlim(2025, 2050) # Rango X para años futuros
ax.set_ylim(0, 25e6)  # Ajuste del rango Y para mayor claridad
ax.set_xlabel("Año") # Etiqueta del eje X
ax.set_ylabel("Población") # Etiqueta del eje Y
ax.set_title("Simulación de crecimiento poblacional (proyecciones 2025 - 2050)") # Título del gráfico

scatter = ax.scatter([], [], color="blue", alpha=0.6, label="Simulación Poblacional") # Puntos dispersos para simulación
ax.legend() # Mostrar leyenda

# Número inicial de puntos dispersos
n_points_start = 50 # Número inicial de puntos dispersos
n_points_growth = 110  # Incremento de puntos en cada cuadro

# Función de actualización para la animación
def update_random(frame): # Función de actualización para la animación
    current_year = years_future[frame] # Año actual para el cuadro actual de la animación
    current_pop_log = pop_log_future[frame] # Población proyectada con modelo logístico 
    
    # Generar puntos aleatorios iniciales y expandir en cada cuadro
    total_points = n_points_start + frame * n_points_growth # Número total de puntos dispersos en el cuadro actual
    random_years = np.random.uniform(2025, current_year, total_points) # Años aleatorios en el rango actual
    random_pops = np.random.uniform(0, current_pop_log, total_points) # Poblaciones aleatorias en el rango actual
    
    # Actualizar datos de dispersión
    scatter.set_offsets(np.c_[random_years, random_pops]) # Actualizar posiciones de puntos dispersos
    return scatter, # Devolver puntos dispersos actualizados

ani = FuncAnimation(fig, update_random, frames=len(years_future), interval=200, blit=True) # Crear animación con FuncAnimation

# Guardar el GIF
writer = PillowWriter(fps=5) # Configurar el escritor de video
ani.save("population_random_growth_expansion.gif", writer=writer) # Guardar la animación como un GIF

plt.show() # Mostrar la animación

# Imprimir proyecciones
print("\nProyecciones de población (2025-2050):") # Imprimir proyecciones de población
print("Año\tExponencial\tLogístico") # Encabezado de la tabla
for year, exp, log in zip(years_future, pop_exp_future, pop_log_future): # Iterar sobre años y proyecciones
    print(f"{year}\t{exp:.0f}\t{log:.0f}") # Imprimir año y proyecciones de población

# Calcular la población estimada para el año 2024 utilizando los modelos ajustados
year_2024 = 2024 # Año para el que se desea estimar la población
t_2024 = year_2024 - 1950 # Convertir el año a "años desde 1950"

pop_exp_2024 = exponential_model(t_2024, P0_exp, r_exp) # Estimar la población para 2024 con el modelo exponencial
pop_log_2024 = logistic_model(t_2024, P0_log, r_log, K_log) # Estimar la población para 2024 con el modelo logístico

print(f"\nPoblación estimada para el año 2024 (Modelo Exponencial): {pop_exp_2024:.0f}") # Imprimir la población estimada para 2024
print(f"Población estimada para el año 2024 (Modelo Logístico): {pop_log_2024:.0f}") # Imprimir la población estimada para 2024

# poblacion para el año 2050
print("\nPoblación proyectada para el año 2050:") # Imprimir la población proyectada para el año 2050
print(f"Modelo exponencial: {pop_exp_future[-1]:.0f}") # Imprimir la población proyectada para 2050 con el modelo exponencial
print(f"Modelo logístico: {pop_log_future[-1]:.0f}") # Imprimir la población proyectada para 2050 con el modelo logístico

print(f"Diferencia entre modelos (2050): {pop_exp_future[-1] - pop_log_future[-1]:.0f}")