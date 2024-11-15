import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation, PillowWriter

# Datos de población y años
years = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2024])
population = np.array([3000000, 4128880, 5455197, 6890346, 9050115, 11589761, 14259687, 18602431])

# Convertir años a "años desde 1950"
t = years - 1950

# Modelo exponencial
def exponential_model(t, P0, r):
    return P0 * np.exp(r * t)

# Modelo logístico
def logistic_model(t, P0, r, K):
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

# Ajustar el modelo exponencial
params_exp, _ = curve_fit(exponential_model, t, population, p0=[3e6, 0.02])
P0_exp, r_exp = params_exp

# Ajustar el modelo logístico
params_log, _ = curve_fit(logistic_model, t, population, p0=[3e6, 0.02, 2e7])
P0_log, r_log, K_log = params_log

# Imprimir constantes
print("Modelo exponencial:")
print(f"P0: {P0_exp:.2f}, r: {r_exp:.6f}")
print("\nModelo logístico:")
print(f"P0: {P0_log:.2f}, r: {r_log:.6f}, K: {K_log:.2f}")

# Predicción para 2025-2050
years_future = np.arange(2025, 2051)
t_future = years_future - 1950

pop_exp_future = exponential_model(t_future, P0_exp, r_exp)
pop_log_future = logistic_model(t_future, P0_log, r_log, K_log)

# Crear GIF
fig, ax = plt.subplots()
ax.set_xlim(2025, 2050)
ax.set_ylim(17e6, 35e6)  # Ajuste del rango Y para mayor claridad
ax.set_xlabel("Año")
ax.set_ylabel("Población")
ax.set_title("Simulación de crecimiento poblacional")

line_exp, = ax.plot([], [], label="Modelo Exponencial", color="blue")
line_log, = ax.plot([], [], label="Modelo Logístico", color="green")
ax.legend()

def update(frame):
    current_years = years_future[:frame]
    current_pop_exp = pop_exp_future[:frame]
    current_pop_log = pop_log_future[:frame]

    line_exp.set_data(current_years, current_pop_exp)
    line_log.set_data(current_years, current_pop_log)

    return line_exp, line_log

ani = FuncAnimation(fig, update, frames=len(years_future), interval=200, blit=True)

# Guardar el GIF
writer = PillowWriter(fps=5)
ani.save("population_simulation.gif", writer=writer)

plt.show()

# Imprimir proyecciones
print("\nProyecciones de población (2025-2050):")
print("Año\tExponencial\tLogístico")
for year, exp, log in zip(years_future, pop_exp_future, pop_log_future):
    print(f"{year}\t{exp:.0f}\t{log:.0f}")