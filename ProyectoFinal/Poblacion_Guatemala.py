import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

# Crear una carpeta temporal para almacenar las imágenes de cada año
Path("frames").mkdir(parents=True, exist_ok=True)

def crecimiento_exponencial(P0, r, t):
    """Simula el crecimiento exponencial de una población."""
    return P0 * np.exp(r * t)
def crecimiento_logistico(P0, r, K, t):
    """Simula el crecimiento logístico de una población."""
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

# Parámetros
P0 = 3000000  # Población inicial en miles (3 millones en 1950)
r = 0.029       # Tasa de crecimiento anual (1.8%)
K = 6000000       # Capacidad de carga para el modelo logístico (en miles)
años = 2050 - 1950  # Años para la simulación (2015 a 2050)

# Cálculo de poblaciones año por año
poblacion_exponencial = [crecimiento_exponencial(P0, r, t) for t in range(años)]
poblacion_logistica = [crecimiento_logistico(P0, r, K, t) for t in range(años)]

# Crear imágenes para cada año
frames = []
for año in range(años):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1950, 1950 + año + 1), poblacion_exponencial[:año+1], label="Crecimiento Exponencial", color="blue")
    plt.plot(range(1950, 1950 + año + 1), poblacion_logistica[:año+1], label="Crecimiento Logístico", color="green")
    plt.xlabel("Años")
    plt.ylabel("Población (en miles)")
    plt.title(f"Simulación de Crecimiento de Población - Año {2015 + año}")
    plt.legend()
    plt.grid(True)
    
    # Guardar cada frame
    filename = f"frames/frame_{2015 + año}.png"
    plt.savefig(filename)
    frames.append(imageio.imread(filename))
    plt.close()

# Crear el GIF
imageio.mimsave("simulacion_crecimiento_poblacion_guatemala.gif", frames, duration=0.5)

# Limpiar los frames temporales
for filename in Path("frames").glob("*.png"):
    filename.unlink()
Path("frames").rmdir()

print("GIF creado: simulacion_crecimiento_poblacion_guatemala.gif")