import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# Datos iniciales
p0 = np.array([0.325514, -0.459460, 0.166229])  # Posición inicial en UA
v0 = np.array([-9.096111, -6.916686, -1.305721])  # Velocidad inicial en UA/año
mu = 4 * np.pi**2  # Constante gravitacional

# Definimos el sistema de ecuaciones diferenciales de segundo orden
def sistema_ecuaciones(w, t):
    x, y, z, vx, vy, vz = w
    r = np.sqrt(x**2 + y**2 + z**2)
    d2x_dt2 = -mu * x / r**3
    d2y_dt2 = -mu * y / r**3
    d2z_dt2 = -mu * z / r**3
    return vx, vy, vz, d2x_dt2, d2y_dt2, d2z_dt2

# Condiciones iniciales para odeint (combinamos posición y velocidad)
w0 = np.concatenate((p0, v0))

# Intervalo de tiempo para la integración (en años)
# Ajusta el valor final para simular más o menos tiempo
t_span = np.linspace(0, 76, 1000)  # Simulamos un periodo orbital completo (aprox. 76 años)

# Resolvemos el sistema de ecuaciones diferenciales
sol = odeint(sistema_ecuaciones, w0, t_span)

# Extraemos las componentes de posición de la solución
x, y, z = sol[:, 0], sol[:, 1], sol[:, 2]

# Graficamos las proyecciones y la trayectoria 3D
fig = plt.figure(figsize=(12, 8))

# Proyección XY
ax1 = fig.add_subplot(221)
ax1.plot(x, y)
ax1.set_xlabel('X (UA)')
ax1.set_ylabel('Y (UA)')
ax1.set_title('Proyección XY')

# Proyección XZ
ax2 = fig.add_subplot(222)
ax2.plot(x, z)
ax2.set_xlabel('X (UA)')
ax2.set_ylabel('Z (UA)')
ax2.set_title('Proyección XZ')

# Proyección YZ
ax3 = fig.add_subplot(223)
ax3.plot(y, z)
ax3.set_xlabel('Y (UA)')
ax3.set_ylabel('Z (UA)')
ax3.set_title('Proyección YZ')

# Trayectoria 3D
ax4 = fig.add_subplot(224, projection='3d')
ax4.plot(x, y, z)
ax4.set_xlabel('X (UA)')
ax4.set_ylabel('Y (UA)')
ax4.set_zlabel('Z (UA)')
ax4.set_title('Trayectoria 3D')

plt.tight_layout()
plt.show()