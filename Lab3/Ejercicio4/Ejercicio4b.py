import numpy as np
from scipy.integrate import odeint

def comet_motion(w, t, mu):
    """
    Defines the system of differential equations for comet motion.

    Args:
        w: Array containing position (x, y, z) and velocity (vx, vy, vz) components.
        t: Time.
        mu: Gravitational parameter.

    Returns:
        Array containing the derivatives of position and velocity components.
    """
    x, y, z, vx, vy, vz = w
    r = np.sqrt(x**2 + y**2 + z**2)
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = -mu * x / r**3
    dvydt = -mu * y / r**3
    dvzdt = -mu * z / r**3
    return [dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt]

# Initial conditions from 1986
p0 = np.array([0.325514, -0.459460, 0.166229])  # Position in AU
v0 = np.array([-9.096111, -6.916686, -1.305721])  # Velocity in AU/year
mu = 4 * np.pi**2  # Gravitational parameter

# Time points for integration (in years from 1986)
t_span = np.array([100, 200])  # For 2086 and 2186

# Initial state vector
w0 = np.concatenate((p0, v0))

# Solve the differential equations using odeint
sol = odeint(comet_motion, w0, t_span, args=(mu,))

# Extract position and velocity at the desired times
p_2086 = sol[0, :3]
v_2086 = sol[0, 3:]
p_2186 = sol[1, :3]
v_2186 = sol[1, 3:]

# Print the results
print("Estimated position and velocity for 9th February 2086:")
print("Position (AU):", p_2086)
print("Velocity (AU/year):", v_2086)

print("\nEstimated position and velocity for 9th February 2186:")
print("Position (AU):", p_2186)
print("Velocity (AU/year):", v_2186)