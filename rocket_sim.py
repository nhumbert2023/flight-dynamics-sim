import math
import numpy as np
import matplotlib.pyplot as plt


def simulate_rocket(
    dt=0.01,
    t_final=25.0,
    m0=50.0,             # kg, initial mass
    mdot=0.8,            # kg/s, fuel burn rate while engine on
    thrust=1200.0,       # N, constant thrust while engine on
    burn_time=8.0,       # s, engine burn duration
    Cd=0.6,              # drag coefficient
    A=0.03,              # m^2, reference area
    rho=1.225,           # kg/m^3, air density (sea level approx)
    g=9.81,              # m/s^2
    launch_angle_deg=85  # degrees from +x axis (90 = straight up)
):
    """
    Simple 2D point-mass rocket simulation:
    Forces: thrust (during burn), gravity, quadratic drag.
    State: x, y, vx, vy, m
    """

    # Time array
    t = np.arange(0.0, t_final + dt, dt)

    # State arrays
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    m = np.zeros_like(t)

    # Initial conditions
    m[0] = m0
    angle = math.radians(launch_angle_deg)

    for i in range(len(t) - 1):
        ti = t[i]

        # Engine on/off
        engine_on = ti <= burn_time and m[i] > (m0 - mdot * burn_time)
        Ti = thrust if engine_on else 0.0

        # Mass update (burn fuel)
        mi = m[i]
        if engine_on:
            mi_next = max(mi - mdot * dt, m0 - mdot * burn_time)
        else:
            mi_next = mi

        # Velocity magnitude for drag
        v = math.hypot(vx[i], vy[i])

        # Drag magnitude: 0.5 * rho * Cd * A * v^2
        D = 0.5 * rho * Cd * A * v * v

        # Drag direction opposite velocity
        if v > 1e-9:
            Dx = -D * (vx[i] / v)
            Dy = -D * (vy[i] / v)
        else:
            Dx, Dy = 0.0, 0.0

        # Thrust direction fixed at launch angle (simple model)
        Tx = Ti * math.cos(angle)
        Ty = Ti * math.sin(angle)

        # Gravity
        Gx = 0.0
        Gy = -mi * g

        # Net force
        Fx = Tx + Dx + Gx
        Fy = Ty + Dy + Gy

        # Acceleration
        ax = Fx / mi
        ay = Fy / mi

        # Integrate (Euler)
        vx[i + 1] = vx[i] + ax * dt
        vy[i + 1] = vy[i] + ay * dt
        x[i + 1] = x[i] + vx[i + 1] * dt
        y[i + 1] = y[i] + vy[i + 1] * dt
        m[i + 1] = mi_next

        # Stop if it hits the ground (y < 0)
        if y[i + 1] < 0:
            # Truncate arrays neatly
            t = t[: i + 2]
            x = x[: i + 2]
            y = y[: i + 2]
            vx = vx[: i + 2]
            vy = vy[: i + 2]
            m = m[: i + 2]
            break

    speed = np.sqrt(vx**2 + vy**2)
    return t, x, y, vx, vy, m, speed


def main():
    t, x, y, vx, vy, m, speed = simulate_rocket()

    # --- Plots ---
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Rocket Trajectory (2D)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, y)
    plt.xlabel("time (s)")
    plt.ylabel("altitude y (m)")
    plt.title("Altitude vs Time")
    plt.grid(True)

    plt.figure()
    plt.plot(t, speed)
    plt.xlabel("time (s)")
    plt.ylabel("speed (m/s)")
    plt.title("Speed vs Time")
    plt.grid(True)

    plt.show()

    # Quick summary
    print(f"Max altitude: {y.max():.1f} m")
    print(f"Range (x at landing): {x[-1]:.1f} m")
    print(f"Flight time: {t[-1]:.2f} s")


if __name__ == "__main__":
    main()
