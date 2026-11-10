"""Linear 2-DoF bicycle model (lateral velocity v, yaw rate r) at constant forward speed u."""

# Vehicle parameters (SI units)
m = 1500.0     # mass [kg]
Iz = 2500.0    # yaw inertia [kg m^2]
a = 1.2        # CG to front axle [m]
b = 1.6        # CG to rear axle [m]
Cf = 80000.0   # front cornering stiffness [N/rad]
Cr = 80000.0   # rear cornering stiffness [N/rad]


def derivatives(v, r, delta, u):
    """Return (v_dot, r_dot) for front steer angle delta [rad] and speed u [m/s]."""
    alpha_f = (v + a * r) / u - delta   # front slip angle
    alpha_r = (v - b * r) / u           # rear slip angle
    Fyf = -Cf * alpha_f
    Fyr = -Cr * alpha_r
    v_dot = (Fyf + Fyr) / m - u * r
    r_dot = (a * Fyf - b * Fyr) / Iz
    return v_dot, r_dot


def simulate(delta_fn, u=20.0, t_end=5.0, dt=0.001):
    """Euler-integrate from rest. delta_fn(t) gives steer angle. Returns lists t, v, r."""
    t, v, r = [0.0], [0.0], [0.0]
    while t[-1] < t_end:
        vd, rd = derivatives(v[-1], r[-1], delta_fn(t[-1]), u)
        v.append(v[-1] + vd * dt)
        r.append(r[-1] + rd * dt)
        t.append(t[-1] + dt)
    return t, v, r


if __name__ == "__main__":
    u, delta = 20.0, 0.02
    t, v, r = simulate(lambda _: delta, u=u)
    # Check against the textbook steady-state yaw rate: r = u*delta / (L + K*u^2)
    L = a + b
    K = m / L * (b / Cf - a / Cr)  # understeer gradient
    r_ss = u * delta / (L + K * u**2)
    assert abs(r[-1] - r_ss) < 1e-3 * r_ss, (r[-1], r_ss)
    print(f"steady-state yaw rate: sim {r[-1]:.4f} rad/s, analytic {r_ss:.4f} rad/s")
