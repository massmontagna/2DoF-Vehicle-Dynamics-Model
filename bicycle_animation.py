"""Animate the 2-DoF bicycle model driving a steer manoeuvre, plus yaw-rate and v plots."""
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bicycle_model import a, b, simulate

u = 20.0                                            # forward speed [m/s]
steer = lambda t: 0.03 if 0.5 < t < 3.0 else -0.03 if 3.0 <= t < 5.5 else 0.0  # lane-change-ish [rad]
dt = 0.001
t, v, r = simulate(steer, u=u, t_end=8.0, dt=dt)

# Integrate global position and heading from body-frame velocities
X, Y, psi = [0.0], [0.0], [0.0]
for vi, ri in zip(v[:-1], r[:-1]):
    X.append(X[-1] + (u * math.cos(psi[-1]) - vi * math.sin(psi[-1])) * dt)
    Y.append(Y[-1] + (u * math.sin(psi[-1]) + vi * math.cos(psi[-1])) * dt)
    psi.append(psi[-1] + ri * dt)

fig, (ax_xy, ax_r) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})
ax_xy.plot(X, Y, "--", color="0.7")
ax_xy.set_aspect("equal")
ax_xy.set_xlabel("X [m]")
ax_xy.set_ylabel("Y [m]")
ax_xy.set_title("2-DoF bicycle model")
body, = ax_xy.plot([], [], "k-", lw=2)
front, = ax_xy.plot([], [], "r-", lw=5)
rear, = ax_xy.plot([], [], "b-", lw=5)

ax_r.plot(t, r, label="yaw rate r [rad/s]")
ax_r.plot(t, v, label="lateral velocity v [m/s]")
ax_r.plot(t, [steer(ti) for ti in t], label="steer δ [rad]")
ax_r.set_xlabel("t [s]")
ax_r.legend(loc="upper right")
cursor = ax_r.axvline(0, color="k")

WHEEL = 0.35  # half wheel length [m], purely visual
step = 40     # simulation samples per frame


def wheel(x, y, angle):
    dx, dy = WHEEL * math.cos(angle), WHEEL * math.sin(angle)
    return [x - dx, x + dx], [y - dy, y + dy]


def update(frame):
    i = frame * step
    x, y, h = X[i], Y[i], psi[i]
    xf, yf = x + a * math.cos(h), y + a * math.sin(h)
    xr, yr = x - b * math.cos(h), y - b * math.sin(h)
    body.set_data([xr, xf], [yr, yf])
    front.set_data(*wheel(xf, yf, h + 5 * steer(t[i])))  # steer exaggerated 5x so it's visible
    rear.set_data(*wheel(xr, yr, h))
    ax_xy.set_xlim(x - 15, x + 15)
    ax_xy.set_ylim(y - 8, y + 8)
    cursor.set_xdata([t[i], t[i]])
    return body, front, rear, cursor


anim = FuncAnimation(fig, update, frames=len(t) // step, interval=step * dt * 1000)
plt.tight_layout()
plt.show()
