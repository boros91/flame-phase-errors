"""Core utilities for the flame problem  y' = y^2 - y^3,  y(0) = delta.

Exact solution via the Wright omega function:
    y_delta(t) = 1 / (1 + omega(a + ln a - t)),   a = 1/delta - 1,
so y_delta(t) = Y(t - c(delta)) with the master profile Y(t) = 1/(1+omega(1-t))
(Y(0) = 1/2) and c(delta) = a + ln a - 1.
"""
import numpy as np
from numba import njit
from scipy.special import wrightomega


def shift(delta):
    a = 1.0 / delta - 1.0
    return a + np.log(a) - 1.0


def master(t):
    """Master profile Y(t) = 1/(1+omega(1-t)), Y(0) = 1/2."""
    return 1.0 / (1.0 + np.real(wrightomega(1.0 - np.asarray(t, dtype=float))))


def exact(t, delta):
    return master(np.asarray(t, dtype=float) - shift(delta))


def T(y, delta):
    """Exact time needed to go from delta to y (0 < delta <= y < 1)."""
    y = np.asarray(y, dtype=float)
    return (1.0 / delta - 1.0 / y) + np.log(y * (1.0 - delta) / (delta * (1.0 - y)))


# ---------------------------------------------------------------- one-step maps
@njit(cache=True)
def f(y):
    return y * y - y * y * y


@njit(cache=True)
def step(method, y, h, phi):
    if method == 0:            # explicit Euler
        return y + h * f(y)
    elif method == 1:          # implicit Euler, Newton on z - y - h f(z) = 0
        z = y + h * f(y)
        for _ in range(50):
            r = z - y - h * f(z)
            dz = r / (1.0 - h * (2 * z - 3 * z * z))
            z -= dz
            if abs(dz) < 1e-16 * abs(z):
                break
        return z
    elif method == 2:          # NSFD (3)
        return (1.0 + 2.0 * phi * y) * y / (1.0 + y * (1.0 + y) * phi)
    elif method == 3:          # NSFD (4), a = 1, b = -3
        a, b = 1.0, -3.0
        return (y + phi * (a * y * y - b * y ** 3)) / (1.0 - phi * (1 - a) * y + phi * (1 - b) * y * y)
    elif method == 4:          # Heun
        k1 = f(y)
        k2 = f(y + h * k1)
        return y + 0.5 * h * (k1 + k2)
    else:                      # classical RK4
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(cache=True)
def run_until(method, delta, h, phi, eta, nmax):
    """Iterate until y_k >= eta (or nmax steps). Returns (k, y_k)."""
    y = delta
    for k in range(nmax):
        if y >= eta:
            return k, y
        y = step(method, y, h, phi)
    return nmax, y


@njit(cache=True)
def trajectory(method, delta, h, phi, n):
    ys = np.empty(n + 1)
    ys[0] = delta
    for k in range(n):
        ys[k + 1] = step(method, ys[k], h, phi)
    return ys


METHODS = {
    'Euler': (0, 1), 'Implicit Euler': (1, 1),
    'NSFD3': (2, 1), 'NSFD4': (3, 1),
    'Heun': (4, 2), 'RK4': (5, 4),
}

# antiderivatives of the delay density g = -C/f^2 (from theory_symbolic.py);
# L1(y) = log(1-y)
_G = {
    'Euler':          lambda y: np.log(y) + 0.5 * np.log1p(-y),
    'Implicit Euler': lambda y: -np.log(y) - 0.5 * np.log1p(-y),
    'NSFD3':          lambda y: 2 * np.log(y) - 1.5 * np.log1p(-y),
    'NSFD4':          lambda y: np.log(y) - 3.5 * np.log1p(-y),
    'Heun':           lambda y: -y**2 / 2 + y / 3 - np.log1p(-y) / 6,
    'RK4':            lambda y: (-y**6 / 96 + y**5 / 48 - y**4 / 32 + y**3 / 90
                                 - y**2 / 240 - y / 120 - np.log1p(-y) / 120),
}


def predicted_delay(name, delta, y, h):
    p = METHODS[name][1]
    return h**p * (_G[name](y) - _G[name](delta))


def phase_delay(name, delta, h, eta=0.5, phi=None):
    """Exact phase delay D = t_k - T(y_k) at the first step with y_k >= eta."""
    m, _ = METHODS[name]
    if phi is None:
        phi = h
    nmax = int(50.0 / (delta * h)) + 10
    k, yk = run_until(m, delta, h, phi, eta, nmax)
    if k >= nmax:
        return np.nan, np.nan, k
    return k * h - T(yk, delta), yk, k
