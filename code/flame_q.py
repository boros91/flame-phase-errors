"""Generalised flame family  y' = f_q(y) = y^q (1 - y),  y(0) = delta,  q = 2, 3, ...
(q = 2: flame problem; q = 3: cubic autocatalysis).

Time function: Theta_q(y) = sum_{j=1}^{q-1} -1/(j y^j) + ln(y/(1-y)),
T_delta(y) = Theta_q(y) - Theta_q(delta), master profile Y_q = Theta_q^{-1},
y_delta(t) = Y_q(t + Theta_q(delta)).  Nearest complex singularities: s = +- i*pi.
"""
import numpy as np
from numba import njit


def Theta(y, q):
    y = np.asarray(y, dtype=float)
    out = np.log(y) - np.log1p(-y)
    for j in range(1, q):
        out = out - 1.0 / (j * y**j)
    return out


def T(y, delta, q):
    return Theta(y, q) - Theta(delta, q)


def _theta_logit(l, q):
    y = 1.0 / (1.0 + np.exp(-l))
    out = l.copy()
    for j in range(1, q):
        out = out - 1.0 / (j * y**j)
    return out


def master(s, q, iters=200):
    """Y_q(s) = Theta_q^{-1}(s): vectorised bisection in the logit variable l = ln(y/(1-y))
    (robust for y -> 0 and y -> 1), followed by Newton polishing in l."""
    s = np.atleast_1d(np.asarray(s, dtype=float))
    lo = np.full(s.shape, -60.0); hi = np.full(s.shape, 60.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = _theta_logit(mid, q) < s
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
        if np.max(hi - lo) < 1e-13:
            break
    l = 0.5 * (lo + hi)
    for _ in range(3):   # dTheta/dl = 1/(y^{q-1}(1-y)) * y(1-y) = y^{1-q}
        y = 1.0 / (1.0 + np.exp(-l))
        l = l - (_theta_logit(l, q) - s) * y**(q - 1)
    return 1.0 / (1.0 + np.exp(-l))


def exact(t, delta, q):
    return master(np.asarray(t, dtype=float) + Theta(delta, q), q)


@njit(cache=True)
def fq(y, q):
    return y**q * (1.0 - y)


@njit(cache=True)
def incr(method, y, h, q):
    """Increment y_{k+1} - y_k (computed directly, so that compensated summation applies)."""
    if method == 0:                       # explicit Euler
        return h * fq(y, q)
    elif method == 1:                     # implicit Euler: d = h f(y + d), Newton in d
        d = h * fq(y, q)
        for _ in range(50):
            z = y + d
            r = d - h * fq(z, q)
            dfz = q * z**(q - 1) - (q + 1) * z**q
            dd = r / (1.0 - h * dfz)
            d -= dd
            if abs(dd) <= 1e-16 * abs(d):
                break
        return d
    elif method == 2:                     # Heun
        k1 = fq(y, q); k2 = fq(y + h * k1, q)
        return 0.5 * h * (k1 + k2)
    elif method == 4:                     # NSFD1 (q = 2, phi(h) = h): Abelman-Patidar (3)
        return h * y * y * (1.0 - y) / (1.0 + h * y * (1.0 + y))
    elif method == 5:                     # NSFD2 (q = 2, phi(h) = h, a = 1, b = -3)
        a, b = 1.0, -3.0
        return h * y * y * (1.0 - y) / (1.0 - h * (1.0 - a) * y + h * (1.0 - b) * y * y)
    else:                                 # RK4 (method 3)
        k1 = fq(y, q); k2 = fq(y + 0.5 * h * k1, q)
        k3 = fq(y + 0.5 * h * k2, q); k4 = fq(y + h * k3, q)
        return h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(cache=True)
def dtheta(y, d, q):
    """Theta(y+d) - Theta(y), evaluated without cancellation."""
    r = d / y
    lr = np.log1p(r)
    out = lr - np.log1p(-d / (1.0 - y))
    yj = 1.0
    for j in range(1, q):
        yj *= y
        out += -np.expm1(-j * lr) / (j * yj)
    return out


@njit(cache=True)
def run_until(method, delta, h, q, eta, nmax):
    """Iterate with Kahan-compensated accumulation of y and exact accumulation of the
    phase delay D_k = t_k - T(y_k) = sum_j [h - (Theta(y_{j+1}) - Theta(y_j))].
    Returns (k, y_k, D_k) at the first k with y_k >= eta."""
    y = delta; c = 0.0; D = 0.0
    for k in range(nmax):
        if y >= eta:
            return k, y, D
        d = incr(method, y, h, q)
        D += h - dtheta(y, d, q)
        # Kahan summation y <- y + d
        dd = d - c
        t = y + dd
        c = (t - y) - dd
        y = t
    return nmax, y, D


METHODS = {'Euler': (0, 1), 'Implicit Euler': (1, 1), 'Heun': (2, 2), 'RK4': (3, 4),
           'NSFD3': (4, 1), 'NSFD4': (5, 1)}   # NSFD only for q = 2

# antiderivatives G of the delay densities g = -C/f^2 (theory_q.py); L = log(1-y)
G = {
    2: {'Euler': lambda y: np.log(y) + 0.5 * np.log1p(-y),
        'Implicit Euler': lambda y: -np.log(y) - 0.5 * np.log1p(-y),
        'Heun': lambda y: -y**2 / 2 + y / 3 - np.log1p(-y) / 6,
        'RK4': lambda y: (-y**6 / 96 + y**5 / 48 - y**4 / 32 + y**3 / 90 - y**2 / 240 - y / 120
                          - np.log1p(-y) / 120),
        'NSFD3': lambda y: 2 * np.log(y) - 1.5 * np.log1p(-y),
        'NSFD4': lambda y: np.log(y) - 3.5 * np.log1p(-y)},
    3: {'Euler': lambda y: 1.5 * np.log(y) + 0.5 * np.log1p(-y),
        'Implicit Euler': lambda y: -1.5 * np.log(y) - 0.5 * np.log1p(-y),
        'Heun': lambda y: -5 * y**3 / 9 + 5 * y**2 / 12 - y / 6 - np.log1p(-y) / 6,
        'RK4': lambda y: (y**9 / 216 - y**8 / 96 - 17 * y**7 / 1680 + 13 * y**6 / 1440 - y**5 / 600
                          - y**4 / 480 - y**3 / 360 - y**2 / 240 - y / 120 - np.log1p(-y) / 120)},
}


def predicted_delay(name, delta, y, h, q):
    p = METHODS[name][1]
    return h**p * (G[q][name](y) - G[q][name](delta))


def phase_delay(name, delta, h, q, eta=0.5):
    m, _ = METHODS[name]
    nmax = int(10.0 * (-Theta(delta, q)) / h) + 10
    k, yk, D = run_until(m, delta, h, q, eta, nmax)
    return D, yk, k
