"""Explicit Euler applied to y' = y^q (1-y), y(0) = delta, written in three dependent variables:
    w = y            (the standard formulation),
    w = ln y         (w' = y^(q-1) (1-y)),
    w = y^(1-q)      (w' = -(q-1) (1-y)),
and the exact phase delay D at the first step with y_k >= 1/2.

Leading term (same computation as for the standard formulation, Section 3): for Euler applied to
w' = F(w) the time defect of one step is (h^2/2) F'(w) + O(h^3), hence
    D ~ (h/2) ln( F(w_n) / F(w_0) ),   with F(w(y)) = phi'(y) f_q(y)  for w = phi(y),
i.e. (h/2) ln[f_q(y)/f_q(delta)], (h/2) ln[y^(q-1)(1-y)/(delta^(q-1)(1-delta))] and
(h/2) ln[(1-y)/(1-delta)] for the three variables.

Writes results/euler_variables.csv (Remark 3.6 of the paper)."""
import numpy as np, pandas as pd
from numba import njit
from flame_q import dtheta, run_until


@njit(cache=True)
def run_w(var, delta, h, q, eta, nmax):
    """Euler in w = ln y (var = 1) or w = y^(1-q) (var = 2). w is accumulated with Kahan
    summation of the increments; D_k = sum_j [h - (Theta(y_{j+1}) - Theta(y_j))] exactly."""
    if var == 1:
        w = np.log(delta)
    else:
        w = delta ** (1 - q)
    c = 0.0; y = delta; D = 0.0
    for k in range(nmax):
        if y >= eta:
            return k, y, D
        if var == 1:
            dw = h * y ** (q - 1) * (1.0 - y)
            d = y * np.expm1(dw)                                   # y_{k+1} - y_k
        else:
            dw = -h * (q - 1) * (1.0 - y)
            d = y * np.expm1(-np.log1p(dw / w) / (q - 1))
        D += h - dtheta(y, d, q)
        dd = dw - c                                                # Kahan: w <- w + dw
        t = w + dd
        c = (t - w) - dd
        w = t
        if var == 1:
            y = np.exp(w)
        else:
            y = w ** (-1.0 / (q - 1))
    return nmax, y, D


def leading(var, delta, y, h, q):
    if var == 0:
        Fy, F0 = y**q * (1 - y), delta**q * (1 - delta)
    elif var == 1:
        Fy, F0 = y**(q - 1) * (1 - y), delta**(q - 1) * (1 - delta)
    else:
        Fy, F0 = 1 - y, 1 - delta
    return 0.5 * h * np.log(Fy / F0)


NAMES = {0: 'y', 1: 'ln y', 2: 'y^(1-q)'}
rows = []
for q, deltas in [(2, [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]), (3, [1e-2, 1e-3, 1e-4])]:
    for h in (0.25, 0.125):
        for d in deltas:
            nmax = int(10 * (d ** (1 - q) / (q - 1) + np.log(1 / d)) / h) + 10
            for var in (0, 1, 2):
                if var == 0:
                    k, yk, D = run_until(0, d, h, q, 0.5, nmax)
                else:
                    k, yk, D = run_w(var, d, h, q, 0.5, nmax)
                L = leading(var, d, yk, h, q)
                rows.append(dict(q=q, h=h, delta=d, variable=NAMES[var], steps=k, y_k=yk,
                                 D=D, leading=L, ratio=D / L, remainder_over_h2=(D - L) / h**2))
df = pd.DataFrame(rows)
df.to_csv('../results/euler_variables.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(float_format=lambda x: f'{x:.4g}'))

# summary quoted in the paper (Remark 3.6)
print('\nq = 2, h = 0.25:'); print(df[(df.q == 2) & (df.h == 0.25) & df.delta.isin([1e-2, 1e-4, 1e-6])][['variable', 'delta', 'D', 'leading']].to_string(index=False))
print('ratio D/leading: min %.4f, max %.4f' % (df.ratio.min(), df.ratio.max()))
print('max |remainder/h^2| = %.3f' % df.remainder_over_h2.abs().max())
