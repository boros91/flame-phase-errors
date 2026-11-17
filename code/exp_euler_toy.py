"""Adaptive explicit Euler on the flame problem y' = y^2 - y^3, y(0) = delta: how the phase delay
accumulates, in closed form.

Two step-size rules, each in an 'ideal' version (the leading local error term is set exactly
equal to the tolerance) and a 'practical' version (Euler propagated, Heun as error estimator,
standard accept/reject controller):

  std   : relative error control on y.  Ideal step h = sqrt(2 tol y / (f'(y) f(y))), so the
          local error is tol*y and the time defect tol*y/f(y) ~ tol/y per step.
          Prediction: D = sum_k tol/y_k with y_{k+1} ~ (1+sqrt(tol)) y_k, i.e.
              D ~ sqrt(tol)/delta,   fraction accumulated by level y: 1 - delta/y.
  time  : time as a dependent variable (rho = y: dy/dtau = y(1-y), dt/dtau = 1/y), absolute
          control of t.  Ideal step k = sqrt(2 tol y/(1-y)), so the local error of t is tol;
          each step adds a delay ~ 2 tol, and there are N ~ sqrt(2/(tol delta)) steps:
              D ~ sqrt(8 tol / delta).

All delays are exact: D_k = t_k - T_delta(y_k).  Integration stops at the first y_k >= 1/2.
Writes results/euler_toy.csv and results/euler_toy_profile.csv."""
import numpy as np, pandas as pd
from numba import njit
from flame_q import T


@njit(cache=True)
def std_ideal(d, tol, nmax):
    y = d; t = 0.0; ys = np.empty(nmax); ts = np.empty(nmax); n = 0
    ys[0] = y; ts[0] = t
    while y < 0.5 and n < nmax - 1:
        f = y * y * (1 - y); fp = 2 * y - 3 * y * y
        h = np.sqrt(2 * tol * y / (fp * f))
        y = y + h * f; t = t + h; n += 1; ys[n] = y; ts[n] = t
    return ys[:n + 1], ts[:n + 1]


@njit(cache=True)
def std_practical(d, tol, nmax):
    y = d; t = 0.0; ys = np.empty(nmax); ts = np.empty(nmax); n = 0; nrej = 0
    ys[0] = y; ts[0] = t
    h = np.sqrt(tol) / y
    while y < 0.5 and n < nmax - 1:
        f = y * y * (1 - y)
        ye = y + h * f                                   # Euler (propagated)
        f1 = ye * ye * (1 - ye)
        err = abs(0.5 * h * (f1 - f))                    # |Heun - Euler|
        sc = tol * abs(y)
        if err <= sc:
            y = ye; t = t + h; n += 1; ys[n] = y; ts[n] = t
        else:
            nrej += 1
        h = h * min(5.0, max(0.2, 0.9 * np.sqrt(sc / max(err, 1e-300))))
    return ys[:n + 1], ts[:n + 1]


@njit(cache=True)
def time_ideal(d, tol, nmax):
    y = d; t = 0.0; ys = np.empty(nmax); ts = np.empty(nmax); n = 0
    ys[0] = y; ts[0] = t
    while y < 0.5 and n < nmax - 1:
        k = np.sqrt(2 * tol * y / (1 - y))
        y, t = y + k * y * (1 - y), t + k / y
        n += 1; ys[n] = y; ts[n] = t
    return ys[:n + 1], ts[:n + 1]


@njit(cache=True)
def time_practical(d, tol, nmax):
    y = d; t = 0.0; ys = np.empty(nmax); ts = np.empty(nmax); n = 0
    ys[0] = y; ts[0] = t
    k = np.sqrt(2 * tol * y)
    while y < 0.5 and n < nmax - 1:
        fy = y * (1 - y); ft = 1.0 / y
        ye = y + k * fy; te = t + k * ft                 # Euler (propagated)
        ey = abs(0.5 * k * (ye * (1 - ye) - fy))         # |Heun - Euler|, y component
        et = abs(0.5 * k * (1.0 / ye - ft))              # |Heun - Euler|, t component
        r = max(ey / (tol * abs(y)), et / tol)           # relative on y, ABSOLUTE on t
        if r <= 1.0:
            y = ye; t = te; n += 1; ys[n] = y; ts[n] = t
        k = k * min(5.0, max(0.2, 0.9 * np.sqrt(1.0 / max(r, 1e-300))))
    return ys[:n + 1], ts[:n + 1]


RUNS = {'std_ideal': std_ideal, 'std_practical': std_practical,
        'time_ideal': time_ideal, 'time_practical': time_practical}

if __name__ == '__main__':
    rows = []
    for tol in [1e-4, 1e-6]:
        for d in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            for name, fun in RUNS.items():
                ys, ts = fun(d, tol, 20_000_000)
                D = ts[-1] - T(np.array([ys[-1]]), d, 2)[0]
                pred = np.sqrt(tol) / d if name.startswith('std') else np.sqrt(8 * tol / d)
                rows.append(dict(method=name, delta=d, tol=tol, D=D, pred=pred, ratio=D / pred,
                                 steps=len(ys) - 1, y_end=ys[-1]))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('../results/euler_toy.csv', index=False)

    # accumulation profile: fraction of the final delay reached at level y
    prof = []
    tol = 1e-4
    for d in [1e-3, 1e-4, 1e-5]:
        for name in ['std_ideal', 'std_practical', 'time_ideal']:
            ys, ts = RUNS[name](d, tol, 20_000_000)
            D = ts - T(ys, d, 2)
            idx = np.unique(np.linspace(0, len(ys) - 1, 400).astype(int))
            for i in idx:
                prof.append(dict(method=name, delta=d, tol=tol, y=ys[i], D=D[i], frac=D[i] / D[-1]))
    pd.DataFrame(prof).to_csv('../results/euler_toy_profile.csv', index=False)
