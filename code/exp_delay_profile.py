"""How the phase delay accumulates along a trajectory, delta = 1e-4 (q = 2).

(1) Fixed-step methods (explicit Euler, implicit Euler, Heun, RK4), h = 0.25: exact D_k at every
    step (Kahan-compensated iterates, D accumulated step by step as in exp_theorem.py), and the
    leading term h^p [G(y_k) - G(delta)] of Theorem 3.4.
(2) SciPy solvers at Moler's tolerances (RelTol = 1e-4, AbsTol = 1e-6): D_k at the accepted steps;
    LSODA and BDF also with time as a dependent variable in (ln y, t), absolute tolerance 1e-4.
    (The MATLAB counterpart is matlab/run_delay_profile.m.)
Writes results/delay_profile_fixed.csv and results/delay_profile_scipy.csv."""
import numpy as np, pandas as pd, warnings
from numba import njit
from scipy.integrate import solve_ivp
from flame_q import incr, dtheta, Theta, predicted_delay, METHODS
warnings.filterwarnings('ignore')
q, d = 2, 1e-4


@njit(cache=True)
def profile(method, delta, h, q, eta, nmax):
    ys = np.empty(nmax); Ds = np.empty(nmax)
    y = delta; c = 0.0; D = 0.0
    for k in range(nmax):
        ys[k] = y; Ds[k] = D
        if y >= eta:
            return ys[:k + 1], Ds[:k + 1]
        dd_ = incr(method, y, h, q)
        D += h - dtheta(y, dd_, q)
        dd = dd_ - c; t = y + dd; c = (t - y) - dd; y = t
    return ys, Ds


rows = []
h = 0.25
for name in ['Euler', 'Implicit Euler', 'Heun', 'RK4']:
    ys, Ds = profile(METHODS[name][0], d, h, q, 0.5, 10**7)
    # keep about 400 points, equally spaced in log y, plus the last one
    idx = np.unique(np.r_[np.searchsorted(ys, np.logspace(np.log10(d), np.log10(ys[-1]), 400)).clip(0, len(ys) - 1), len(ys) - 1])
    for i in idx:
        rows.append(dict(method=name, h=h, k=i, y=ys[i], D=Ds[i], pred=predicted_delay(name, d, ys[i], h, q)))
    print(f'{name:15s} steps={len(ys) - 1}  D_final={Ds[-1]: .4e}  pred={predicted_delay(name, d, ys[-1], h, q): .4e}')
pd.DataFrame(rows).to_csv('../results/delay_profile_fixed.csv', index=False)

rows = []
tend = 2 / d
for m in ['RK45', 'LSODA', 'BDF', 'Radau']:
    s = solve_ivp(lambda t, y: y**2 - y**3, [0, tend], [d], method=m, rtol=1e-4, atol=1e-6)
    t, y = s.t, s.y[0]
    ok = (y < 0.99)
    for k in np.flatnonzero(ok):
        D = t[k] - (Theta(y[k], q) - Theta(d, q)) if y[k] > d else np.nan
        rows.append(dict(solver=m, form='std', k=k, t=t[k], y=y[k], D=D))
    print(m, 'std steps', len(t) - 1)
# time as a dependent variable, unknowns (ln y, t), rho = y, absolute tolerance 1e-4 on both
# (form 'sundlog' of matlab/run_one_sundman.m)
for m in ['LSODA', 'BDF']:
    ev = lambda tau, u: u[1] - tend; ev.terminal = True; ev.direction = 1
    G = lambda tau, u: [1 - np.exp(u[0]), np.exp(-u[0])]   # d(ln y)/dtau = f/(y rho), dt/dtau = 1/rho
    s = solve_ivp(G, [0, 10 * tend + 100], [np.log(d), 0.0], method=m, rtol=2.3e-14, atol=[1e-4, 1e-4], events=ev)
    t, y = s.y[1], np.exp(s.y[0])
    for k in np.flatnonzero(y < 0.99):
        D = t[k] - (Theta(y[k], q) - Theta(d, q)) if y[k] > d else np.nan
        rows.append(dict(solver=m, form='sundlog', k=k, t=t[k], y=y[k], D=D))
    print(m, 'sundlog steps', len(t) - 1)
pd.DataFrame(rows).to_csv('../results/delay_profile_scipy.csv', index=False)
