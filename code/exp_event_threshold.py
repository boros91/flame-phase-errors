"""Two checks of Sect. 4.2 of the paper.

(1) Threshold robustness: is the front delay the same at every level of the front?
    MATLAB R2023b runs of matlab/run_delay_profile.m (results/matlab/delay_profile_matlab.csv,
    delta = 1e-4, RelTol = 1e-4, AbsTol = 1e-6, standard formulation): delay D at the first computed
    point with y_k >= level, for level = 0.1, 0.5, 0.9.
(2) Event location: does locating the front with an event function give the delayed time?
    SciPy solve_ivp (RK45, Radau, BDF, LSODA), delta = 1e-4, rtol = 1e-4, atol = 1e-6, event y = 1/2:
    the event time t_e minus the exact time T(1/2), compared with D_front (first computed point with
    y_k >= 1/2) of the same run.

Writes results/threshold_levels.csv and results/event_location.csv."""
import os
import numpy as np, pandas as pd, scipy
from scipy.integrate import solve_ivp
import sys
sys.path.insert(0, os.path.dirname(__file__))
from flame import T

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
delta = 1e-4

# (1) threshold levels, MATLAB
p = pd.read_csv(os.path.join(RES, 'matlab', 'delay_profile_matlab.csv'))
p = p[p.form == 'std']
rows = []
for s, g in p.groupby('solver', sort=False):
    g = g.sort_values('k')
    y, t = g.y.values, g.t.values
    ok = (y > delta * (1 + 1e-12)) & (y < 0.99)
    D = t[ok] - T(y[ok], delta); yo = y[ok]
    r = dict(solver=s)
    for lev in (0.1, 0.5, 0.9):
        k = np.argmax(yo >= lev) if np.any(yo >= lev) else None
        r[f'D_{lev}'] = D[k] if k is not None else np.nan
    r['max_rel_dev'] = max(abs(r['D_0.1'] - r['D_0.5']), abs(r['D_0.9'] - r['D_0.5'])) / abs(r['D_0.5'])
    rows.append(r)
L = pd.DataFrame(rows)
L.to_csv(os.path.join(RES, 'threshold_levels.csv'), index=False)
print('MATLAB R2023b, delta = 1e-4, benchmark tolerance: D at the first point with y_k >= level')
print(L.to_string(index=False, float_format=lambda v: f'{v:.4g}'))

# (2) event location, SciPy
f = lambda t, y: y ** 2 - y ** 3
jac = lambda t, y: np.array([[2 * y[0] - 3 * y[0] ** 2]])
ev = lambda t, y: y[0] - 0.5
ev.terminal = False
t_half = float(T(0.5, delta))
rows = []
for m in ['RK45', 'Radau', 'BDF', 'LSODA']:
    kw = dict(jac=jac) if m != 'RK45' else {}
    sol = solve_ivp(f, [0, 2 / delta], [delta], method=m, rtol=1e-4, atol=1e-6, events=ev, **kw)
    te = sol.t_events[0][0]
    y = sol.y[0]; ok = (y > delta * (1 + 1e-12)) & (y < 0.99)
    D = sol.t[ok] - T(y[ok], delta); k = np.argmax(y[ok] >= 0.5)
    rows.append(dict(solver=m, t_exact=t_half, t_event=te, D_event=te - t_half, D_front=D[k],
                     scipy=scipy.__version__))
E = pd.DataFrame(rows)
E.to_csv(os.path.join(RES, 'event_location.csv'), index=False)
print(f'\nSciPy {scipy.__version__}, delta = 1e-4, rtol = 1e-4, atol = 1e-6, event y = 1/2 (exact time {t_half:.3f})')
print(E.drop(columns='scipy').to_string(index=False, float_format=lambda v: f'{v:.4g}'))
