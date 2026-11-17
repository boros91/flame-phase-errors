"""Moler's experiment revisited with SciPy: flame problem, RelTol = 1e-4, AbsTol = 1e-6
(Moler 2014 used RelTol = 1e-4 with MATLAB's default AbsTol = 1e-6).
For every run: total steps, steps in the slow phase (t < T(1/2) - 20), steps in the stiff
phase (t > T(1/2) + 20), predicted stiff-phase steps of explicit RK (length / stability
boundary), maximal error and phase delay.  Writes results/moler_scipy.csv and the step-size
histories for delta = 1e-4 (results/moler_steps_1e-4.npz)."""
import numpy as np, csv, time, warnings
from scipy.integrate import solve_ivp
from flame import T, exact

warnings.filterwarnings('ignore')
f = lambda t, y: y**2 - y**3
jac = lambda t, y: np.array([[2 * y[0] - 3 * y[0]**2]])
BETA = {'RK45': 3.3066, 'RK23': 2.5127, 'DOP853': 6.3937}   # real stability boundaries
rows, hist = [], {}
for d in [2e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    th = T(0.5, d)
    for m in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        for warm in (False, True):
            if warm and (m not in ('RK45', 'Radau', 'BDF') or d > 1e-2):
                continue
            t0, y0 = (T(0.01, d), 0.01) if warm else (0.0, d)
            kw = dict(rtol=1e-4, atol=1e-6, method=m)
            if m in ('Radau', 'BDF', 'LSODA'):
                kw['jac'] = jac
            tic = time.time()
            s = solve_ivp(f, (t0, 2 / d), [y0], **kw)
            el = time.time() - tic
            t, y = s.t, s.y[0]
            ok = (y > d * (1 + 1e-12)) & (y < 0.99)
            D = t[ok] - T(y[ok], d)
            Dh = D[np.argmax(y[ok] >= 0.5)]
            name = m + (' (warm)' if warm else '')
            r = dict(delta=d, solver=name, steps=len(t) - 1, slow=int((t[1:] < th - 20).sum()),
                     stiff=int((t[1:] > th + 20).sum()),
                     pred_stiff=(2 / d - th) / BETA[m] if m in BETA else np.nan,
                     nfev=s.nfev, maxerr=np.abs(y - exact(t, d)).max(), D_half=Dh,
                     maxD=np.abs(D).max(), sec=round(el, 2))
            rows.append(r)
            print({k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if d == 1e-4 and not warm:
                hist[m] = np.vstack([t[1:], np.diff(t)])
with open('../results/moler_scipy.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
np.savez('../results/moler_steps_1e-4.npz', **hist)
