"""Experiment 2: adaptive solvers (scipy) on the flame problem.
For every accepted step we compute the exact phase delay D_k = t_k - T_delta(y_k).
Also: 'warm start' variant that integrates analytically up to y = 0.01 and
starts the solver there (translation property)."""
import numpy as np, csv, time, warnings, sys
from scipy.integrate import solve_ivp
from flame import exact, T

warnings.filterwarnings('ignore')
rhs = lambda t, y: y**2 - y**3
jac = lambda t, y: np.array([[2 * y[0] - 3 * y[0]**2]])
tols = [(1e-3, 1e-6), (1e-8, 1e-12), (1e-12, 1e-20)]
deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
solvers = ['RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']


def metrics(t, y, d):
    e = np.abs(y - exact(t, d))
    ok = (y > d * (1 + 1e-12)) & (y < 0.99)
    D = t[ok] - T(y[ok], d)
    yy = y[ok]
    i = np.searchsorted(yy, 0.5) if np.all(np.diff(yy) > 0) else np.argmax(yy >= 0.5)
    Dhalf = D[min(i, len(D) - 1)] if len(D) else np.nan
    return e.max(), Dhalf, np.abs(D).max() if len(D) else np.nan


def run(solver, d, rtol, atol, warm=False):
    t0, y0 = 0.0, d
    if warm:
        y0 = 0.01 if d < 0.01 else d
        t0 = T(y0, d)
    kw = dict(rtol=rtol, atol=atol, method=solver)
    if solver in ('Radau', 'BDF', 'LSODA'):
        kw['jac'] = jac
    tic = time.time()
    sol = solve_ivp(rhs, (t0, 2 / d), [y0], **kw)
    el = time.time() - tic
    t, y = sol.t, sol.y[0]
    if warm:   # prepend the analytic part (exact), measure errors on solver steps only
        pass
    E, Dh, Dm = metrics(t, y, d)
    return dict(solver=solver + (' (warm)' if warm else ''), delta=d, rtol=rtol, atol=atol,
                maxerr=E, D_half=Dh, maxD=Dm, steps=len(t) - 1, nfev=sol.nfev,
                status=sol.status, sec=round(el, 2))


if __name__ == '__main__':
    rows = []
    for d in deltas:
        for rtol, atol in tols:
            for s in solvers + ['RK45w', 'Radauw']:
                warm = s.endswith('w')
                name = s[:-1] if warm else s
                if name in ('RK45', 'DOP853') and d < 1e-5:
                    continue          # >1e6 steps in pure python; skipped
                r = run(name, d, rtol, atol, warm)
                rows.append(r)
                print(f"{r['solver']:13s} d={d:.0e} rtol={rtol:.0e} maxerr={r['maxerr']:.2e} "
                      f"D(1/2)={r['D_half']: .2e} max|D|={r['maxD']:.2e} steps={r['steps']} "
                      f"nfev={r['nfev']} t={r['sec']}s", flush=True)
    with open('../results/solver_table.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
