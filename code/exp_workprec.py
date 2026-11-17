"""Work-precision study for the flame problem (q = 2) with SciPy.
Three strategies for every solver and tolerance tol in 10^-3 ... 10^-12:
  standard : RelTol = tol,           AbsTol = tol/100      (the benchmark ratio)
  scaled   : RelTol = tol*delta,     AbsTol = tol*delta^2/100
  warm     : start at y = 0.01 on the exact translate, RelTol = tol, AbsTol = tol/100
Error = max over the accepted steps and over a dense grid across the front (dense output),
cost = number of f evaluations (and Jacobian evaluations / LU decompositions for implicit).
Writes results/workprec.csv."""
import numpy as np, pandas as pd, warnings, sys
from scipy.integrate import solve_ivp
from flame import T, exact

warnings.filterwarnings('ignore')
f = lambda t, y: y**2 - y**3
jac = lambda t, y: np.array([[2 * y[0] - 3 * y[0]**2]])
SOLVERS = ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']
TOLS = [10.0**(-k) for k in range(3, 13)]


def run(d, m, tol, strategy):
    if strategy == 'standard':
        t0, y0, rt, at = 0.0, d, tol, tol / 100
    elif strategy == 'scaled':
        t0, y0, rt, at = 0.0, d, tol * d, tol * d**2 / 100
    else:
        t0, y0, rt, at = T(0.01, d), 0.01, tol, tol / 100
    rt = max(rt, 2.3e-14)                       # SciPy lower bound on rtol
    kw = dict(rtol=rt, atol=at, method=m, dense_output=True)
    if m in ('Radau', 'BDF', 'LSODA'):
        kw['jac'] = jac
    s = solve_ivp(f, (t0, 2 / d), [y0], **kw)
    if not s.success:
        return None
    c = T(0.5, d)
    tg = np.concatenate([s.t, np.linspace(max(t0, c - 40), min(2 / d, c + 40), 4001)])
    yg = s.sol(tg)[0]
    err = np.abs(yg - exact(tg, d)).max()
    t, y = s.t, s.y[0]
    err_steps = np.abs(y - exact(t, d)).max()
    ok = (y > d * (1 + 1e-12)) & (y < 0.99) & (y > y0 * (1 + 1e-12) if strategy == 'warm' else True)
    D = t[ok] - T(y[ok], d)
    return dict(delta=d, solver=m, strategy=strategy, tol=tol, rtol=rt, atol=at, steps=len(t) - 1,
                nfev=s.nfev, njev=s.njev, nlu=s.nlu, error=err, error_steps=err_steps,
                D_half=D[np.argmax(y[ok] >= 0.5)] if ok.any() else np.nan)


if __name__ == '__main__':
    rows = []
    for d in [1e-4, 1e-6]:
        for m in SOLVERS:
            if m in ('RK23', 'RK45', 'DOP853') and d < 1e-4:
                continue                          # stiff phase too long for explicit codes
            for strat in ['standard', 'scaled', 'warm']:
                for tol in TOLS:
                    r = run(d, m, tol, strat)
                    if r:
                        rows.append(r)
                        print(d, m, strat, f'{tol:.0e}', r['steps'], r['nfev'], f"{r['error']:.2e}",
                              f"{r['D_half']:.2e}", flush=True)
                pd.DataFrame(rows).to_csv('../results/workprec.csv', index=False)
