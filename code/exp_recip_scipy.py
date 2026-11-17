"""y' = y^q (1-y), y(0) = delta, solved in the variable w = y^(1-q) (w' = -(q-1)(1-y)) with SciPy.
In the slow phase w ~ delta^(1-q) - (q-1) t, so w is a clock: an absolute error eps in w is a time
shift eps/(q-1). Three settings per solver:
  std   original equation, rtol = tol, atol = tol/100
  rel   w-equation, rtol = atol = tol         (relative control of w = relative control of y)
  abs   w-equation, rtol = 2.3e-14, atol = tol (absolute control of w ~ absolute control of t)
Front delay D at the first computed point with y >= 1/2, maximal error at the computed points.
Writes results/recip_scipy.csv."""
import numpy as np, pandas as pd, warnings
from scipy.integrate import solve_ivp
from flame_q import Theta, exact
warnings.filterwarnings('ignore')


def run(q, d, tol, form, method):
    tend = 2 / ((q - 1) * d**(q - 1))
    if form == 'std':
        F = lambda t, u: u**q * (1 - u); u0 = [d]; rt, at = tol, tol / 100
        toy = lambda u: u
    else:
        F = lambda t, w: -(q - 1) * (1 - np.maximum(w, 1e-300)**(-1.0 / (q - 1)))
        u0 = [d**(1 - q)]
        toy = lambda w: w**(-1.0 / (q - 1))
        rt, at = (tol, tol) if form == 'rel' else (2.3e-14, tol)
    s = solve_ivp(F, [0, tend], u0, method=method, rtol=rt, atol=at)
    t = s.t; y = toy(s.y[0])
    err = np.max(np.abs(y - exact(t, d, q)))
    ok = (y > d * (1 + 1e-12)) & (y < 0.99)
    D = t[ok] - (Theta(y[ok], q) - Theta(d, q))
    k = np.flatnonzero(y[ok] >= 0.5)
    return (D[k[0]] if len(k) else np.nan), err, s.nfev, bool(s.success and s.t[-1] >= tend * (1 - 1e-10))


rows = []
for q, ds in [(2, [1e-2, 1e-4, 1e-6]), (3, [1e-2, 1e-3, 1e-4])]:
    for tol in [1e-4, 1e-8]:
        for d in ds:
            for m in ['RK45', 'LSODA', 'BDF', 'Radau']:
                if m == 'RK45' and 2 / ((q - 1) * d**(q - 1)) > 2e4:
                    continue
                for form in ['std', 'rel', 'abs']:
                    try:
                        D, e, n, ok = run(q, d, tol, form, m)
                    except Exception:
                        D, e, n, ok = np.nan, np.nan, np.nan, False
                    rows.append(dict(q=q, tol=tol, delta=d, solver=m, form=form, D_half=D, maxerr=e, nfev=n, completed=ok))
                    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('../results/recip_scipy.csv', index=False)
