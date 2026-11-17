"""Time as a dependent variable with a separate tolerance on t: sweep of the absolute tolerance on t
for SciPy solvers that accept component-wise tolerances (RK23, RK45, DOP853, LSODA).
q = 2, rho = y. y-tolerances fixed: rtol_y = 1e-6, atol_y = 1e-8."""
import numpy as np, pandas as pd, warnings
from scipy.integrate import solve_ivp
from flame_q import T, exact
warnings.filterwarnings('ignore')
q = 2

def run(d, m, rtol_t, atol_t, form):
    tend = 2 / d
    if form == 'standard':
        kw = dict(rtol=1e-6, atol=1e-8, method=m)
        if m == 'LSODA': kw['jac'] = lambda t, y: np.array([[2 * y[0] - 3 * y[0]**2]])
        s = solve_ivp(lambda t, y: y**2 - y**3, (0, tend), [d], **kw)
        t, y = s.t, s.y[0]
    else:
        ev = lambda tau, u: u[1] - tend; ev.terminal = True
        s = solve_ivp(lambda tau, u: [u[0] * (1 - u[0]), 1.0 / u[0]], (0, 1e3 * tend), [d, 0.0], method=m,
                      events=ev, rtol=np.array([1e-6, rtol_t]), atol=np.array([1e-8, atol_t]))
        t, y = s.y[1], s.y[0]
    ok = (y > d * (1 + 1e-12)) & (y < 0.99)
    D = t[ok] - T(y[ok], d, q)
    return dict(delta=d, solver=m, form=form, rtol_t=rtol_t, atol_t=atol_t, nfev=s.nfev,
                D_half=D[np.argmax(y[ok] >= 0.5)], err=np.abs(y - exact(t, d, q)).max())

rows = []
for d in [1e-4, 1e-6]:
    for m in ['RK23', 'RK45', 'DOP853', 'LSODA']:
        if m != 'LSODA' and d < 1e-4:
            continue
        rows.append(run(d, m, np.nan, np.nan, 'standard'))
        rows.append(run(d, m, 1e-6, 1e-8, 'tau, same tol on t'))
        for at in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
            rows.append(run(d, m, 1e-13, at, 'tau, absolute tol on t'))
        print(m, d, flush=True)
df = pd.DataFrame(rows); df.to_csv('../results/ttol_sweep.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False, float_format=lambda x: f'{x:.2g}'))
