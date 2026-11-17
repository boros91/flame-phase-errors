"""Sundman transformation with a generic rate function rho(y) = |f(y)|/|y| + |f'(y)| (no knowledge
of the degeneracy order), absolute control of t. q = 2 and q = 3."""
import numpy as np, pandas as pd, warnings
from scipy.integrate import solve_ivp
from flame_q import T, exact
warnings.filterwarnings('ignore')

def run(d, q, m, tol, rho_kind):
    f = lambda y: y**q * (1 - y)
    fp = lambda y: q * y**(q - 1) - (q + 1) * y**q
    rho = (lambda y: abs(f(y)) / abs(y) + abs(fp(y))) if rho_kind == 'generic' else (lambda y: y**(q - 1))
    F = lambda tau, u: [f(u[0]) / rho(u[0]), 1.0 / rho(u[0])]
    tend = 2 / ((q - 1) * d**(q - 1))
    ev = lambda tau, u: u[1] - tend; ev.terminal = True
    s = solve_ivp(F, (0, 1e3 * tend), [d, 0.0], method=m, events=ev,
                  rtol=np.array([tol, 1e-13]), atol=np.array([tol / 100, tol]))
    t, y = s.y[1], s.y[0]
    ok = (y > d * (1 + 1e-12)) & (y < 0.99)
    D = t[ok] - T(y[ok], d, q)
    return dict(q=q, delta=d, solver=m, rho=rho_kind, tol=tol, nfev=s.nfev,
                D_half=D[np.argmax(y[ok] >= 0.5)], err=np.abs(y - exact(t, d, q)).max(), t_end=t[-1] / tend)

rows = []
for q, ds in [(2, [1e-2, 1e-4, 1e-6, 1e-8]), (3, [1e-1, 1e-2, 1e-3, 1e-4])]:
    for d in ds:
        for tol in [1e-4, 1e-8]:
            for rk in ['power', 'generic']:
                rows.append(run(d, q, 'LSODA', tol, rk))
df = pd.DataFrame(rows); df.to_csv('../results/sundman_rho.csv', index=False)
pd.set_option('display.width', 200)
print(df.to_string(index=False, float_format=lambda x: f'{x:.2g}'))
