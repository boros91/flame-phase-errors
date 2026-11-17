"""Experiments for the second example, cubic autocatalysis y' = y^3 (1-y), y(0) = delta,
t in [0, 1/delta^2]:  adaptive solvers (with and without warm start) and sinh-mapped
collocation (pinned vs IVP form).  Writes results/q3_solvers.csv, results/q3_colloc.csv."""
import numpy as np, csv, warnings
from scipy.integrate import solve_ivp
from flame_q import T, exact, Theta
from collocation_q import solve as csolve, error as cerror, interp_error

warnings.filterwarnings('ignore')
q = 3
rhs = lambda t, y: y**3 * (1 - y)
jac = lambda t, y: np.array([[3 * y[0]**2 - 4 * y[0]**3]])


def metrics(t, y, d):
    e = np.abs(y - exact(t, d, q))
    ok = (y > d * (1 + 1e-12)) & (y < 0.99)
    D = t[ok] - T(y[ok], d, q)
    return e.max(), (np.abs(D).max() if ok.any() else np.nan)


rows = []
for d in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    for rtol, atol in [(1e-8, 1e-12), (1e-12, 1e-24)]:
        for name in ['RK45', 'DOP853', 'Radau', 'BDF', 'LSODA', 'Radau (warm)']:
            meth = name.split()[0]; warm = 'warm' in name
            if meth in ('RK45', 'DOP853') and d < 1e-3:
                continue
            y0, t0 = d, 0.0
            if warm and d < 0.05:
                y0 = 0.05; t0 = T(0.05, d, q)
            kw = dict(rtol=rtol, atol=atol, method=meth)
            if meth in ('Radau', 'BDF', 'LSODA'):
                kw['jac'] = jac
            sol = solve_ivp(rhs, (t0, 1 / d**2), [y0], **kw)
            E, Dm = metrics(sol.t, sol.y[0], d)
            rows.append(dict(delta=d, rtol=rtol, atol=atol, solver=name, maxerr=E, maxD=Dm,
                             steps=len(sol.t) - 1, status=sol.status))
            print(rows[-1], flush=True)
with open('../results/q3_solvers.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

rows = []
for d in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    for N in [32, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512]:
        sp = csolve(d, N, q, pin=True)
        si = csolve(d, N, q, pin=False)
        rows.append(dict(delta=d, N=N, L=sp['L'], interp=interp_error(d, N, q),
                         pin=cerror(sp), pin_its=sp['its'], pin_conv=int(sp['converged']),
                         y0_relerr=abs(sp['y'][-1] / d - 1), pin_cond=sp['cond'],
                         ivp=cerror(si), ivp_its=si['its'], ivp_conv=int(si['converged']),
                         ivp_cond=si['cond']))
        print({k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
with open('../results/q3_colloc.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
