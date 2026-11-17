"""Best constant time shift: for a numerical solution (t_k, y_k) find
    tau* = argmin_tau max_k |y_k - y_exact(t_k - tau)|
and compare the error before and after the shift with the phase delays D_k = t_k - T(y_k).
q = 2.  Adaptive SciPy solvers at the benchmark tolerance and fixed-step methods."""
import numpy as np, pandas as pd, warnings
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from flame import T, exact
from flame_q import incr

warnings.filterwarnings('ignore')
f = lambda t, y: y**2 - y**3
jac = lambda t, y: np.array([[2 * y[0] - 3 * y[0]**2]])


@njit(cache=True)
def trajectory(method, delta, h, tend):
    n = int(tend / h)
    y = np.empty(n + 1); y[0] = delta; c = 0.0; yy = delta
    for k in range(n):
        d = incr(method, yy, h, 2)
        dd = d - c; t = yy + dd; c = (t - yy) - dd; yy = t
        y[k + 1] = yy
    return y


def analyse(t, y, d, name):
    err0 = np.abs(y - exact(t, d)).max()
    ok = (y > d * (1 + 1e-12)) & (y < 1 - 1e-9)
    D = t[ok] - T(y[ok], d); yo = y[ok]
    Dat = lambda lev: D[np.argmax(yo >= lev)] if (yo >= lev).any() else np.nan
    c = Dat(0.5)
    win = np.abs(t - (T(0.5, d) + c)) < 300 + 2 * abs(c)
    tw, yw = t[win], y[win]
    E = lambda tau: np.abs(yw - exact(tw - tau, d)).max()
    grid = c + np.linspace(-5, 5, 201) * max(1.0, abs(c) * 0.05)
    tau0 = grid[np.argmin([E(g) for g in grid])]
    w = grid[1] - grid[0]
    r = minimize_scalar(E, bounds=(tau0 - 2 * w, tau0 + 2 * w), method='bounded', options=dict(xatol=1e-12))
    return dict(delta=d, method=name, err=err0, D_01=Dat(0.1), D_05=c, D_09=Dat(0.9),
                tau_best=r.x, err_shifted=r.fun, ratio=err0 / r.fun)


rows = []
for d in [1e-2, 1e-4, 1e-6]:
    for m in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        if m in ('RK23', 'RK45', 'DOP853') and d < 1e-4:
            continue
        kw = dict(rtol=1e-4, atol=1e-6, method=m)
        if m in ('Radau', 'BDF', 'LSODA'):
            kw['jac'] = jac
        s = solve_ivp(f, (0, 2 / d), [d], **kw)
        rows.append(analyse(s.t, s.y[0], d, m + ' (1e-4)'))
    for name, code, h in [('Euler', 0, 0.25), ('impl. Euler', 1, 0.25), ('Heun', 2, 0.25), ('RK4', 3, 0.25)]:
        tend = T(0.999, d) + 50
        y = trajectory(code, d, h, tend)
        t = h * np.arange(len(y))
        rows.append(analyse(t, y, d, f'{name} h={h}'))
df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(df.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
df.to_csv('../results/bestshift.csv', index=False)
