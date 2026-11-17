"""Sundman time transformation for y' = y^2(1-y): new independent variable tau with
dt/dtau = 1/rho(y), rho = y (local rate of the slow phase), so that
    dy/dtau = y(1-y)        (logistic, no degeneracy at y = 0)
    dt/dtau = 1/y           (time becomes a dependent variable)
Error control: relative tol on y, ABSOLUTE tol on t (rtol_t tiny), so the phase is controlled.
Compare with the standard formulation at the same tol."""
import numpy as np, pandas as pd, warnings
from scipy.integrate import solve_ivp
from flame import T, exact
warnings.filterwarnings('ignore')

def run_std(d, m, tol):
    kw = dict(rtol=tol, atol=tol / 100, method=m)
    if m in ('Radau', 'BDF', 'LSODA'):
        kw['jac'] = lambda t, y: np.array([[2 * y[0] - 3 * y[0]**2]])
    s = solve_ivp(lambda t, y: y**2 - y**3, (0, 2 / d), [d], **kw)
    return s.t, s.y[0], s.nfev

def run_sund(d, m, tol, absolute_t=True):
    F = lambda tau, u: [u[0] * (1 - u[0]), 1.0 / u[0]]
    J = lambda tau, u: np.array([[1 - 2 * u[0], 0.0], [-1.0 / u[0]**2, 0.0]])
    ev = lambda tau, u: u[1] - 2 / d; ev.terminal = True
    kw = dict(rtol=np.array([tol, 1e-13]) if absolute_t else tol, atol=np.array([tol / 100, tol]), method=m, events=ev)
    if m in ('Radau', 'BDF', 'LSODA'):
        kw['jac'] = J
    s = solve_ivp(F, (0, 10 / d), [d, 0.0], **kw)
    return s.y[1], s.y[0], s.nfev

rows = []
for tol in [1e-4, 1e-6, 1e-8]:
    for d in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        for m in ['RK45', 'Radau', 'BDF', 'LSODA']:
            if m == 'RK45' and d < 1e-4:
                continue
            forms = [('standard', run_std), ('sundman_rel', lambda d, m, tol: run_sund(d, m, tol, False))]
            if m in ('RK45', 'LSODA'):
                forms.append(('sundman_abs', run_sund))
            for form, fn in forms:
                t, y, nf = fn(d, m, tol)
                ok = (y > d * (1 + 1e-12)) & (y < 0.99)
                D = t[ok] - T(y[ok], d)
                Dh = D[np.argmax(y[ok] >= 0.5)]
                rows.append(dict(tol=tol, delta=d, solver=m, form=form, nfev=nf, steps=len(t) - 1,
                                 D_half=Dh, err=np.abs(y - exact(t, d)).max()))
df = pd.DataFrame(rows); df.to_csv('../results/sundman.csv', index=False)
pd.set_option('display.width', 200)
print(df.pivot_table(index=['tol', 'solver', 'delta'], columns='form', values=['err', 'D_half', 'nfev']).to_string(float_format=lambda x: f'{x:.2g}'))
