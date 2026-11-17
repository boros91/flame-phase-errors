"""Nodal versus common-grid error (Sect. 4.5 and Table S6 of the paper).

The paper reports the maximal error at the computed points ("nodal error"). Here the same runs are also
evaluated on a common, front-resolving set of output times, with the solvers' dense output (SciPy 1.18):
    t in [0, t_end] on 2001 uniform points, plus 4001 points on [c - 40, c + 40] around the exact front
    c = T_delta(1/2), plus 4001 points on [c_num - 40, c_num + 40] around the computed front c_num
    (first computed point with y >= 1/2).
Formulations: standard (rtol = tol, atol = tol/100) and clock w = y^(1-q) (rtol = 2.3e-14, atol = tol),
q = 2 (delta = 1e-4, 1e-6) and q = 3 (delta = 1e-3, 1e-4), tol = 1e-4 and 1e-8, solvers BDF, Radau, LSODA
(RK45 only where t_end <= 2e4).
Writes results/common_grid_scipy.csv; prints the ratio common-grid error / nodal error.
With the argument --refine 4 the numbers of output times are multiplied by 4 (grid-refinement check);
output results/common_grid_scipy_refine4.csv.
The evaluation set does not contain the accepted steps (nodal errors are computed separately); for the clock
formulation the dense output of w is evaluated and then inverted, y = w^(-1/(q-1))."""
import sys
import numpy as np, pandas as pd, warnings, scipy
from scipy.integrate import solve_ivp
from flame_q import Theta, exact
warnings.filterwarnings('ignore')
REF = int(sys.argv[sys.argv.index('--refine') + 1]) if '--refine' in sys.argv else 1


def run(q, d, tol, form, method):
    tend = 2 / ((q - 1) * d**(q - 1))
    if form == 'std':
        F = lambda t, u: u**q * (1 - u); u0 = [d]; rt, at = tol, tol / 100
        toy = lambda u: u
        jac = lambda t, u: np.array([[q * u[0]**(q - 1) - (q + 1) * u[0]**q]])
    else:
        F = lambda t, w: -(q - 1) * (1 - np.maximum(w, 1e-300)**(-1.0 / (q - 1)))
        u0 = [d**(1 - q)]; rt, at = 2.3e-14, tol
        toy = lambda w: np.maximum(w, 1e-300)**(-1.0 / (q - 1))
        jac = lambda t, w: np.array([[-np.maximum(w[0], 1e-300)**(-q / (q - 1.0))]])
    kw = {} if method == 'RK45' else dict(jac=jac)
    s = solve_ivp(F, [0, tend], u0, method=method, rtol=rt, atol=at, dense_output=True, **kw)
    t = s.t; y = toy(s.y[0])
    nodal = np.max(np.abs(y - exact(t, d, q)))
    c = Theta(0.5, q) - Theta(d, q)
    k = np.flatnonzero(y >= 0.5)
    cn = t[k[0]] if len(k) else c
    tg = np.unique(np.concatenate([np.linspace(0, tend, 2000 * REF + 1), np.linspace(c - 40, c + 40, 4000 * REF + 1),
                                   np.linspace(cn - 40, cn + 40, 4000 * REF + 1)]))
    tg = tg[(tg >= 0) & (tg <= t[-1])]
    yg = toy(s.sol(tg)[0])
    eg = np.abs(yg - exact(tg, d, q)); grid = np.max(eg); tmax = tg[np.argmax(eg)]
    done = bool(s.success and t[-1] >= tend * (1 - 1e-10))
    return nodal, grid, s.nfev, len(t), done, tmax - c


rows = []
for q, ds in [(2, [1e-4, 1e-6]), (3, [1e-3, 1e-4])]:
    for d in ds:
        for tol in [1e-4, 1e-8]:
            for m in ['RK45', 'BDF', 'Radau', 'LSODA']:
                if m == 'RK45' and 2 / ((q - 1) * d**(q - 1)) > 2e4:
                    continue
                for form in ['std', 'clock']:
                    nodal, grid, nf, npts, done, tm = run(q, d, tol, form, m)
                    rows.append(dict(q=q, delta=d, tol=tol, solver=m, form=form, nodal_err=nodal, grid_err=grid,
                                     ratio=grid / nodal, t_at_max_minus_c=tm, nfev=nf, npoints=npts, completed=done,
                                     scipy=scipy.__version__))
                    print({k: v for k, v in rows[-1].items() if k != 'scipy'}, flush=True)
R = pd.DataFrame(rows)
R.to_csv('../results/common_grid_scipy.csv' if REF == 1 else f'../results/common_grid_scipy_refine{REF}.csv', index=False)
print('\nratio common-grid / nodal error, by formulation:')
print(R.groupby('form').ratio.describe())
print('\nclock runs: largest common-grid error per q and tol')
print(R[R.form == 'clock'].groupby(['q', 'tol']).grid_err.max())
print('std runs: smallest common-grid error per q and tol')
print(R[R.form == 'std'].groupby(['q', 'tol']).grid_err.min())
