"""Table for Paper A: standard formulation vs time as a dependent variable (Sundman transform),
LSODA (SciPy), y' = y^q (1-y). Writes results/sundman_table.csv and tables/sundman.tex."""
import numpy as np, pandas as pd, warnings
from scipy.integrate import solve_ivp
from flame_q import T, exact
warnings.filterwarnings('ignore')

def fq(q): return (lambda y: y**q * (1 - y)), (lambda y: q * y**(q - 1) - (q + 1) * y**q)

def measure(t, y, d, q):
    ok = (y > d * (1 + 1e-12)) & (y < 0.99)
    D = t[ok] - T(y[ok], d, q)
    return D[np.argmax(y[ok] >= 0.5)], np.abs(y - exact(t, d, q)).max()

def standard(d, q, tol):
    f, fp = fq(q); tend = 2 / ((q - 1) * d**(q - 1))
    s = solve_ivp(lambda t, y: f(y), (0, tend), [d], method='LSODA', rtol=tol, atol=tol / 100,
                  jac=lambda t, y: np.array([[fp(y[0])]]))
    return (s.t, s.y[0], s.nfev) if s.success else (None, None, s.nfev)

def transformed(d, q, tol, kind):
    f, fp = fq(q); tend = 2 / ((q - 1) * d**(q - 1))
    rho = (lambda y: y**(q - 1)) if kind == 'power' else (lambda y: abs(f(y) / y) + abs(fp(y)))
    ev = lambda tau, u: u[1] - tend; ev.terminal = True
    s = solve_ivp(lambda tau, u: [f(u[0]) / rho(u[0]), 1.0 / rho(u[0])], (0, 1e3 * tend), [d, 0.0],
                  method='LSODA', events=ev, rtol=np.array([tol, 1e-13]), atol=np.array([tol / 100, tol]))
    return s.y[1], s.y[0], s.nfev

rows = []
for q, ds in [(2, [1e-2, 1e-4, 1e-6, 1e-8]), (3, [1e-2, 1e-3, 1e-4])]:
    for d in ds:
        for tol in [1e-4, 1e-8]:
            r = dict(q=q, delta=d, tol=tol)
            t, y, nf = standard(d, q, tol)
            r['std_D'], r['std_err'] = measure(t, y, d, q) if t is not None else (np.nan, np.nan)
            r['std_nfev'] = nf
            for kind in ['power', 'generic']:
                t, y, nf = transformed(d, q, tol, kind)
                r[kind + '_D'], r[kind + '_err'] = measure(t, y, d, q); r[kind + '_nfev'] = nf
            rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows); df.to_csv('../results/sundman_table.csv', index=False)

def sci(x):
    if not np.isfinite(x): return '--'
    a, e = f'{x:.1e}'.split('e'); return f'${a}\\cdot10^{{{int(e)}}}$'
def dl(d):
    e = int(round(np.log10(d))); return f'$10^{{{e}}}$'
lines = []
for q in (2, 3):
    first_q = True
    for _, r in df[df.q == q].iterrows():
        cells = [f'${q}$' if first_q else '', dl(r.delta), f'$10^{{{int(round(np.log10(r.tol)))}}}$',
                 sci(abs(r.std_D)), sci(r.std_err), f'{int(r.std_nfev)}',
                 sci(abs(r.power_D)), sci(r.power_err), f'{int(r.power_nfev)}',
                 sci(r.generic_err), f'{int(r.generic_nfev)}']
        lines.append(' & '.join(cells) + r' \\'); first_q = False
    lines.append(r'\midrule')
open('../tables/sundman.tex', 'w').write('\n'.join(lines[:-1]) + '\n')
