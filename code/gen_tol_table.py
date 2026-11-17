"""Table 8 of the paper (Sect. 4.4; r_min = 100*eps, the smallest RelTol MATLAB accepts): the price of tightening the tolerances.
For a maximal error <= 1e-2: the largest RelTol (AbsTol = RelTol/100) that achieves it, the number of
accepted steps, and in brackets the steps relative to the benchmark RelTol = 1e-4.
Reads results/tol_sweep_needed.csv (code/gen_tol_sweep.py); writes tables/tol_needed_v24.tex."""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(__file__)
N = pd.read_csv(os.path.join(HERE, '..', 'results', 'tol_sweep_needed.csv'))
N = N[np.isclose(N.eps, 1e-2)]
SOLV = ['ode45', 'ode23', 'ode23s', 'ode23t', 'ode15s']


def tex_pow(v):
    if np.isclose(v, 2.220446049250313e-14, rtol=1e-6, atol=0):
        return '$r_{\\min}$'
    return f'$10^{{{int(round(np.log10(v)))}}}$'


def ratio(r):
    return f'{r:.1f}' if r < 10 else f'{r:.0f}'


lines = []
for d in sorted(N.delta.unique(), reverse=True):
    cells = [tex_pow(d)]
    for s in SOLV:
        r = N[(N.delta == d) & (N.solver == s)].iloc[0]
        cells.append('none' if np.isnan(r.RelTol) else
                     f'{tex_pow(r.RelTol)}: {int(r.nsteps)} ({ratio(r.cost_ratio)})')
    lines.append(' & '.join(cells) + r'\\')
out = os.path.join(HERE, '..', 'tables', 'tol_needed_v24.tex')
with open(out, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
