"""Can the phase error be removed by tightening the tolerances?  Analysis of the MATLAB runs of
matlab/run_tol_sweep_par.m (results/matlab/tol_sweep_matlab.csv; MATLAB R2023b).

Writes
  results/tol_sweep_exponents.csv   fitted exponents of |D_front| in RelTol (delta = 1e-4) and in delta
  results/tol_sweep_needed.csv      largest RelTol (AbsTol = RelTol/100) giving max error <= 1e-2 / 1e-4,
                                    and its number of steps, per solver and delta
  results/tol_sweep_best.csv        smallest max error reached (any RelTol down to 100u) per solver and delta
  results/tol_sweep_clock.csv       the clock remedy (unknown 1/y, RelTol = 100u, AbsTol = tol) from
                                    results/matlab/recip_matlab.csv, for comparison
and prints the numbers quoted in the note."""
import os
import numpy as np, pandas as pd

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
t = pd.read_csv(os.path.join(RES, 'matlab', 'tol_sweep_matlab.csv'))
A = t[t.part == 'A'].copy()
B = t[t.part == 'B'].copy()
SOLV = ['ode45', 'ode23', 'ode23s', 'ode23t', 'ode15s']

# exponents
rows = []
for s in SOLV:
    x = A[(A.solver == s) & (A.delta == 1e-4) & (A.RelTol <= 1e-6) & (A.RelTol >= 1e-11)]
    pr = np.polyfit(np.log10(x.RelTol), np.log10(np.abs(x.D_half)), 1)[0]
    y = A[(A.solver == s) & (A.RelTol == 1e-8) & (A.delta <= 1e-3)]
    pd_ = np.polyfit(np.log10(y.delta), np.log10(np.abs(y.D_half)), 1)[0]
    rows.append(dict(solver=s, exponent_RelTol_delta1e4=pr, exponent_delta_RelTol1e8=pd_))
E = pd.DataFrame(rows); E.to_csv(os.path.join(RES, 'tol_sweep_exponents.csv'), index=False)
print('exponents (|D| ~ RelTol^a at delta = 1e-4, RelTol 1e-6..1e-11;  |D| ~ delta^b at RelTol = 1e-8, delta 1e-3..1e-6)')
print(E.round(2).to_string(index=False))

# needed tolerance
rows = []
for eps in [1e-2, 1e-4]:
    for d in sorted(A.delta.unique(), reverse=True):
        for s in SOLV:
            x = A[(A.delta == d) & (A.solver == s) & (A.maxerr <= eps) & (A.completed == 1)].sort_values('RelTol', ascending=False)
            base = A[(A.delta == d) & (A.solver == s) & (np.isclose(A.RelTol, 1e-4))].iloc[0]
            rows.append(dict(eps=eps, delta=d, solver=s,
                             RelTol=x.RelTol.iloc[0] if len(x) else np.nan,
                             nsteps=x.nsteps.iloc[0] if len(x) else np.nan,
                             nsteps_benchmark=base.nsteps,
                             cost_ratio=x.nsteps.iloc[0] / base.nsteps if len(x) else np.nan))
N = pd.DataFrame(rows); N.to_csv(os.path.join(RES, 'tol_sweep_needed.csv'), index=False)
for eps in [1e-2, 1e-4]:
    print(f'\nlargest RelTol (AbsTol = RelTol/100) for max error <= {eps:g}:  RelTol / steps / steps relative to RelTol = 1e-4')
    x = N[N.eps == eps]
    tab = x.assign(cell=[('none' if np.isnan(r.RelTol) else f'{r.RelTol:.0e} / {int(r.nsteps)} / {r.cost_ratio:.0f}x')
                         for r in x.itertuples()]).pivot(index='delta', columns='solver', values='cell')
    print(tab[SOLV].sort_index(ascending=False).to_string())

# best reachable
rows = []
for d in sorted(A.delta.unique(), reverse=True):
    for s in SOLV:
        x = A[(A.delta == d) & (A.solver == s) & (A.completed == 1)]
        k = x.maxerr.idxmin()
        rows.append(dict(delta=d, solver=s, best_maxerr=x.maxerr.min(), at_RelTol=x.RelTol[k], nsteps=x.nsteps[k],
                         n_incomplete=int(np.sum(A[(A.delta == d) & (A.solver == s)].completed == 0))))
Bst = pd.DataFrame(rows); Bst.to_csv(os.path.join(RES, 'tol_sweep_best.csv'), index=False)
print('\nsmallest max error over all RelTol down to 100u:')
print(Bst.pivot(index='delta', columns='solver', values='best_maxerr')[SOLV].sort_index(ascending=False).map(lambda v: f'{v:.1e}').to_string())
print('\nruns that did not reach t = 2/delta:')
print(A[A.completed == 0][['delta', 'solver', 'RelTol', 'nsteps', 'warning']].to_string(index=False))

# part B: plateau when AbsTol < RelTol * delta
print('\npart B (delta = 1e-4): D_front for AbsTol = RelTol*delta/100 vs AbsTol -> 0 (smallest AbsTol)')
for s in ['ode45', 'ode23s', 'ode15s']:
    for rt in sorted(B.RelTol.unique()):
        x = B[(B.solver == s) & (B.RelTol == rt)].sort_values('AbsTol')
        print(f'  {s:7s} RelTol {rt:.0e}: D(AbsTol=1e-6) {x[np.isclose(x.AbsTol, 1e-6)].D_half.iloc[0]:10.3g}   '
              f'D(AbsTol=1e-20) {x.D_half.iloc[0]:10.3g}   steps {int(x.nsteps.iloc[0])}')

# the clock remedy for comparison (for comparison with Table 9)
r = pd.read_csv(os.path.join(RES, 'matlab', 'recip_matlab.csv'))
C = r[(r.q == 2) & (r.form == 'recip')][['delta', 'tol', 'solver', 'RelTol', 'AbsTol', 'nsteps', 'maxerr', 'D_half']]
C.to_csv(os.path.join(RES, 'tol_sweep_clock.csv'), index=False)
print('\nclock remedy (unknown 1/y, RelTol = 100u, AbsTol = tol):')
print(C[C.solver.isin(SOLV)].sort_values(['delta', 'tol', 'solver']).to_string(index=False))
