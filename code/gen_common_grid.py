"""Summary of the nodal versus common-grid error check (Sect. 4.5, Table S6 of the paper).
Reads results/matlab/common_grid_matlab.csv (matlab/run_common_grid.m, MATLAB R2023b, deval) and
results/common_grid_scipy.csv (code/exp_common_grid.py, SciPy dense output).
Writes results/common_grid_summary.csv and prints the numbers quoted in the paper."""
import os
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
M = pd.read_csv(os.path.join(RES, 'matlab', 'common_grid_matlab.csv'))
M['ratio'] = M.grid_err / M.nodal_err
S = pd.read_csv(os.path.join(RES, 'common_grid_scipy.csv'))
S['form'] = S.form.replace({'clock': 'recip'})
rows = []
for src, T in (('MATLAB', M), ('SciPy', S)):
    for form, g in T.groupby('form'):
        rows.append(dict(source=src, form=form, runs=len(g), completed=int(g.completed.sum()),
                         ratio_min=g.ratio.min(), ratio_median=g.ratio.median(), ratio_max=g.ratio.max(),
                         n_ratio_above_1_1=int((g.ratio > 1.1).sum()), max_grid_err=g.grid_err.max()))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(RES, 'common_grid_summary.csv'), index=False)
pd.set_option('display.width', 200)
print(R.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
print('\nMATLAB runs with ratio > 1.1:')
print(M[M.ratio > 1.1][['q', 'delta', 'tol', 'solver', 'form', 'nodal_err', 'grid_err', 'ratio']].to_string(index=False))
print('MATLAB clock runs without that one, max ratio:', M[(M.form == 'recip') & (M.ratio <= 1.1)].ratio.max())
x = M[(M.q == 3) & (M.delta == 1e-4) & (M.tol == 1e-4) & (M.solver == 'ode23t')]
print(x[['form', 'nodal_err', 'grid_err']].to_string(index=False))

# ---------------------------------------------------------------- Table S6 of the paper: MATLAB runs
R4 = pd.read_csv(os.path.join(RES, 'matlab', 'common_grid_matlab_refine4.csv'))   # matlab/run_common_grid_refine.m
key = ['q', 'delta', 'tol', 'solver', 'form']
MM = M.merge(R4[key + ['grid_err']], on=key, suffixes=('', '_r4'))
print('\nMATLAB grid refinement x4: max relative change of the maxima', (MM.grid_err_r4 / MM.grid_err - 1).abs().max())
S4 = pd.read_csv(os.path.join(RES, 'common_grid_scipy_refine4.csv'))                # exp_common_grid.py --refine 4
print('SciPy grid refinement x4: max relative change of the maxima', (S4.grid_err / S.grid_err - 1).abs().max())


def tex(v):
    m, e = f'{v:.1e}'.split('e'); e = int(e)
    if e == 0:
        return f'${m}$'
    return f'$10^{{{e}}}$' if m == '1.0' and abs(v / 10.0**e - 1) < 1e-9 else f'${m}\\cdot10^{{{e}}}$'


rows = []
for (q, d, tol, so), g in MM.groupby(['q', 'delta', 'tol', 'solver'], sort=False):
    a = g[g.form == 'std'].iloc[0]; b = g[g.form == 'recip'].iloc[0]
    rows.append(f'${q}$ & {tex(d)} & {tex(tol)} & \\texttt{{{so}}} & {tex(a.nodal_err)} & {tex(a.grid_err)} & '
                f'{tex(b.nodal_err)} & {tex(b.grid_err)} & {tex(b.grid_err_r4)}\\\\')
with open(os.path.join(os.path.dirname(__file__), '..', 'tables', 'common_grid_v25.tex'), 'w') as fh:
    fh.write('\n'.join(rows) + '\n')
print('wrote tables/common_grid_v25.tex (Table S6)')
