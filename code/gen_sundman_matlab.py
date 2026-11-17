"""Table for the MATLAB runs of time as a dependent variable (matlab/run_sundman.m).

Reads results/matlab/sundman_matlab.csv (copy of matlab/results_matlab/sundman_matlab.csv) and
writes
  tables/m_sundman.tex    ode15s and ode23s: maximal error for the standard formulation,
                                 the transformed system with MATLAB's tolerance interface ('sund')
                                 and the transformed system in log y with absolute control of t
                                 ('sundlog'), and the cost ratio nfevals(sundlog)/nfevals(std)
  results/sundman_matlab_summary.csv   per solver and form: range of the maximal error and of
                                 the cost ratio over all (q, delta, tol)."""
import numpy as np, pandas as pd

df = pd.read_csv('../results/matlab/sundman_matlab.csv')
df = df[df.completed == 1]

def sci(x):
    if not np.isfinite(x): return '--'
    a, e = f'{x:.1e}'.split('e'); return f'${a}\\cdot10^{{{int(e)}}}$'

def dl(x):
    return f'$10^{{{int(round(np.log10(x)))}}}$'

key = ['q', 'delta', 'tol']
piv_err = df.pivot_table(index=key + ['solver'], columns='form', values='maxerr')
piv_nf = df.pivot_table(index=key + ['solver'], columns='form', values='nfevals')
piv_D = df.pivot_table(index=key + ['solver'], columns='form', values='D_half', dropna=False)

lines = []
for q in (2, 3):
    first = True
    sub = sorted({k[:3] for k in piv_err.index if k[0] == q}, key=lambda k: (-k[1], -k[2]))
    for (qq, d, tol) in sub:
        cells = [f'${q}$' if first else '', dl(d), dl(tol)]
        for s in ('ode15s', 'ode23s'):
            k = (qq, d, tol, s)
            if k in piv_err.index:
                e = piv_err.loc[k]; n = piv_nf.loc[k]
                ratio = n.get('sundlog', np.nan) / n.get('std', np.nan)
                skip = not np.isfinite(piv_D.loc[k].get('std', np.nan))   # standard run skipped the front
                cells += [sci(e.get('std', np.nan)), sci(e.get('sund', np.nan)),
                          sci(e.get('sundlog', np.nan)),
                          (f'{ratio:.1f}' + ('$^*$' if skip else '')) if np.isfinite(ratio) else '--']
            else:
                cells += ['--'] * 4
        lines.append(' & '.join(cells) + r' \\'); first = False
    lines.append(r'\midrule')
open('../tables/m_sundman.tex', 'w').write('\n'.join(lines[:-1]) + '\n')

std = df[df.form == 'std'].set_index(key + ['solver']).nfevals
rows = []
for (s, form), g in df.groupby(['solver', 'form']):
    g = g.set_index(key + ['solver'])
    r = (g.nfevals / std.reindex(g.index)).dropna()
    rows.append(dict(solver=s, form=form, n=len(g), maxerr_min=g.maxerr.min(), maxerr_max=g.maxerr.max(),
                     cost_ratio_min=r.min() if len(r) else np.nan, cost_ratio_max=r.max() if len(r) else np.nan))
summ = pd.DataFrame(rows)
summ.to_csv('../results/sundman_matlab_summary.csv', index=False)
pd.set_option('display.width', 200)
print(summ.to_string(float_format=lambda x: f'{x:.2e}'))
print('MATLAB version(s):', df.matlab.unique() if 'matlab' in df else 'not recorded')

# ---------------------------------------------------------------- 'sundlog': growth of |D| with delta and tol
# accumulation model (Section 4.4): D ~ tol^alpha * delta^(-(q-1)(1-alpha)), alpha = p/(p+1).
# Runs limited by the rounding of t (t ~ 1e8 near the front) are excluded:
# (q, delta, tol) = (2, 1e-8, 1e-8) and (3, 1e-4, 1e-8).
lg = df[(df.form == 'sundlog') & np.isfinite(df.D_half)].copy()
lg = lg[~(((lg.q == 2) & (lg.delta == 1e-8) & (lg.tol == 1e-8)) | ((lg.q == 3) & (lg.delta == 1e-4) & (lg.tol == 1e-8)))]
ex = []
for (s, q, tol), g in lg.groupby(['solver', 'q', 'tol']):
    if len(g) >= 2:
        sl = np.polyfit(np.log10(g.delta), np.log10(np.abs(g.D_half)), 1)[0]
        ex.append(dict(solver=s, q=q, tol=tol, kind='delta', n=len(g), exponent=sl))
for (s, q, d), g in lg.groupby(['solver', 'q', 'delta']):
    if len(g) == 2:
        sl = np.log10(np.abs(g.D_half.iloc[0]) / np.abs(g.D_half.iloc[1])) / np.log10(g.tol.iloc[0] / g.tol.iloc[1])
        ex.append(dict(solver=s, q=q, delta=d, kind='tol', n=2, exponent=sl))
ex = pd.DataFrame(ex)
ex.to_csv('../results/sundman_matlab_exponents.csv', index=False)
print('\nsundlog: exponent of |D| in delta (expected -(q-1)(1-alpha)) and in tol (expected alpha):')
print(ex.groupby(['kind', 'solver', 'q']).exponent.agg(['min', 'max', 'count']).round(2))

# cost ratio sundlog / std, excluding runs in which the standard formulation skips the front (D_half NaN)
stdD = df[df.form == 'std'].set_index(key + ['solver']).D_half
lgc = df[df.form == 'sundlog'].set_index(key + ['solver'])
r = (lgc.nfevals / std.reindex(lgc.index))[np.isfinite(stdD.reindex(lgc.index))].dropna()
print('\nsundlog cost ratio by solver:'); print(r.groupby(level='solver').agg(['min', 'max']).round(2))

# generic rate function versus rho = y^(q-1), log formulation
a = df[df.form == 'sundlog'].set_index(key + ['solver']); b = df[df.form == 'sundloggen'].set_index(key + ['solver'])
rr = (b.maxerr / a.maxerr.reindex(b.index)).dropna(); cc = (b.nfevals / a.nfevals.reindex(b.index)).dropna()
print('\nsundloggen / sundlog: error ratio %.2f-%.2f, cost ratio %.2f-%.2f' % (rr.min(), rr.max(), cc.min(), cc.max()))
print('incomplete runs:'); print(pd.read_csv('../results/matlab/sundman_matlab.csv').query('completed == 0')[['q', 'delta', 'tol', 'solver', 'form', 'nsteps', 'nfevals']])
