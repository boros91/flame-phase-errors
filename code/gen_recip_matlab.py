"""Summary of the MATLAB runs with the unknown w = y^(1-q) (matlab/run_recip.m).

Reads results/matlab/recip_matlab.csv (copy of matlab/results_matlab/recip_matlab.csv) and, for
comparison, the 'sundlog' rows of results/matlab/sundman_matlab.csv (time as a dependent
variable, Table 9). Prints, per solver and formulation, the range of the maximal error, of the
front delay and of the cost relative to the standard formulation, and the error of each run of
'recip' next to 'std' and 'sundlog'. Writes results/recip_matlab_summary.csv and
results/recip_matlab_compare.csv."""
import numpy as np, pandas as pd

r = pd.read_csv('../results/matlab/recip_matlab.csv')
print('MATLAB version(s):', r.matlab.unique() if 'matlab' in r else 'not recorded')
print('incomplete runs:'); print(r[r.completed == 0][['q', 'delta', 'tol', 'solver', 'form', 'nsteps', 'maxerr']].to_string(index=False))
r = r[r.completed == 1]
key = ['q', 'delta', 'tol', 'solver']
std = r[r.form == 'std'].set_index(key)

rows = []
for (s, form), g in r.groupby(['solver', 'form']):
    g = g.set_index(key)
    cost = (g.nfevals / std.nfevals.reindex(g.index)).dropna()
    for tol in sorted(g.index.get_level_values('tol').unique()):
        gt = g.xs(tol, level='tol')
        ct = cost[cost.index.get_level_values('tol') == tol]
        rows.append(dict(solver=s, form=form, tol=tol, n=len(gt),
                         maxerr_min=gt.maxerr.min(), maxerr_max=gt.maxerr.max(),
                         absD_max=gt.D_half.abs().max(),
                         cost_min=ct.min() if len(ct) else np.nan, cost_max=ct.max() if len(ct) else np.nan))
summ = pd.DataFrame(rows)
summ.to_csv('../results/recip_matlab_summary.csv', index=False)
pd.set_option('display.width', 220)
print(summ.to_string(float_format=lambda x: f'{x:.2e}'))

# run-by-run comparison: std, reciprel, recip, and sundlog from the Sundman runs
c = r.pivot_table(index=key, columns='form', values='maxerr')
c['cost_recip'] = r[r.form == 'recip'].set_index(key).nfevals / std.nfevals.reindex(r[r.form == 'recip'].set_index(key).index)
try:
    s = pd.read_csv('../results/matlab/sundman_matlab.csv')
    s = s[(s.completed == 1) & (s.form == 'sundlog')].set_index(key)
    c['sundlog'] = s.maxerr.reindex(c.index)
    c['cost_sundlog'] = s.nfevals.reindex(c.index) / std.nfevals.reindex(c.index)
except FileNotFoundError:
    pass
c.to_csv('../results/recip_matlab_compare.csv')
print(c.to_string(float_format=lambda x: f'{x:.1e}'))

# delta-dependence of the delay with absolute control of w (log of |D| vs log delta)
lg = r[(r.form == 'recip') & np.isfinite(r.D_half)]
for (sname, q, tol), g in lg.groupby(['solver', 'q', 'tol']):
    if len(g) >= 2:
        sl = np.polyfit(np.log10(g.delta), np.log10(np.abs(g.D_half) + 1e-300), 1)[0]
        print(f'recip {sname:8s} q={q} tol={tol:.0e}: |D| from {np.abs(g.D_half).min():.1e} to {np.abs(g.D_half).max():.1e}, slope in delta {sl:+.2f}')

# ---------------------------------------------------------------- table for the paper (m_clock.tex)
# rows: q, delta, tol; for ode45, ode15s, ode23s: maximal error of 'recip' and cost nfevals(recip)/nfevals(std)
def sci(x):
    if not np.isfinite(x): return '--'
    a, e = f'{x:.1e}'.split('e'); return f'${a}\\cdot10^{{{int(e)}}}$'
def dl(x):
    return f'$10^{{{int(round(np.log10(x)))}}}$'
rr = r[r.form == 'recip'].set_index(key)
lines = []
for q in (2, 3):
    first = True
    sub = sorted({k[:3] for k in rr.index if k[0] == q}, key=lambda k: (-k[1], -k[2]))
    for (qq, d, tol) in sub:
        cells = [f'${q}$' if first else '', dl(d), dl(tol)]
        for s in ('ode45', 'ode15s', 'ode23s'):
            k = (qq, d, tol, s)
            if k in rr.index:
                cost = rr.loc[k].nfevals / std.loc[k].nfevals
                cells += [sci(rr.loc[k].maxerr), f'{cost:.1f}']
            else:
                cells += ['--', '--']
        lines.append(' & '.join(cells) + r' \\'); first = False
    lines.append(r'\midrule')
open('../tables/m_clock.tex', 'w').write('\n'.join(lines[:-1]) + '\n')

# numbers quoted in the text
print('\nrecip, max error by q and tol:'); print(r[r.form == 'recip'].groupby(['q', 'tol']).maxerr.max())
print('reciprel, max error by q and tol:'); print(r[r.form == 'reciprel'].groupby(['q', 'tol']).maxerr.max())
cr = c.cost_recip.dropna(); print('cost recip: median %.2f, min %.2f, max %.2f' % (cr.median(), cr.min(), cr.max()))
if 'sundlog' in c:
    both = c.dropna(subset=['recip', 'sundlog'])
    print('recip more accurate than sundlog in %d of %d runs' % ((both.recip < both.sundlog).sum(), len(both)))
