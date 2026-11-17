"""Work-precision figure for Paper A (SciPy runs)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C = {'RK45': '#2a78d6', 'DOP853': '#1baf7a', 'Radau': '#eb6834', 'BDF': '#e87ba4', 'LSODA': '#7a5af5'}
LS = {'standard': '-', 'warm': '--', 'scaled': ':'}
MK = {'standard': 'o', 'warm': 's', 'scaled': '^'}
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
wp = pd.read_csv('../results/workprec.csv'); ts = pd.read_csv('../results/tolscale.csv')
fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0))
for k, d in enumerate([1e-4, 1e-6]):
    a = ax[k]
    for m in C:
        for st in ['standard', 'warm']:
            s = wp[(wp.delta == d) & (wp.solver == m) & (wp.strategy == st)].sort_values('tol')
            if len(s) == 0:
                continue
            a.loglog(s.error_steps, s.nfev, LS[st] + MK[st], color=C[m], ms=2.5, mfc='none',
                     label=f'{m}' if st == 'standard' else None)
    a.set_xlabel('maximal error'); a.set_xlim(1e-13, 3)
    a.set_title(fr'({"ab"[k]}) $\delta=10^{{{int(np.log10(d))}}}$; solid: standard, dashed: warm', fontsize=7.5)
ax[0].set_ylabel('function evaluations')
ax[0].legend(fontsize=6.5, loc='lower left')
a = ax[2]
for m in ['Radau', 'BDF', 'LSODA']:
    for st in ['standard', 'scaled', 'warm']:
        s = ts[(ts.solver == m) & (ts.strategy == st)].sort_values('delta')
        a.loglog(s.delta, s.error_steps, LS[st] + MK[st], color=C[m], ms=3, mfc='none',
                 label=f'{m}, {st}' if m == 'BDF' else None)
a.invert_xaxis(); a.set_xlabel(r'$\delta$'); a.set_ylabel('maximal error')
a.set_title(r'(c) tol $=10^{-6}$; standard/scaled/warm', fontsize=7.5)
a.legend(fontsize=6, loc='upper left')
fig.tight_layout(); fig.savefig('../figs/fig8_workprec.pdf'); plt.close(fig)
