"""Work-precision figure for Paper A from the MATLAB R2023b runs (matlab/run_workprec.m)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C = {'ode45': '#2a78d6', 'ode113': '#1baf7a', 'ode15s': '#e87ba4', 'ode23s': '#eda100',
     'ode23t': '#7a5af5', 'ode23tb': '#555555'}
LS = {'standard': '-', 'warm': '--', 'scaled': ':'}
MK = {'standard': 'o', 'warm': 's', 'scaled': '^'}
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
m = pd.read_csv('../results/matlab/workprec_matlab.csv'); m = m[m.completed == 1]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.1), sharey=True)
for k, d in enumerate([1e-4, 1e-6]):
    a = ax[k]
    for sv in C:
        for st in ['standard', 'warm']:
            s = m[(m.delta == d) & (m.solver == sv) & (m.strategy == st)].sort_values('tol')
            if len(s):
                a.loglog(s.maxerr, s.nfevals, LS[st] + MK[st], color=C[sv], ms=2.5, mfc='none',
                         label=sv if st == 'standard' else None)
    a.set_xlabel('maximal error'); a.set_xlim(1e-12, 3)
    a.set_title(fr'({"ab"[k]}) $\delta=10^{{{int(np.log10(d))}}}$; solid: standard, dashed: warm', fontsize=7.5)
ax[0].set_ylabel('function evaluations')
ax[0].legend(fontsize=6.5, loc='upper left', ncol=2)
a = ax[2]
for d, col in [(1e-4, '#e87ba4'), (1e-6, '#8a2b57')]:
    for st in ['standard', 'scaled', 'warm']:
        s = m[(m.delta == d) & (m.solver == 'ode15s') & (m.strategy == st)].sort_values('tol')
        a.loglog(s.maxerr, s.nfevals, LS[st] + MK[st], color=col, ms=2.8, mfc='none',
                 label=fr'$\delta=10^{{{int(np.log10(d))}}}$, {st}')
a.set_xlabel('maximal error'); a.set_xlim(1e-12, 3)
a.set_title('(c) ode15s: standard, scaled, warm', fontsize=7.5)
a.legend(fontsize=5.8, loc='upper left')
fig.tight_layout(); fig.savefig('../figs/fig8_workprec_matlab.pdf'); plt.close(fig)
