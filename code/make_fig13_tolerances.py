"""Figure 9 of the paper (r_min = 100*eps in the labels): tightening the tolerances (MATLAB R2023b, matlab/run_tol_sweep_par.m;
numbers in code/gen_tol_sweep.py).
(a) |D_front| against RelTol (AbsTol = RelTol/100), delta = 1e-4, with slopes 2/3 and 1.
(b) ode15s, delta = 1e-4: |D_front| against AbsTol for fixed RelTol; the dotted lines mark AbsTol = RelTol*delta.
(c) work and precision: maximal error against steps, delta = 1e-4 (solid) and 1e-6 (dashed), RelTol = 1e-3 ... r_min;
    stars: unknown 1/y with RelTol = r_min, AbsTol = 1e-4 and 1e-8 (recip_matlab.csv), delta = 1e-6.
(d) smallest maximal error reachable with RelTol >= r_min against delta; stars: unknown 1/y, AbsTol = 1e-8, ode15s.
Writes figs/fig13_tolerances.pdf/.png."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, '..', 'results')
OUT = os.path.join(HERE, '..', 'figs')
t = pd.read_csv(os.path.join(RES, 'matlab', 'tol_sweep_matlab.csv'))
A = t[(t.part == 'A') & (t.completed == 1)]; B = t[t.part == 'B']
C = pd.read_csv(os.path.join(RES, 'tol_sweep_clock.csv'))
COL = {'ode45': (0.00, 0.45, 0.74), 'ode23': (0.49, 0.18, 0.56), 'ode23s': (0.85, 0.33, 0.10),
       'ode23t': (0.40, 0.40, 0.40), 'ode15s': (0.47, 0.67, 0.19)}
MK = {'ode45': 'o', 'ode23': 'd', 'ode23s': 's', 'ode23t': 'v', 'ode15s': '^'}
SOLV = ['ode45', 'ode23', 'ode23s', 'ode23t', 'ode15s']
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6, 'axes.grid': True, 'grid.color': '#e3e3e3',
                     'font.family': 'sans-serif', 'pdf.fonttype': 42, 'legend.frameon': True})
fig, ax = plt.subplots(2, 2, figsize=(11, 8.2))
ax = ax.ravel()

# (a)
for s in SOLV:
    x = A[(A.solver == s) & (A.delta == 1e-4)].sort_values('RelTol')
    ax[0].loglog(x.RelTol, np.abs(x.D_half), MK[s] + '-', color=COL[s], ms=4, lw=1, label=s)
rr = np.array([1e-12, 1e-5])
ax[0].loglog(rr, 3e4 * rr ** (2 / 3), 'k--', lw=0.8); ax[0].text(1.5e-12, 3e4 * 1e-8 * 1.6, 'slope 2/3', fontsize=8)
ax[0].loglog(rr, 1e5 * rr, 'k:', lw=0.8); ax[0].text(3e-7, 1e5 * 1e-7 * 0.15, 'slope 1', fontsize=8)
ax[0].invert_xaxis(); ax[0].set_xlabel('RelTol  (AbsTol = RelTol/100)'); ax[0].set_ylabel(r'$|D_{\mathrm{front}}|$')
ax[0].legend(loc='lower left', fontsize=7.5)
ax[0].set_title(r'(a) the delay against the tolerance, $\delta=10^{-4}$', fontsize=10)

# (b)
cm = plt.cm.viridis(np.linspace(0, 0.85, 5))
for c, rt in zip(cm, sorted(B.RelTol.unique(), reverse=True)):
    x = B[(B.solver == 'ode15s') & (B.RelTol == rt)].sort_values('AbsTol')
    ax[1].loglog(x.AbsTol, np.abs(x.D_half), 'o-', color=c, ms=4, lw=1, label=f'RelTol = {rt:.0e}')
    ax[1].axvline(rt * 1e-4, color=c, lw=0.8, ls=':')
ax[1].invert_xaxis(); ax[1].set_xlabel('AbsTol'); ax[1].set_ylabel(r'$|D_{\mathrm{front}}|$ (ode15s)')
ax[1].legend(loc='lower left', fontsize=7.5)
ax[1].set_title(r'(b) ode15s, $\delta=10^{-4}$: AbsTol below RelTol$\cdot\delta$ (dotted) does not help',
                fontsize=10)

# (c)
for s in SOLV:
    for d, ls, mfc in [(1e-4, '-', None), (1e-6, '--', 'white')]:
        x = A[(A.solver == s) & (A.delta == d)].sort_values('nsteps')
        ax[2].loglog(x.nsteps, x.maxerr, MK[s] + ls, color=COL[s], ms=4, lw=1, mfc=mfc or COL[s],
                     label=f'{s}' if d == 1e-4 else None)
for s in ['ode45', 'ode23s', 'ode15s']:
    x = C[(C.solver == s) & (C.delta == 1e-6)]
    ax[2].loglog(x.nsteps, x.maxerr, '*', color=COL[s], ms=13, mec='k', mew=0.6)
ax[2].plot([], [], 'k-', label=r'$\delta=10^{-4}$'); ax[2].plot([], [], 'k--', label=r'$\delta=10^{-6}$')
ax[2].plot([], [], '*', color='#999999', mec='k', ms=11, label=r'unknown $1/y$, $\delta=10^{-6}$')
ax[2].set_xlabel('steps'); ax[2].set_ylabel('maximal error'); ax[2].set_ylim(1e-11, 3)
ax[2].legend(loc='lower left', fontsize=7, ncol=2)
ax[2].set_title('(c) error against cost, RelTol from $10^{-3}$ down to $r_{\\min}$', fontsize=10)

# (d)
for s in SOLV:
    x = A[A.solver == s].groupby('delta').maxerr.min()
    ax[3].loglog(x.index, x.values, MK[s] + '-', color=COL[s], ms=5, lw=1.1, label=s + ' (best RelTol)')
x = C[(C.solver == 'ode15s') & (C.tol == 1e-8)].sort_values('delta')
ax[3].loglog(x.delta, x.maxerr, '*-', color=COL['ode15s'], ms=11, mec='k', mew=0.6, lw=0.8,
             label=r'ode15s, unknown $1/y$, AbsTol $=10^{-8}$')
ax[3].invert_xaxis(); ax[3].set_xlabel(r'$\delta$'); ax[3].set_ylabel('smallest maximal error')
ax[3].legend(loc='upper left', fontsize=7)
ax[3].set_title(r'(d) the best that RelTol $\geq r_{\min}$ can do', fontsize=10)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, 'fig13_tolerances.' + ext), dpi=200)
print('wrote fig13_tolerances')
