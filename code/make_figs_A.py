"""All figures of the paper (PDF, vector)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flame import exact, master, shift, T, trajectory
from flame_q import predicted_delay as pdq
predicted_delay = lambda m, d, y, h: pdq(m, d, y, h, 2)
from collocation import singularity

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#555555']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
FIG = '../figs/'
R = '../results/'

# ---------------------------------------------------------------- Fig 1
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
deltas = [1e-1, 1e-2, 1e-3, 1e-4]
for i, d in enumerate(deltas):
    t = np.linspace(0, 2 / d, 4000)
    ax[0].plot(t * d, exact(t, d), color=C[i], label=fr'$\delta=10^{{{int(np.log10(d))}}}$')
    tt = np.linspace(-60, 2 / d - shift(d), 4000)
    ax[1].plot(tt, exact(tt + shift(d), d), color=C[i], lw=3.2 - 0.7 * i)
ax[0].set_xlabel(r'$\delta t$'); ax[0].set_ylabel(r'$y_\delta$'); ax[0].legend(loc='upper left')
ax[0].set_title(r'(a) solutions on $[0,2/\delta]$, rescaled time', fontsize=9)
ax[1].set_xlim(-60, 15); ax[1].set_xlabel(r'$t-c(\delta)$')
ax[1].set_title(r'(b) same curves, shifted by $c(\delta)$', fontsize=9)
fig.tight_layout(); fig.savefig(FIG + 'fig1A_translation.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig 6 (q = 3)
from flame_q import predicted_delay as pdq3
th = pd.read_csv(R + 'theorem_remainder.csv'); th = th[th.q == 3]
s3 = pd.read_csv(R + 'q3_solvers.csv'); c3 = pd.read_csv(R + 'q3_colloc.csv')
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.3))
for i, m in enumerate(['Euler', 'Implicit Euler', 'Heun']):
    s = th[(th.method == m) & (th.h == 0.25)]
    ax[0].plot(s.delta, s.D, MK[i], color=C[i], ms=4.5, mfc='none', label=m.replace('Euler', 'Euler').replace('Implicit', 'implicit'))
    dd = np.logspace(-4.2, -1.8, 40)
    ax[0].plot(dd, [pdq3(m, d, 0.5, 0.25, 3) for d in dd], '-', color=C[i], lw=1)
ax[0].set_xscale('log'); ax[0].invert_xaxis(); ax[0].set_xlabel(r'$\delta$'); ax[0].set_ylabel('$D$ at $y=1/2$')
ax[0].legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2); ax[0].set_title('(a) fixed step, $h=0.25$', fontsize=9)
mm = pd.read_csv(R + 'matlab/moler_matlab.csv')
rel = mm[(mm.q == 3) & (mm.RelTol == 1e-12) & (mm.warm == 0) & (mm.completed == 1)]
for i, name in enumerate(['ode45', 'ode23', 'ode23s', 'ode15s', 'ode23tb']):
    s = rel[rel.solver == name]
    ax[1].loglog(s.delta, s.maxD, MK[i] + '-', color=C[i], ms=3.5, mfc='none', lw=1, label=name)
s = mm[(mm.q == 3) & (mm.RelTol == 1e-12) & (mm.warm == 1) & (mm.solver == 'ode15s')]
ax[1].loglog(s.delta, s.maxerr, 'P--', color=C[5], ms=3.5, lw=1, label='ode15s warm (error)')
dd = np.array([1e-5, 1e-1]); ax[1].loglog(dd, 1e-10 / dd**2, ':', color='#999999', lw=0.8)
ax[1].invert_xaxis(); ax[1].set_xlabel(r'$\delta$'); ax[1].set_ylabel(r'$\max_k|D_k|$')
ax[1].legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2); ax[1].set_title(r'(b) MATLAB, RelTol$=10^{-12}$', fontsize=9)
fig.tight_layout(); fig.savefig(FIG + 'fig6A_q3.pdf'); plt.close(fig)

print('ok')
