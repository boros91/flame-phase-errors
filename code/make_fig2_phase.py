"""Figure 2 of the paper (fig2_phase.pdf): phase delay at y = 1/2 for fixed-step methods.
Reads results/theorem_remainder.csv (written by exp_theorem.py)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flame import exact, master, shift, T, trajectory
from flame_q import predicted_delay as pdq
predicted_delay = lambda m, d, y, h: pdq(m, d, y, h, 2)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#555555']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
FIG = '../figs/'
R = '../results/'


# ---------------------------------------------------------------- Fig 2
df = pd.read_csv(R + 'theorem_remainder.csv'); df = df[df.q == 2]
df['ratio'] = df.D / df.pred
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
h = 0.25
for i, m in enumerate(['Euler', 'Implicit Euler', 'Heun', 'RK4']):
    s = df[(df.method == m) & (df.h == h)]
    ax[0].plot(s.delta, s.D, MK[i], color=C[i], ms=5, mfc='none', label=m)
    dd = np.logspace(-6.3, -1.7, 50)
    ax[0].plot(dd, [predicted_delay(m, d, 0.5, h) for d in dd], '-', color=C[i], lw=1)
ax[0].set_xscale('log'); ax[0].invert_xaxis()
ax[0].set_xlabel(r'$\delta$'); ax[0].set_ylabel(r'phase delay $D$ at $y=1/2$')
ax[0].set_title(r'(a) $h=0.25$: markers computed, lines predicted', fontsize=9)
hs = ax[0].get_legend_handles_labels()
ax[0].legend(hs[0], ['Euler', 'implicit Euler', 'Heun', 'RK4'], fontsize=6.5, ncol=1,
             loc='upper left', bbox_to_anchor=(1.0, 1.0))
for i, m in enumerate(['Euler', 'Implicit Euler', 'Heun', 'RK4']):
    s = df[(df.method == m) & (df.delta == 1e-4)]
    ax[1].loglog(s.h, np.abs(s.D), MK[i], color=C[[0, 2, 4, 5][i]], ms=5, mfc='none',
                 label={'Euler': 'Euler', 'Implicit Euler': 'implicit Euler', 'Heun': 'Heun', 'RK4': 'RK4'}[m])
    hh = np.array([0.1, 0.6])
    ax[1].loglog(hh, [abs(predicted_delay(m, 1e-4, 0.5, x)) for x in hh], '-', color=C[[0, 2, 4, 5][i]], lw=1)
ax[1].set_xlabel(r'$h$'); ax[1].set_ylabel(r'$|D|$'); ax[1].legend(fontsize=6.5, loc='lower right')
ax[1].set_xticks([0.125, 0.25, 0.5]); ax[1].set_xticklabels(['1/8', '1/4', '1/2']); ax[1].minorticks_off()
ax[1].set_title(r'(b) $\delta=10^{-4}$: slopes $p=1,1,2,4$', fontsize=9)
fig.tight_layout(); fig.savefig(FIG + 'fig2_phase.pdf'); plt.close(fig)
