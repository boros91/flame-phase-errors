"""Fig. 3 of Paper A: global error = phase error, explicit Euler (results of exp_errphase.py)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C = ['#2a78d6', '#eb6834']
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
z = np.load('../results/traj_euler.npz')
t, ys, ye, D = z['t'], z['ys'], z['ye'], z['D']
w = (t > 9975) & (t < 10045)
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
ax[0].plot(t[w], ye[w], color=C[0], label='exact')
ax[0].plot(t[w], ys[w], '--', color=C[1], label='explicit Euler, $h=0.25$')
ax[0].set_xlabel('$t$'); ax[0].set_ylabel('$y$'); ax[0].legend(fontsize=7)
ax[0].set_title(r'(a) $\delta=10^{-4}$, near the transition', fontsize=9)
e = ys - ye
lin = -(ys**2 - ys**3) * D
ax[1].plot(t[w], e[w], color=C[1], label=r'error $y_k-y(t_k)$')
ax[1].plot(t[w], lin[w], ':', color='k', lw=1.2, label=r'$-f(y_k)\,D_k$')
ax[1].set_xlabel('$t$'); ax[1].legend(fontsize=7)
ax[1].set_title(r'(b) global error $\approx$ phase error', fontsize=9)
fig.tight_layout(); fig.savefig('../figs/fig3_error_is_phase_euler.pdf')
