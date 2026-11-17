"""Figure for the adaptive-Euler example (results of exp_euler_toy.py) -> figs/fig9_euler_toy.pdf."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#555555']
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
R = '../results/'
LAB = {'std_ideal': r'$y$-control, ideal', 'std_practical': r'$y$-control, Euler--Heun',
       'time_ideal': r'$t$-control, ideal', 'time_practical': r'$t$-control, Euler--Heun'}
res = pd.read_csv(R + 'euler_toy.csv'); prof = pd.read_csv(R + 'euler_toy_profile.csv')

fig, ax = plt.subplots(1, 2, figsize=(7, 2.7))

# (a) accumulation profile, delta = 1e-4 (the curves for other delta coincide)
x = np.logspace(0, 5, 300)
d = 1e-4
g = prof[(prof.method == 'std_ideal') & (prof.delta == d)]
ax[0].plot(g.y / d, g.frac, color=C[0], lw=2.2, alpha=0.6, label=r'relative control of $y$')
g = prof[(prof.method == 'time_ideal') & (prof.delta == d)]
ax[0].plot(g.y / d, g.frac, color=C[2], lw=2.2, alpha=0.6, label=r'absolute control of $t$')
ax[0].plot(x, 1 - 1 / x, color='k', lw=0.8, ls='--', label=r'$1-\delta/y$')
ax[0].plot(x, 1 - 1 / np.sqrt(x), color='k', lw=0.8, ls=':', label=r'$1-\sqrt{\delta/y}$')
ax[0].set_xscale('log'); ax[0].set_xlim(1, 1e4); ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel(r'$y_k/\delta$'); ax[0].set_ylabel(r'$D_k/D_{\mathrm{final}}$')
ax[0].set_title(r'(a) where the delay is created, $\delta=10^{-4}$', fontsize=9, loc='left')
ax[0].legend(fontsize=7, loc='lower right')

# (b) delay versus delta
dd = np.logspace(-6, -2, 50)
for j, tol in enumerate([1e-4, 1e-6]):
    e = int(np.log10(tol))
    for name, mk, col in [('std_ideal', 'o', C[0]), ('std_practical', 's', C[1]),
                          ('time_ideal', '^', C[2]), ('time_practical', 'v', C[3])]:
        g = res[(res.method == name) & (res.tol == tol)]
        ax[1].plot(g.delta, np.abs(g.D), mk, color=col, ms=4.5, mew=0.9,
                   mfc=col if j == 0 else 'none', label=LAB[name] if j == 0 else None)
    ax[1].plot(dd, np.sqrt(tol) / dd, color='k', lw=0.8, ls='--')
    ax[1].plot(dd, np.sqrt(8 * tol / dd), color='k', lw=0.8, ls=':')
ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[1].set_xlabel(r'$\delta$'); ax[1].set_ylabel(r'front delay $|D|$ at $y=1/2$')
ax[1].set_title(r'(b) $\mathrm{tol}=10^{-4}$ (filled), $10^{-6}$ (open)', fontsize=9, loc='left')
ax[1].legend(fontsize=7, loc='upper right')
fig.tight_layout()
fig.savefig('../figs/fig9_euler_toy.pdf')
fig.savefig('../figs/fig9_euler_toy.png', dpi=150)
