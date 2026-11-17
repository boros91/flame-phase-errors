"""Figure: how the phase delay accumulates along the trajectory, delta = 1e-4 -> figs/fig10_delay_profile.pdf.

(a) fixed-step methods, h = 0.25 (results/delay_profile_fixed.csv, exp_delay_profile.py):
    D_k / D_final versus y_k, computed (solid) and leading term of Theorem 3.4 (dashed).
(b) adaptive solvers at Moler's tolerances: D_k / D_final at the accepted steps.
(c) |D_k| for two stiff solvers, standard formulation versus time as a dependent variable in
    (ln y, t) with absolute tolerance on t.
(b), (c) use results/matlab/delay_profile_matlab.csv (matlab/run_delay_profile.m) if present,
otherwise the SciPy runs in results/delay_profile_scipy.csv (draft)."""
import os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flame_q import Theta

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#555555']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})
R = '../results/'
q, d = 2, 1e-4

fx = pd.read_csv(R + 'delay_profile_fixed.csv')
mpath = R + 'matlab/delay_profile_matlab.csv'
if os.path.exists(mpath):
    a = pd.read_csv(mpath)
    a = a[(a.y > d) & (a.y < 0.99)].copy()
    a['D'] = a.t - (Theta(a.y.values, q) - Theta(d, q))
    STD = ['ode45', 'ode113', 'ode23s', 'ode15s']
    PAIR = ['ode23s', 'ode15s']
    src = 'MATLAB'
else:
    a = pd.read_csv(R + 'delay_profile_scipy.csv').dropna(subset=['D'])
    a = a[a.y > d]
    STD = ['RK45', 'Radau', 'BDF', 'LSODA']
    PAIR = ['BDF', 'LSODA']
    src = 'SciPy (draft)'
print('adaptive data:', src)

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
ylab = r'$y_k$'

# (a) fixed step
for i, m in enumerate(['Euler', 'Implicit Euler', 'Heun', 'RK4']):
    g = fx[fx.method == m]
    Df = g.D.iloc[-1]
    lab = {'Implicit Euler': 'implicit Euler'}.get(m, m) + f' ($D={Df:.2g}$)'.replace('e-0', r'\cdot10^{-').replace('$D=1.9\\cdot10^{-6', '$D=1.9\\cdot10^{-6}')
    ax[0].plot(g.y, g.D / Df, color=C[i], lw=2.4 if i == 0 else 1.4, ls='-', label=lab, zorder=3 - 0.1 * i)
    ax[0].plot(g.y, g.pred / g.pred.iloc[-1], color='k', lw=0.6, ls='--', zorder=4)
ax[0].set_xscale('log'); ax[0].set_ylim(-0.05, 1.05); ax[0].set_xlim(d, 0.6)
ax[0].set_xlabel(ylab); ax[0].set_ylabel(r'$D_k/D_{\mathrm{final}}$')
ax[0].set_title(r'(a) fixed step, $h=0.25$', fontsize=9, loc='left')

# (b) adaptive, standard formulation
for i, s in enumerate(STD):
    g = a[(a.solver == s) & (a.form == 'std')].sort_values('k')
    Df = g[g.y >= 0.5].D.iloc[0]
    gg = g[g.y <= g[g.y >= 0.5].y.iloc[0]]
    ax[1].step(gg.y, gg.D / Df, where='post', color=C[i], lw=1.2)
    sel = (gg.y < 1e-3).values | (np.arange(len(gg)) % 6 == 0)
    ax[1].plot(gg.y[sel], (gg.D / Df)[sel], MK[i], color=C[i], ms=3.5, mfc='white', mew=0.9,
               label=f'{s} ($D={Df:.3g}$)')
ax[1].set_xscale('log'); ax[1].set_xlim(d, 0.6); ax[1].set_ylim(-0.6, 1.25)
ax[1].set_xlabel(ylab)
ax[1].set_title(r'(b) adaptive, $\mathrm{RelTol}=10^{-4}$', fontsize=9, loc='left')

# (c) |D_k|: standard versus time as a dependent variable
for i, s in enumerate(PAIR):
    col = C[[2, 3][i]] if src == 'MATLAB' else C[[2, 3][i]]
    for form, ls, lab in [('std', '-', s), ('sundlog', '--', s + r', $(\ln y,t)$')]:
        g = a[(a.solver == s) & (a.form == form)].sort_values('k')
        g = g[g.y <= 0.6]
        ax[2].plot(g.y, np.abs(g.D), ls, color=col, lw=1.3, marker=MK[[2, 3][i]], ms=3, mfc='white', mew=0.8,
                   label=lab, markevery=1 if form == 'std' else max(1, len(g) // 12))
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlim(d, 0.6); ax[2].set_ylim(1e-6, 1e3)
ax[2].set_xlabel(ylab); ax[2].set_ylabel(r'$|D_k|$')
ax[2].set_title(r'(c) controlling $t$', fontsize=9, loc='left')

for k in range(3):
    ax[k].axvline(2 * d, color='#999999', lw=0.6, ls=':')
ax[0].text(2.3 * d, 0.93, r'$y=2\delta$', fontsize=7, color='#555555')
fig.tight_layout()
# legends below the panels
for k in range(3):
    ax[k].legend(fontsize=6.2, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1, handlelength=1.8)
fig.subplots_adjust(bottom=0.42)
fig.savefig('../figs/fig10_delay_profile.pdf')
plt.close(fig)

# numbers quoted in the text (MATLAB data only): share of the final delay incurred by y = 10 delta,
# accepted steps until then, largest later change of D_k relative to D_final, max |D_k| up to y = 1/2
if src == 'MATLAB':
    rows = []
    for (s, form), g in a.groupby(['solver', 'form']):
        g = g.sort_values('k'); i5 = g[g.y >= 0.5].index[0]; Df = g.loc[i5, 'D']; g = g.loc[:i5]
        f10 = g[g.y <= 10 * d]; after = g[g.y > 10 * d]
        rows.append(dict(solver=s, form=form, D_final=Df, frac_at_10delta=f10.D.iloc[-1] / Df,
                         steps_to_10delta=int(f10.k.iloc[-1]), max_change_after=np.max(np.abs(after.D - Df)) / abs(Df),
                         maxabsD_to_half=g.D.abs().max()))
    sm = pd.DataFrame(rows); sm.to_csv(R + 'delay_profile_matlab_summary.csv', index=False)
    print(sm.to_string(float_format=lambda x: f'{x:.4g}'))
