"""Tables and figures from the MATLAB R2023b runs (matlab/run_moler_flame.m).
Input : results/matlab/moler_matlab.csv, results/matlab/steps_q2_delta1e-4.csv
Output: paper/tables/m_*.tex, figs/fig4_matlab.pdf, figs/fig6b_matlab.pdf, figs/fig7_matlab.pdf"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flame import T as T2
from flame_q import T as Tq

R = '../results/matlab/'; OUT = '../tables/'; FIG = '../figs/'
m = pd.read_csv(R + 'moler_matlab.csv')
st = pd.read_csv(R + 'steps_q2_delta1e-4.csv')
BETA = {'ode45': 3.3066, 'ode23': 2.5127}
NAME = {'ode45': r'\texttt{ode45}', 'ode23': r'\texttt{ode23}', 'ode113': r'\texttt{ode113}',
        'ode23s': r'\texttt{ode23s}', 'ode15s': r'\texttt{ode15s}', 'ode15s_BDF': r'\texttt{ode15s} (BDF)',
        'ode23t': r'\texttt{ode23t}', 'ode23tb': r'\texttt{ode23tb}'}
ORDER = ['ode45', 'ode23', 'ode113', 'ode23s', 'ode15s', 'ode15s_BDF', 'ode23t', 'ode23tb']


def sci(x, d=1):
    if not np.isfinite(x):
        return '--'
    if x == 0:
        return '$0$'
    a, e = f'{x:.{d}e}'.split('e')
    return f'${a}\\cdot10^{{{int(e)}}}$'


def dl(d):
    e = int(round(np.log10(d))); mm = d / 10**e
    return f'$10^{{{e}}}$' if abs(mm - 1) < 1e-9 else f'${mm:.0f}\\cdot10^{{{e}}}$'


q2 = m[m.q == 2]
# ---------------------------------------------------------------- Moler's table, delta = 1e-4
s = q2[(q2.delta == 1e-4) & (q2.RelTol == 1e-4)]
lines = []
for warm in (0, 1):
    for sv in ORDER:
        r = s[(s.solver == sv) & (s.warm == warm)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        lines.append(' & '.join([NAME[sv] + (' (warm)' if warm else ''), f'{int(r.nsteps)}', f'{int(r.nfailed)}',
                                 f'{int(r.nfevals)}', f'{int(r.npds)}', f'{int(r.ndecomps)}',
                                 f'{int(r.steps_slow)}', f'{int(r.steps_stiff)}',
                                 sci(abs(r.D_half)), sci(r.maxerr)]) + r' \\')
    if warm == 0:
        lines.append(r'\midrule')
open(OUT + 'm_moler1e-4.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- stiff-phase steps of explicit codes
lines = []
for d in [1e-3, 1e-4, 1e-5, 1e-6]:
    L = 2 / d - T2(0.5, d)
    g = q2[(q2.delta == d) & (q2.RelTol == 1e-4) & (q2.warm == 0)].set_index('solver')
    lines.append(' & '.join([dl(d), f'{L:.0f}',
                             f'{int(g.loc["ode23"].steps_stiff)}', f'{L / BETA["ode23"]:.0f}',
                             f'{int(g.loc["ode45"].steps_stiff)}', f'{L / BETA["ode45"]:.0f}',
                             f'{int(g.loc["ode113"].steps_stiff)}']) + r' \\')
open(OUT + 'm_stiffsteps.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- delta sweep at Moler's tolerance
lines = []
for d in [2e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    g = q2[(q2.delta == d) & (q2.RelTol == 1e-4) & (q2.warm == 0)].set_index('solver')
    cells = [dl(d)]
    for sv in ['ode45', 'ode23s', 'ode15s']:
        r = g.loc[sv]
        cells += [f'{int(r.nsteps)}', sci(abs(r.D_half)), sci(r.maxerr)]
    lines.append(' & '.join(cells) + r' \\')
open(OUT + 'm_sweep.tex', 'w').write('\n'.join(lines) + '\n')

# ---------------------------------------------------------------- tight tolerances, q = 2 and q = 3
for q, ds in [(2, [2e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]), (3, [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])]:
    lines = []
    for rt in [1e-8, 1e-12]:
        first = True
        for sv in ORDER:
            cells = [f'$10^{{{int(np.log10(rt))}}}$' if first else '', NAME[sv]]
            for d in ds:
                r = m[(m.q == q) & (m.delta == d) & (m.RelTol == rt) & (m.warm == 0) & (m.solver == sv)]
                if len(r) == 0:
                    cells.append('--'); continue
                r = r.iloc[0]
                v = sci(r.maxD)
                if not r.completed:
                    v = v[:-1] + '{}^{\\dagger}$'
                cells.append(v)
            lines.append(' & '.join(cells) + r' \\')
            first = False
        cells = ['', NAME['ode15s'] + ' (warm), error']
        for d in ds:
            r = m[(m.q == q) & (m.delta == d) & (m.RelTol == rt) & (m.warm == 1) & (m.solver == 'ode15s')]
            cells.append(sci(r.iloc[0].maxerr) if len(r) else '--')
        lines.append(' & '.join(cells) + r' \\')
        lines.append(r'\addlinespace[2pt]')
    open(OUT + f'm_tight_q{q}.tex', 'w').write('\n'.join(lines[:-1]) + '\n')

# ---------------------------------------------------------------- figures
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#7a5af5', '#555555']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
plt.rcParams.update({'pdf.fonttype': 42, 'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5, 'legend.frameon': False,
                     'font.family': 'serif', 'mathtext.fontset': 'cm',
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.4))
for i, sv in enumerate(['ode45', 'ode23s', 'ode15s']):
    t = st[st.solver == sv].t.values
    ax[0].semilogy(t[1:], np.diff(t), '.-', color=C[[0, 3, 4][i]], ms=2.5, lw=0.7, label=sv)
ax[0].axhline(3.3066, color='#999999', lw=0.8, ls=':')
ax[0].axvline(T2(0.5, 1e-4), color='#999999', lw=0.6, ls='--')
ax[0].set_xlabel('$t$'); ax[0].set_ylabel('step size $h$')
ax[0].set_title(r'(a) $\delta=10^{-4}$, RelTol$=10^{-4}$', fontsize=9)
ax[0].legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
mol = q2[(q2.RelTol == 1e-4) & (q2.warm == 0)]
dd = np.logspace(-6.1, -1.6, 40)
for i, sv in enumerate(['ode23', 'ode45', 'ode113']):
    s = mol[mol.solver == sv]
    ax[1].loglog(s.delta, s.nsteps, MK[i], color=C[i], ms=4, mfc='none', label=sv)
    if sv in BETA:
        ax[1].loglog(dd, (2 / dd - T2(0.5, dd)) / BETA[sv], '-', color=C[i], lw=0.8)
for i, sv in enumerate(['ode23s', 'ode15s', 'ode23tb']):
    s = mol[mol.solver == sv]
    ax[1].loglog(s.delta, s.nsteps, MK[i + 3] + '--', color=C[i + 3], ms=4, lw=0.8, label=sv)
ax[1].invert_xaxis(); ax[1].set_xlabel(r'$\delta$'); ax[1].set_ylabel('accepted steps')
ax[1].set_title(r'(b) cost; lines: $L_{\mathrm{stiff}}/\beta$', fontsize=9)
ax[1].legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3)
for i, sv in enumerate(['ode45', 'ode23', 'ode23s', 'ode15s', 'ode23tb']):
    s = mol[mol.solver == sv]
    ax[2].loglog(s.delta, np.abs(s.D_half), MK[i] + '-', color=C[[0, 1, 3, 4, 5][i]], ms=3.5, mfc='none', lw=1, label=sv)
for i, sv in enumerate(['ode45', 'ode15s']):
    s = q2[(q2.RelTol == 1e-4) & (q2.warm == 1) & (q2.solver == sv)]
    ax[2].loglog(s.delta, np.abs(s.D_half), MK[[0, 4][i]] + ':', color=C[[0, 4][i]], ms=3.5, lw=1, label=sv + ' (warm)')
ax[2].invert_xaxis(); ax[2].set_xlabel(r'$\delta$'); ax[2].set_ylabel(r'$|D|$ at $y=1/2$')
ax[2].set_title('(c) front position error', fontsize=9)
ax[2].legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
fig.tight_layout(); fig.savefig(FIG + 'fig7_matlab.pdf'); plt.close(fig)

for q, fname, ds_ref in [(2, 'fig4_matlab.pdf', (1e-6, 2e-2)), (3, 'fig6b_matlab.pdf', (1e-5, 1e-1))]:
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
    for j, rt in enumerate([1e-8, 1e-12]):
        s0 = m[(m.q == q) & (m.RelTol == rt) & (m.warm == 0) & (m.completed == 1)]
        for i, sv in enumerate(['ode45', 'ode23', 'ode113', 'ode23s', 'ode15s', 'ode23tb']):
            s = s0[s0.solver == sv]
            ax[j].loglog(s.delta, s.maxD, MK[i] + '-', color=C[i], ms=3.5, mfc='none', lw=1, label=sv)
        s = m[(m.q == q) & (m.RelTol == rt) & (m.warm == 1) & (m.solver == 'ode15s')]
        ax[j].loglog(s.delta, s.maxerr, 'P--', color=C[6], ms=3.5, lw=1, label='ode15s warm (error)')
        dd = np.array(ds_ref)
        ax[j].loglog(dd, 1e-1 * rt / dd**(q - 1), ':', color='#999999', lw=0.8)
        ax[j].invert_xaxis(); ax[j].set_xlabel(r'$\delta$')
        ax[j].set_title(fr'$q={q}$, RelTol$=10^{{{int(np.log10(rt))}}}$', fontsize=9)
    ax[0].set_ylabel(r'$\max_k|D_k|$')
    ax[1].legend(fontsize=6, loc='upper center', bbox_to_anchor=(-0.1, -0.2), ncol=4)
    fig.tight_layout(); fig.savefig(FIG + fname); plt.close(fig)
print('done')
