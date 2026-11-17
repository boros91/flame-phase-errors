"""Generate LaTeX table bodies from the CSV results (no hand transcription)."""
import pandas as pd, numpy as np, os
from collocation import singularity
R = '../results/'; OUT = '../tables/'
EXTRA = '../results/tables_not_in_paper/'  # tables of earlier drafts, not used in the paper
os.makedirs(EXTRA, exist_ok=True)


def sci(x, d=2):
    if not np.isfinite(x):
        return '--'
    if x == 0:
        return '$0$'
    s = f'{x:.{d}e}'
    m, e = s.split('e')
    return f'${m}\\cdot10^{{{int(e)}}}$'


# Table: fixed-step phase delays, h = 0.25
df = pd.read_csv(R + 'phase_table.csv')
names = {'Euler': 'explicit Euler', 'Implicit Euler': 'implicit Euler', 'NSFD3': 'NSFD1, $\\varphi=h$',
         'NSFD4': 'NSFD2, $\\varphi=h$', 'Heun': 'Heun', 'RK4': 'RK4'}
lines = []
for m in names:
    cells = [names[m]]
    for d in [1e-2, 1e-4, 1e-6]:
        r = df[(df.method == m) & (df.h == 0.25) & (df.delta == d)].iloc[0]
        cells += [sci(r.D, 3), f'{r.ratio:.3f}']
    lines.append(' & '.join(cells) + r' \\')
open(EXTRA + 'phase.tex', 'w').write('\n'.join(lines) + '\n')

# Table: adaptive solvers
sv = pd.read_csv(R + 'solver_table.csv')
lines = []
for d in [1e-4, 1e-6]:
    for rt in [1e-8, 1e-12]:
        s = sv[(sv.delta == d) & (sv.rtol == rt)]
        first = True
        for _, r in s.iterrows():
            lines.append(' & '.join([
                f'$10^{{{int(np.log10(d))}}}$' if first else '',
                f'$10^{{{int(np.log10(rt))}}}$/$10^{{{int(np.log10(r.atol))}}}$' if first else '',
                r.solver, sci(r.maxerr), sci(r.maxD), f'{int(r.steps)}']) + r' \\')
            first = False
        lines.append(r'\addlinespace[2pt]')
open(EXTRA + 'solvers.tex', 'w').write('\n'.join(lines[:-1]) + '\n')

# Table: collocation
cs = pd.read_csv(R + 'colloc_sinh.csv')
lines = []
for d, g in cs.groupby('delta', sort=False):
    x0, eps = singularity(d)
    L = (np.arcsinh((1 - x0) / eps) - np.arcsinh((-1 - x0) / eps)) / 2
    n = int(g[g.pin <= 1e-10].N.min())
    r = g[g.N == n].iloc[0]
    r384 = g[g.N == 384].iloc[0]
    lines.append(' & '.join([f'$10^{{{int(np.log10(d))}}}$', f'{L:.2f}', f'{2*L/np.pi*np.log(1e10):.0f}', f'{n}',
                             sci(r.pin, 1), f'{int(r.pin_its)}', sci(r.y0_relerr, 1),
                             (sci(r384.ivp, 1) if r384.ivp_conv else sci(r384.ivp, 1)[:-1] + '{}^{\\ast}$'), f'{int(r384.ivp_its)}'])
                 + r' \\')
open(EXTRA + 'colloc.tex', 'w').write('\n'.join(lines) + '\n')
print(open(EXTRA + 'phase.tex').read()); print(open(EXTRA + 'colloc.tex').read())


# ---------------------------------------------------------------- version 2 tables
TH = R + 'theorem_remainder.csv'
if os.path.exists(TH):
    th = pd.read_csv(TH)
    # Table 2 (v2): q = 2, h = 0.25, compensated summation
    lines = []
    for m in names:
        cells = [names[m]]
        for d in [1e-2, 1e-4, 1e-6]:
            r = th[(th.q == 2) & (th.method == m) & (th.h == 0.25) & (th.delta == d)].iloc[0]
            cells += [sci(r.D, 3), f'{r.D / r.pred:.3f}']
        lines.append(' & '.join(cells) + r' \\')
    open(EXTRA + 'phase_v2.tex', 'w').write('\n'.join(lines) + '\n')
    # remainder table: R/h^{p+1}
    def dlab(d):
        e = int(np.floor(np.log10(d) + 1e-9)); m_ = d / 10**e
        return f'$10^{{{e}}}$' if abs(m_ - 1) < 1e-9 else f'${m_:.0f}\\cdot10^{{{e}}}$'

    def num(v):
        if abs(v) >= 1e-3:
            return f'${v:.3f}$'
        m_, e = f'{v:.2e}'.split('e')
        return f'${m_}\\cdot10^{{{int(e)}}}$'
    lab = {'Euler': 'explicit Euler', 'Implicit Euler': 'implicit Euler', 'NSFD3': 'NSFD1',
           'NSFD4': 'NSFD2', 'Heun': 'Heun', 'RK4': 'RK4'}
    lines = []
    for q in [2, 3]:
        ds = sorted(th[th.q == q].delta.unique(), reverse=True)
        if q == 3:
            lines.append(r'\midrule')
        lines.append(' & '.join([f'$q={q}$', '$h$'] + [dlab(d) for d in ds]) + r' \\')
        lines.append(r'\midrule')
        for m in ['Euler', 'Implicit Euler', 'NSFD3', 'NSFD4', 'Heun', 'RK4']:
            if q == 3 and m.startswith('NSFD'):
                continue
            for h in [0.25, 0.125]:
                s_ = th[(th.q == q) & (th.method == m) & (th.h == h)].set_index('delta').loc[ds]
                lines.append(' & '.join([lab[m] if h == 0.25 else '', f'{h}'] + [num(v) for v in s_.R_scaled])
                             + r' \\')
    open(EXTRA + 'remainder.tex', 'w').write('\n'.join(lines) + '\n')

Q3S = R + 'q3_solvers.csv'
if os.path.exists(Q3S):
    s3 = pd.read_csv(Q3S)
    ds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    lines = []
    for rt in [1e-8, 1e-12]:
        first = True
        for name in ['RK45', 'DOP853', 'Radau', 'BDF', 'LSODA', 'Radau (warm)']:
            cells = [f'$10^{{{int(np.log10(rt))}}}$' if first else '', name]
            for d in ds:
                r = s3[(s3.rtol == rt) & (s3.solver == name) & (s3.delta == d)]
                if len(r) == 0:
                    cells.append('--'); continue
                r = r.iloc[0]
                v = r.maxerr if 'warm' in name else r.maxD
                cells.append(sci(v, 1) if r.status == 0 else sci(v, 1)[:-1] + '{}^{\\dagger}$')
            lines.append(' & '.join(cells) + r' \\')
            first = False
        lines.append(r'\addlinespace[2pt]')
    open(EXTRA + 'q3_solvers.tex', 'w').write('\n'.join(lines[:-1]) + '\n')

Q3C = R + 'q3_colloc.csv'
if os.path.exists(Q3C):
    c3 = pd.read_csv(Q3C)
    lines = []
    for d, g in c3.groupby('delta', sort=False):
        L = g.L.iloc[0]
        n = int(g[g.pin <= 1e-10].N.min())
        r = g[g.N == n].iloc[0]
        r384 = g[g.N == 384].iloc[0]
        lines.append(' & '.join([f'$10^{{{int(round(np.log10(d)))}}}$', f'{L:.2f}', f'{2*L/np.pi*np.log(1e10):.0f}',
                                 f'{n}', sci(r.pin, 1), f'{int(r.pin_its)}', sci(r.y0_relerr, 1),
                                 (sci(r384.ivp, 1) if r384.ivp_conv else sci(r384.ivp, 1)[:-1] + '{}^{\\ast}$'), f'{int(r384.ivp_its)}'])
                     + r' \\')
    open(EXTRA + 'q3_colloc.tex', 'w').write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------- version 3 tables
if os.path.exists(TH):
    # Table 2 / remainder table without the NSFD rows
    names3 = {k: v for k, v in names.items() if not k.startswith('NSFD')}
    lines = []
    for m in names3:
        cells = [names3[m]]
        for d in [1e-2, 1e-4, 1e-6]:
            r = th[(th.q == 2) & (th.method == m) & (th.h == 0.25) & (th.delta == d)].iloc[0]
            cells += [sci(r.D, 3), f'{r.D / r.pred:.3f}']
        lines.append(' & '.join(cells) + r' \\')
    open(OUT + 'phase_v3.tex', 'w').write('\n'.join(lines) + '\n')
    txt = open(EXTRA + 'remainder.tex').read().split('\n')
    keep, skip = [], 0
    for l in txt:
        if l.startswith('NSFD'):
            skip = 2
        if skip:
            skip -= 1; continue
        keep.append(l)
    open(OUT + 'remainder_v3.tex', 'w').write('\n'.join(keep))

MS = R + 'moler_scipy.csv'
if os.path.exists(MS):
    ms = pd.read_csv(MS)
    lines = []
    for d in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        g = ms[ms.delta == d].set_index('solver')
        a, b, c = g.loc['RK45'], g.loc['Radau'], g.loc['BDF']
        lines.append(' & '.join([f'$10^{{{int(round(np.log10(d)))}}}$',
                                 f'{int(a.steps)}', f'{int(a.slow)}', f'{int(a.stiff)}', f'{a.pred_stiff:.0f}',
                                 sci(abs(a.D_half), 1), sci(a.maxerr, 1),
                                 f'{int(b.steps)}', sci(abs(b.D_half), 1), sci(b.maxerr, 1),
                                 f'{int(c.steps)}', sci(abs(c.D_half), 1), sci(c.maxerr, 1)]) + r' \\')
    open(EXTRA + 'moler.tex', 'w').write('\n'.join(lines) + '\n')
    lines = []
    for d in [1e-3, 1e-4, 1e-5, 1e-6]:
        g = ms[ms.delta == d].set_index('solver')
        cells = [f'$10^{{{int(round(np.log10(d)))}}}$']
        for m in ['RK23', 'RK45', 'DOP853']:
            cells += [f'{int(g.loc[m].stiff)}', f'{g.loc[m].pred_stiff:.0f}']
        lines.append(' & '.join(cells) + r' \\')
    open(EXTRA + 'stiffsteps.tex', 'w').write('\n'.join(lines) + '\n')
