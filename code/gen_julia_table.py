"""Table of the Julia (OrdinaryDiffEq) runs (Table S4 of the paper) from results/julia/time_tolerance_julia.csv
(written by julia/flame_time_tolerance.jl). Writes tables/julia_ttol.tex."""
import numpy as np, pandas as pd

d = pd.read_csv('../results/julia/time_tolerance_julia.csv')
d = d[d.solver.notna()].copy()           # the CSV contains a few stray lines from Julia's output
d['delta'] = d.delta.astype(float)

def sci(x):
    a, e = f'{abs(x):.1e}'.split('e'); return f'${a}\\cdot10^{{{int(e)}}}$'

lines = []
for delta, solvers in [(1e-4, ['Tsit5', 'Vern7', 'QNDF', 'TRBDF2', 'KenCarp4', 'FBDF']),
                       (1e-6, ['QNDF', 'TRBDF2', 'KenCarp4', 'FBDF'])]:
    first = True
    for s in solvers:
        g = d[(d.delta == delta) & (d.solver == s)]
        std = g[g.form == 'standard'].iloc[0]
        tt = g[g.form == 'tau_abs_tol_t'].set_index('abstol_t')
        cells = [f'$10^{{{int(round(np.log10(delta)))}}}$' if first else '', s,
                 sci(std.D_half), f'{int(std.nf)}'] + [sci(tt.loc[a].D_half) for a in (1e-4, 1e-6, 1e-8)] + [f'{int(tt.loc[1e-6].nf)}']
        lines.append(' & '.join(cells) + r' \\'); first = False
    lines.append(r'\midrule')
open('../tables/julia_ttol.tex', 'w').write('\n'.join(lines[:-1]) + '\n')
