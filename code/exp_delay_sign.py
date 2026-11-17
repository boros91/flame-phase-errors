"""Sign of the front delay D(1/2) of the MATLAB solvers over all runs of matlab/run_moler_flame.m
(Paper A, Section 4.2): D < 0 means the computed front is early, D > 0 late.
Writes results/delay_sign.csv (per solver: runs, early, late)."""
import numpy as np, pandas as pd

m = pd.read_csv('../results/matlab/moler_matlab.csv')
m = m[(m.warm == 0) & (m.completed == 1) & np.isfinite(m.D_half)]
g = m.groupby('solver').D_half
out = pd.DataFrame({'runs': g.size(), 'early': g.apply(lambda x: int((x < 0).sum())),
                    'late': g.apply(lambda x: int((x > 0).sum()))})
out.to_csv('../results/delay_sign.csv')
print(out)
b = m[(m.q == 2) & (m.delta == 1e-4) & (m.RelTol == 1e-4)].set_index('solver').D_half
print('benchmark (q=2, delta=1e-4, RelTol=1e-4):'); print(b.round(2))
ex = m[(m.solver == 'ode23s') & (m.D_half > 0)][['q', 'delta', 'RelTol', 'D_half', 'maxerr']]
print('ode23s late runs:'); print(ex)
