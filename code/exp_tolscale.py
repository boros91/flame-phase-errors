"""Tolerance scaling versus warm start, tol = 1e-6, several delta (q = 2). Writes results/tolscale.csv."""
import pandas as pd
from exp_workprec import run
rows = []
for d in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    for m in ['RK45', 'Radau', 'BDF', 'LSODA']:
        if m == 'RK45' and d < 1e-4:
            continue
        for strat in ['standard', 'scaled', 'warm']:
            r = run(d, m, 1e-6, strat)
            if r:
                rows.append(r)
df = pd.DataFrame(rows); df.to_csv('../results/tolscale.csv', index=False)
pd.set_option('display.width', 200)
print(df.pivot_table(index=['solver', 'delta'], columns='strategy', values=['error_steps', 'nfev']).to_string(float_format=lambda x: f'{x:.2g}'))
