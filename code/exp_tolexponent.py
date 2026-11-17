"""Dependence of the front delay of the MATLAB solvers on RelTol in the relative regime.

The adaptive-Euler model (exp_euler_toy.py) predicts D ~ K * RelTol^(p/(p+1)) / delta^(q-1),
where p+1 is the power of h in the local error estimate that the code controls (exponent 1
if the code propagates a higher-order solution than the one it controls, local extrapolation).

Data: MATLAB R2023b runs (results/matlab/).
  (a) work-precision study, 'scaled' strategy: RelTol = tol*delta, AbsTol = tol*delta^2/100,
      always relative regime; least-squares slope of log|D(1/2)| against log RelTol over the
      completed runs with |D| > 1e-12 (rounding floor), per solver and delta.
  (b) solver comparison, q = 2 and 3: RelTol 1e-8 (AbsTol 1e-12) versus 1e-12 (AbsTol 1e-20)
      for the delta values where both are in the relative regime (AbsTol < RelTol*delta),
      slope from max|D_k|.
  (c) work-precision study, 'standard' strategy (RelTol = tol, AbsTol = tol/100), delta = 1e-4
      and 1e-6: always the absolute regime. Slope of log|D(1/2)| against log AbsTol over the runs
      with 1e-10 < |D| < 1, and the growth factor |D(1e-6)|/|D(1e-4)| at equal tol (two decades
      of delta; prediction 100^(1+alpha), alpha = p/(p+1)).
Writes results/tol_exponent.csv (a, b, c) and results/tol_exponent_delta.csv (c, growth factors)."""
import numpy as np, pandas as pd

rows = []
w = pd.read_csv('../results/matlab/workprec_matlab.csv')
s = w[(w.strategy == 'scaled') & (w.completed == 1) & (w.q == 2)]
for (solver, d), g in s.groupby(['solver', 'delta']):
    g = g[np.abs(g.D_half) > 1e-12]
    if len(g) < 3:
        continue
    slope = np.polyfit(np.log10(g.RelTol), np.log10(np.abs(g.D_half)), 1)[0]
    rows.append(dict(source='workprec_scaled', q=2, solver=solver, delta=d, n=len(g),
                     RelTol_min=g.RelTol.min(), RelTol_max=g.RelTol.max(), exponent=slope))

m = pd.read_csv('../results/matlab/moler_matlab.csv')
m = m[(m.warm == 0) & (m.completed == 1)]
for (q, solver, d), g in m.groupby(['q', 'solver', 'delta']):
    a = g[g.RelTol == 1e-8]; b = g[g.RelTol == 1e-12]
    if len(a) == 1 and len(b) == 1 and 1e-12 < 1e-8 * d:
        slope = np.log10(a.maxD.iloc[0] / b.maxD.iloc[0]) / 4
        rows.append(dict(source='moler_tight', q=q, solver=solver, delta=d, n=2,
                         RelTol_min=1e-12, RelTol_max=1e-8, exponent=slope))

st = w[(w.strategy == 'standard') & (w.completed == 1) & (w.q == 2)]
for (solver, d), g in st.groupby(['solver', 'delta']):
    g = g[(np.abs(g.D_half) > 1e-10) & (np.abs(g.D_half) < 1)]
    if len(g) < 3:
        continue
    slope = np.polyfit(np.log10(g.AbsTol), np.log10(np.abs(g.D_half)), 1)[0]
    rows.append(dict(source='workprec_standard_abs', q=2, solver=solver, delta=d, n=len(g),
                     RelTol_min=g.RelTol.min(), RelTol_max=g.RelTol.max(), exponent=slope))
a = st.pivot_table(index=['solver', 'tol'], columns='delta', values='D_half').abs().dropna()
a = a[a[1e-6] < 1e3]
a['growth'] = a[1e-6] / a[1e-4]
gr = a.groupby('solver').growth.agg(['median', 'min', 'max', 'count']).reset_index()
gr.to_csv('../results/tol_exponent_delta.csv', index=False)
print(gr.round(0))

df = pd.DataFrame(rows)
df.to_csv('../results/tol_exponent.csv', index=False)
pd.set_option('display.width', 200)
print(df.groupby(['source', 'q', 'solver']).exponent.agg(['min', 'max', 'count']).round(2))
