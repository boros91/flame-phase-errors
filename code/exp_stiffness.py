"""Operational evidence for stiffness after the front (Paper A, Section 4.1), from the MATLAB R2023b runs.

  (a) stiff-phase steps versus tolerance, delta = 1e-4, standard strategy of the
      work-precision study (RelTol = tol, AbsTol = tol/100, tol = 1e-3 ... 1e-12);
  (b) stiff-phase steps versus delta at the benchmark tolerance (RelTol 1e-4, AbsTol 1e-6);
  (c) rejected steps versus tolerance (same runs as (a)).
Writes results/stiffness_tol.csv, results/stiffness_delta.csv and prints the ranges quoted in the text."""
import pandas as pd

w = pd.read_csv('../results/matlab/workprec_matlab.csv')
a = w[(w.strategy == 'standard') & (w.delta == 1e-4) & (w.completed == 1) & (w.q == 2)]
tab = a.pivot_table(index='tol', columns='solver', values=['steps_stiff', 'nfailed'])
tab.to_csv('../results/stiffness_tol.csv')
print('(a) stiff-phase steps, delta = 1e-4, tol = 1e-3..1e-12:')
print(a.groupby('solver').steps_stiff.agg(['min', 'max']))
print('(c) rejected steps:')
print(a.groupby('solver').nfailed.agg(['min', 'max']))
print(a[a.tol <= 1e-5].groupby('solver').nfailed.agg(['min', 'max']).rename(columns=lambda c: c + '_tol<=1e-5'))

m = pd.read_csv('../results/matlab/moler_matlab.csv')
b = m[(m.q == 2) & (m.RelTol == 1e-4) & (m.warm == 0) & (m.completed == 1)]
tb = b.pivot_table(index='delta', columns='solver', values='steps_stiff')
tb.to_csv('../results/stiffness_delta.csv')
print('(b) stiff-phase steps versus delta, benchmark tolerance:')
print(tb[['ode45', 'ode23', 'ode15s', 'ode23s']])
