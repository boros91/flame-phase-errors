"""Experiment: the remainder R = D_k - h^p int_delta^{y_k} g of Theorem 3.2 is O(h^{p+1})
uniformly in delta.  Compensated summation + exact delay accumulation (flame_q.py).
Writes results/theorem_remainder.csv"""
import numpy as np, csv
from flame_q import phase_delay, predicted_delay, METHODS

rows = []
cases = {2: [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 3: [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]}
for q, deltas in cases.items():
    for name, (m, p) in METHODS.items():
        if name.startswith('NSFD') and q != 2:
            continue
        for h in [0.5, 0.25, 0.125]:
            for d in deltas:
                D, yk, k = phase_delay(name, d, h, q)
                P = predicted_delay(name, d, yk, h, q)
                R = D - P
                rows.append(dict(q=q, method=name, p=p, h=h, delta=d, yk=yk, D=D, pred=P, R=R,
                                 R_scaled=R / h**(p + 1), steps=k))
                print(f'q={q} {name:15s} h={h:<6} d={d:.0e} D={D: .6e} pred={P: .6e} '
                      f'R/h^(p+1)={R / h**(p + 1): .4f}', flush=True)
with open('../results/theorem_remainder.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
