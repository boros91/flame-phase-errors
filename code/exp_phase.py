"""Experiment 1: exact phase delays of fixed-step methods vs. the asymptotic
prediction D ~ h^p * int_delta^{y_k} g.  Writes results/phase_table.csv."""
import numpy as np, csv, os
from flame import phase_delay, predicted_delay, exact, T, trajectory, METHODS

os.makedirs('../results', exist_ok=True)
deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
rows = []
for name in ['Euler', 'Implicit Euler', 'NSFD3', 'NSFD4', 'Heun', 'RK4']:
    for h in [0.5, 0.25, 0.125]:
        for d in deltas:
            D, yk, k = phase_delay(name, d, h)
            P = predicted_delay(name, d, yk, h)
            rows.append(dict(method=name, h=h, delta=d, D=D, pred=P, ratio=D / P, steps=k))
            print(f'{name:15s} h={h:<6} delta={d:.0e}  D={D: .6e}  pred={P: .6e}  D/pred={D/P:.4f}')
with open('../results/phase_table.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

# NSFD with phi(h) = 1 - exp(-h): the numerical clock runs at rate phi(h)/h
print('\nNSFD3 with phi = 1-exp(-h):')
for h in [0.5, 2.0]:
    for d in [1e-2, 1e-3, 1e-4]:
        D, yk, k = phase_delay('NSFD3', d, h, phi=1 - np.exp(-h))
        tstar = T(0.5, d)
        print(f'  h={h} delta={d:.0e}: arrival at t={k*h:.1f} (exact {tstar:.1f}), '
              f'D={D:.1f}, (h/phi-1)/delta={(h/(1-np.exp(-h))-1)/d:.1f}, 2/delta={2/d:.0f}')

# global error = phase error: check |e_k| <= (4/27)|D_k| and e_k ~ -f(y_k) D_k
print('\nGlobal error vs phase delay along a trajectory (NSFD3, h=0.25, delta=1e-4):')
d, h = 1e-4, 0.25
n = int(2 / d / h)
ys = trajectory(2, d, h, h, n)
t = np.arange(n + 1) * h
ye = exact(t, d)
mask = ys < 1 - 1e-12
Dk = t[mask] - T(ys[mask], d)
e = ys - ye
print('  max|e| =', np.abs(e).max(), '  (4/27) max|D| =', 4 / 27 * np.abs(Dk).max(),
      '  max over front of f(y)|D| =', np.max((ys[mask]**2 - ys[mask]**3) * np.abs(Dk)))
np.savez('../results/traj_nsfd3.npz', t=t, ys=ys, ye=ye, D=np.where(mask, t - T(np.minimum(ys, 1 - 1e-15), d), np.nan))
