"""Global error versus phase error along an explicit Euler trajectory (Paper A, Fig. 3):
explicit Euler, h = 0.25, delta = 1e-4, on [0, 2/delta]. Writes results/traj_euler.npz and prints
max|e|, (4/27) max|D| and the delay at y = 1/2."""
import numpy as np
from flame import exact, T, trajectory

d, h = 1e-4, 0.25
n = int(2 / d / h)
ys = trajectory(0, d, h, h, n)            # method 0 = explicit Euler
t = np.arange(n + 1) * h
ye = exact(t, d)
mask = ys < 1 - 1e-12
D = np.where(mask, t - T(np.minimum(ys, 1 - 1e-15), d), np.nan)
e = ys - ye
k = np.argmax(ys >= 0.5)
print('max|e| = %.3f, (4/27) max|D| = %.3f, D at y=1/2: %.3f' % (np.abs(e).max(), 4 / 27 * np.nanmax(np.abs(D)), D[k]))
np.savez('../results/traj_euler.npz', t=t, ys=ys, ye=ye, D=D)
