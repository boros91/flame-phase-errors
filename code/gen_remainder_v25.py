"""Table 4 of the paper: tables/remainder_v3.tex (written by gen_tables.py) with the three RK4 entries
for q = 3 that are affected by rounding marked with a dagger (h = 0.25, delta = 1e-4; h = 0.125,
delta = 3e-4 and 1e-4). Writes tables/remainder_v25.tex."""
import os

T = os.path.join(os.path.dirname(__file__), '..', 'tables')
L = open(os.path.join(T, 'remainder_v3.tex')).read().rstrip('\n').split('\n')
assert L[-2].startswith('RK4 & 0.25') and L[-1].startswith(' & 0.125')
L[-2] = L[-2].replace(r'$-2.46\cdot10^{-6}$', r'$-2.46\cdot10^{-6}$$^\dagger$')
L[-1] = L[-1].replace(r'$4.77\cdot10^{-6}$', r'$4.77\cdot10^{-6}$$^\dagger$').replace(r'$8.24\cdot10^{-5}$', r'$8.24\cdot10^{-5}$$^\dagger$')
assert L[-2].count('dagger') == 1 and L[-1].count('dagger') == 2
open(os.path.join(T, 'remainder_v25.tex'), 'w').write('\n'.join(L) + '\n')
print('wrote tables/remainder_v25.tex')
