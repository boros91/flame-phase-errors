"""Time as a dependent variable with SUNDIALS/CVODE (BDF): standard formulation versus the
Sundman-transformed system, with the standard tolerance interface (scalar RelTol, vector AbsTol)
and with a user-supplied error weight function giving an ABSOLUTE tolerance on t.
Same problems, tolerances and measurements as exp_sundman_table.py (LSODA).

Needs SUNDIALS (tested with 6.4.1, e.g. apt install libsundials-dev) and gcc; compiles
sundials_ttol.c on first use. Writes results/sundials_ttol.csv."""
import os, subprocess, tempfile
import numpy as np, pandas as pd
from flame_q import T, exact

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, 'sundials_ttol')
if not os.path.exists(BIN):
    subprocess.run(['gcc', '-O2', '-o', BIN, os.path.join(HERE, 'sundials_ttol.c'),
                    '-lsundials_cvode', '-lsundials_nvecserial', '-lsundials_sunmatrixdense',
                    '-lsundials_sunlinsoldense', '-lm'], check=True)

def measure(t, y, d, q):          # identical to exp_sundman_table.measure
    ok = (y > d * (1 + 1e-12)) & (y < 0.99)
    D = t[ok] - T(y[ok], d, q)
    return D[np.argmax(y[ok] >= 0.5)], np.abs(y - exact(t, d, q)).max()

def run(q, d, tol, form, rho='power'):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'traj.csv')
        p = subprocess.run([BIN, str(q), repr(d), repr(tol), form, rho, out])
        lines = open(out).read().strip().split('\n')
    stats = dict(kv.split('=') for kv in lines[-1].split(',')[1:])
    tr = np.array([[float(v) for v in l.split(',')] for l in lines[1:-1]])
    t, y = tr[:, 0], tr[:, 1]
    D, err = measure(t, y, d, q) if p.returncode == 0 else (np.nan, np.nan)
    return dict(D=D, err=err, nst=int(stats['nst']), nfev=int(stats['nfe']) + int(stats['nfeLS']),
                ok=p.returncode == 0)

VARIANTS = [('std', 'std', 'power'), ('sv', 'sv', 'power'), ('wf', 'wf', 'power'), ('wfgen', 'wf', 'generic')]
rows = []
for q, ds in [(2, [1e-2, 1e-4, 1e-6, 1e-8]), (3, [1e-2, 1e-3, 1e-4])]:
    for d in ds:
        for tol in [1e-4, 1e-8]:
            r = dict(q=q, delta=d, tol=tol)
            for name, form, rho in VARIANTS:
                res = run(q, d, tol, form, rho)
                for k, v in res.items():
                    r[f'{name}_{k}'] = v
            rows.append(r); print({k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, '..', 'results', 'sundials_ttol.csv'), index=False)
