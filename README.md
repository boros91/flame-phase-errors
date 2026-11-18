# flame-phase-errors

Code and data for the paper

> Imre Boros, *Right shape, wrong time: phase errors of ODE solvers in the flame problem* (2026), preprint, [arXiv:2611.23027](https://arxiv.org/abs/2611.23027).

The paper studies the flame family y' = y^q (1 - y), y(0) = delta, q >= 2. It shows that the global error of ODE solvers on this problem is a phase error: the computed front has the right shape but arrives at the wrong time. The phase delay is computed exactly from the accepted steps. The repository contains the Python, MATLAB, Julia and C code and the data that produce every figure and table of the paper and of its supplementary Tables S1-S6, and the numbers quoted in the text.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24314322.svg)](https://doi.org/10.5281/zenodo.24314322)

Version v1.0.0 of this repository is the version used in the paper; it is archived on Zenodo, https://doi.org/10.5281/zenodo.24314322. Corrections, if any, will appear as new releases.

## Contents

| Folder | Contents |
|---|---|
| `code/` | Python scripts: experiments (`exp_*.py`), figures (`make_fig*.py`), tables (`gen_*.py`), shared modules, and the C program `sundials_ttol.c` (SUNDIALS/CVODE) |
| `matlab/` | MATLAB scripts for the MATLAB solver runs (see `matlab/README_matlab.txt`) |
| `julia/` | Julia script for the OrdinaryDiffEq runs and two Python drivers (juliacall) |
| `results/` | Data written by the scripts. `results/matlab/` and `results/julia/` hold the MATLAB and Julia runs used in the paper |
| `figs/` | The figures of the paper (PDF), as used in the manuscript |
| `tables/` | The LaTeX table bodies of the paper, as written by the `gen_*.py` scripts |

## Software

- Python 3.13 with the packages in `requirements.txt` (NumPy 2.5, SciPy 1.18, pandas 3.0, Matplotlib 3.11, SymPy 1.14, mpmath 1.3, Numba 0.68)
- MATLAB R2023b Update 3 (23.2.0.2409890), no toolboxes needed
- Julia 1.13.1 with OrdinaryDiffEq 7.8.1, OrdinaryDiffEqBDF 2.4.12, OrdinaryDiffEqSDIRK 2.9.7, called through juliacall (versions in `results/julia/julia_versions.txt`)
- SUNDIALS 6.4.1 (CVODE) and gcc, for `code/exp_sundials.py`

## How to run

```
pip install -r requirements.txt
cd code
python gen_tables.py          # for example
```

Run the Python scripts from inside `code/`. They read from `../results` and write to `../results`, `../figs` and `../tables`, so **running them overwrites the files of the repository**. Use a fresh clone if you want to compare your output with the stored one (for example with `git diff`).

Run the MATLAB scripts from inside `matlab/`. They write to `matlab/results_matlab/`; copy the CSV files to `results/matlab/` before running the Python scripts that read them. The Julia drivers in `julia/` write `time_tolerance_julia.csv` next to themselves; copy it to `results/julia/`. The MATLAB and Julia data used in the paper are already in `results/`, so all tables and figures can be regenerated in Python alone.

The table scripts `gen_tables.py`, `gen_matlab.py`, `gen_remainder_v25.py`, `gen_tol_table.py`, `gen_recip_matlab.py`, `gen_sundman_matlab.py`, `gen_julia_table.py`, `gen_common_grid.py` and `exp_sundman_table.py` reproduce the files in `tables/` byte for byte from the data in `results/` (checked with Python 3.13). The PDFs in `figs/` are those of the manuscript; they were made with Matplotlib 3.10.9. Regenerated with the pinned Matplotlib 3.11.2, the figures show the same data, but tick placement and line rendering can differ slightly. Rerunning an experiment script reproduces its CSV file in `results/`, up to differences in the last digits that come from the linear algebra libraries.

Some scripts also compute quantities that are not reported in the paper (additional schemes, diagnostics, tables of earlier drafts in `results/tables_not_in_paper/`, PNG copies of figures); they are kept so that the scripts run unchanged.

## Which script produces what

Numbering of figures, tables and sections as in the paper.

### Figures

| Item | Script(s) | Output |
|---|---|---|
| Fig. 1 (translation structure) | `make_figs_A.py` | `figs/fig1A_translation.pdf` |
| Fig. 2 (phase delay, fixed step) | `exp_theorem.py` -> `make_fig2_phase.py` | `figs/fig2_phase.pdf` |
| Fig. 3 (global error = phase error, Euler) | `exp_errphase.py` -> `make_fig_errphase.py` | `figs/fig3_error_is_phase_euler.pdf` |
| Fig. 4 (adaptive explicit Euler) | `exp_euler_toy.py` -> `make_fig_euler_toy.py` | `figs/fig9_euler_toy.pdf` |
| Fig. 5 (MATLAB step sizes, delta = 1e-4) | `matlab/run_moler_flame.m` -> `gen_matlab.py` | `figs/fig7_matlab.pdf` |
| Fig. 6 (how the delay accumulates) | `exp_delay_profile.py` (panel a), `matlab/run_delay_profile.m` (b, c) -> `make_fig_delay_profile.py` | `figs/fig10_delay_profile.pdf` |
| Fig. 7 (MATLAB maximal delay vs. delta) | `matlab/run_moler_flame.m` -> `gen_matlab.py` | `figs/fig4_matlab.pdf` |
| Fig. 8 (work and precision, MATLAB) | `matlab/run_workprec.m` -> `make_fig_workprec_matlab.py` | `figs/fig8_workprec_matlab.pdf` |
| Fig. 9 (tightening the tolerances) | `matlab/run_tol_sweep_par.m` -> `gen_tol_sweep.py` -> `make_fig13_tolerances.py` | `figs/fig13_tolerances.pdf` |
| Fig. 10 (case q = 3) | `exp_theorem.py`, `exp_q3.py` -> `make_figs_A.py` | `figs/fig6A_q3.pdf` |

### Tables

| Item | Script(s) | Output |
|---|---|---|
| Table 1 (notation) | - | - |
| Table 2 (delay densities) | `theory_symbolic.py`, `theory_q.py` (symbolic; values typed into the paper) | - |
| Table 3 (fixed-step phase delays) | `exp_phase.py` -> `gen_tables.py` | `tables/phase_v3.tex` |
| Table 4 (remainder of the delay theorem) | `exp_theorem.py` -> `gen_tables.py` (`remainder_v3.tex`) -> `gen_remainder_v25.py` (marks the entries affected by rounding) | `tables/remainder_v25.tex` |
| Table 5 (MATLAB, benchmark run) | `matlab/run_moler_flame.m` -> `gen_matlab.py` | `tables/m_moler1e-4.tex` |
| Table 6 (steps in the stiff phase) | `matlab/run_moler_flame.m` -> `gen_matlab.py` | `tables/m_stiffsteps.tex` |
| Table 7 (MATLAB, sweep over delta) | `matlab/run_moler_flame.m` -> `gen_matlab.py` | `tables/m_sweep.tex` |
| Table 8 (price of tightening the tolerances) | `matlab/run_tol_sweep_par.m` -> `gen_tol_sweep.py` -> `gen_tol_table.py` | `tables/tol_needed_v24.tex` |
| Table 9 (MATLAB, unknown w = y^(1-q)) | `matlab/run_recip.m` -> `gen_recip_matlab.py` | `tables/m_clock.tex` |
| Table S1 (MATLAB, tight tolerances, q = 2) | `matlab/run_moler_flame.m` -> `gen_matlab.py` | `tables/m_tight_q2.tex` |
| Table S2 (time as dependent variable, LSODA) | `exp_sundman_table.py` | `tables/sundman.tex` |
| Table S3 (time as dependent variable, MATLAB) | `matlab/run_sundman.m` -> `gen_sundman_matlab.py` | `tables/m_sundman.tex` |
| Table S4 (Julia, absolute tolerance on t) | `julia/run_flame_julia.py` -> `gen_julia_table.py`; rerun with `julia/run_flame_julia_v24.py` gave identical output (versions in `results/julia/julia_versions.txt`) | `tables/julia_ttol.tex` |
| Table S5 (MATLAB, tight tolerances, q = 3) | `matlab/run_moler_flame.m` -> `gen_matlab.py` | `tables/m_tight_q3.tex` |
| Table S6 (nodal vs. common-grid error) | `matlab/run_common_grid.m`, `matlab/run_common_grid_refine.m` -> `gen_common_grid.py` | `tables/common_grid_v25.tex` |

### Numbers quoted in the text

| Item | Script(s) | Output (in `results/`) |
|---|---|---|
| Remark 3.7 (explicit Euler in y, ln y, y^(1-q)) | `exp_euler_variables.py` | `euler_variables.csv` |
| Stiffness after the front (Sect. 4.1) | `exp_stiffness.py` (reads `results/matlab`) | `stiffness_tol.csv`, `stiffness_delta.csv` |
| Front delay at the levels 0.1/0.5/0.9, event location (Sect. 4.2) | `exp_event_threshold.py` (reads `results/matlab`) | `threshold_levels.csv`, `event_location.csv` |
| Sign of the delay (Sect. 4.2) | `exp_delay_sign.py` (reads `results/matlab`) | `delay_sign.csv` |
| Tolerance exponents (Sect. 4.2) | `exp_tolexponent.py` (reads `results/matlab`) | `tol_exponent.csv`, `tol_exponent_delta.csv` |
| Where the delay is created (Sect. 4.2, SciPy RK45) | `exp_delay_profile.py`; `make_fig_delay_profile.py` | `delay_profile_scipy.csv`, `delay_profile_fixed.csv`, `delay_profile_matlab_summary.csv` |
| Tightening the tolerances (Sect. 4.4) | `gen_tol_sweep.py` (reads `results/matlab`) | `tol_sweep_exponents.csv`, `tol_sweep_needed.csv`, `tol_sweep_best.csv`, `tol_sweep_clock.csv` |
| Time as a dependent variable, MATLAB exponents and cost ratios (Sect. 4.5) | `gen_sundman_matlab.py` | `sundman_matlab_exponents.csv`, `sundman_matlab_summary.csv` |
| SUNDIALS/CVODE with absolute control of t (Sect. 4.5) | `exp_sundials.py` (compiles `sundials_ttol.c`) | `sundials_ttol.csv` |
| A clock as unknown (Sect. 4.5), recipe (Sect. 4.6) | `gen_recip_matlab.py` (reads `results/matlab`); SciPy check `exp_recip_scipy.py` | `recip_matlab_summary.csv`, `recip_matlab_compare.csv`, `recip_scipy.csv` |
| Nodal and common-grid errors (Sect. 4.5) | `matlab/run_common_grid.m`, `matlab/run_common_grid_refine.m` (`deval`; output times x1 and x4), `exp_common_grid.py [--refine 4]` (SciPy dense output) -> `gen_common_grid.py` | `common_grid_summary.csv`, `common_grid_scipy*.csv` |
| Case q = 3 (Sect. 5) | `exp_q3.py`, `exp_theorem.py` | `q3_solvers.csv`, `q3_colloc.csv`, `theorem_remainder.csv` (console output: `q3_log.txt`, `theorem_log.txt`) |
| SciPy solvers, time tolerance, rescaling, best constant shift | `exp_solvers.py`, `exp_moler.py`, `exp_workprec.py`, `exp_tolscale.py`, `exp_sundman.py`, `exp_sundman_rho.py`, `exp_ttol.py`, `exp_rescale.py`, `exp_bestshift.py` | `solver_table.csv`, `moler_scipy.csv`, `workprec.csv`, `tolscale.csv`, `sundman*.csv`, `ttol_sweep.csv`, `rescale.csv`, `bestshift.csv` |

Shared modules: `flame.py` (exact solution, q = 2), `flame_q.py` (general q, exact phase delays), `theory_q.py`, `theory_symbolic.py` (delay densities), `collocation.py`, `collocation_q.py` (collocation reference solutions used by some figure scripts).

## Fit ranges

All exponents are least-squares slopes of log10|D| in log-log scale.

- Sect. 4.2, exponents in RelTol and AbsTol (`exp_tolexponent.py`): (a) 'scaled' work-precision runs (RelTol = tol*delta, tol = 1e-3 ... 1e-12), completed runs with |D| > 1e-12, per solver and delta; (b) two-point slope between RelTol = 1e-8 and 1e-12 where both are in the relative regime; (c) 'standard' runs, slope in AbsTol over 1e-10 < |D| < 1, and growth factor |D(1e-6)|/|D(1e-4)| at equal tol.
- Sect. 4.4, tightening the tolerances (`gen_tol_sweep.py`): slope in RelTol over 1e-11 <= RelTol <= 1e-6 at delta = 1e-4; slope in delta over 1e-6 <= delta <= 1e-3 at RelTol = 1e-8.
- Sect. 4.5, time as dependent variable (`gen_sundman_matlab.py`, formulation 'sundlog'): slope in delta over all delta per solver, q and tol, excluding the two runs (q, delta, tol) = (2, 1e-8, 1e-8) and (3, 1e-4, 1e-8), which are limited by r_min and rounding; two-point slope in tol (1e-4, 1e-8).
- Sect. 4.5, clock (`gen_recip_matlab.py`): slope in delta over all delta per solver, q and tol.

## Failed runs and warnings

The MATLAB CSV files in `results/matlab/` record for every run whether t_end was reached (column `completed`) and the MATLAB version; most of them also record the last warning issued by the solver (column `warning`). Runs that did not complete are kept in the files and marked in the tables; the scripts listed above state which runs are excluded from fits.

## Citation

If you use this code, please cite the paper and this repository (see `CITATION.cff`).

## License

MIT, see `LICENSE`.
