MATLAB scripts for the paper "Right shape, wrong time: phase errors of ODE solvers in the flame problem"
(Moler's solver comparison, extended)

Files
  run_moler_flame.m   solver comparison (Moler's setting, extended)
  run_workprec.m      work-precision study (standard / scaled tolerances / warm start)
  run_sundman.m       time as a dependent variable (Sundman transformation), four formulations:
                      std, sund (MATLAB tolerance interface), sundlog and sundloggen
                      (unknowns [log y; t], RelTol = 100*eps, AbsTol = [tol; tol]: absolute
                      control of t with unmodified solvers)
  run_one_sundman.m   one run for run_sundman.m
  run_recip.m         unknown w = y^(1-q) (= 1/y for q = 2), forms std, reciprel (RelTol = tol)
                      and recip (RelTol = 100*eps, AbsTol = tol: absolute control of w ~ of t)
  run_one_recip.m     one run for run_recip.m
  run_delay_profile.m accepted steps (t, y) in Moler's setting, for the delay-accumulation figure
  run_tol_sweep_par.m tightening the tolerances (Sect. 4.4): part A (ode45, ode23, ode23s, ode23t, ode15s;
                      delta 1e-2..1e-6; RelTol 1e-3..1e-13 and 100*eps; AbsTol = RelTol/100), part B
                      (RelTol x AbsTol grid, delta = 1e-4); parfor loop (runs serially without the Parallel
                      Computing Toolbox), resumable, with a time limit of 300 s per run
  run_one_capped.m    one run for run_tol_sweep_par.m (run_one.m with the time limit)
  run_common_grid.m   nodal versus common-grid error (deval), standard and clock formulation (Sect. 4.5);
                      results_matlab/common_grid_matlab.csv, summarised by code/gen_common_grid.py
  run_common_grid_refine.m  the same with four times as many output times (grid-refinement check);
                      results_matlab/common_grid_matlab_refine4.csv
  run_one.m           one solver run: statistics, slow/stiff step split, error, phase delay
  flame_exact.m       exact solution of y' = y^q (1-y), y(0) = delta (no toolbox needed)
  flame_master.m      master profile Y_q = Theta_q^{-1}
  flame_theta.m       time function Theta_q

How to run
  1. Put this folder on the MATLAB path (or cd into it).
  2. >> run_moler_flame
  3. The results are written to the subfolder results_matlab:
       moler_matlab.csv            one row per run (all solvers, deltas, tolerances)
       steps_q2_delta1e-4.csv      step points for Moler's setting (delta = 1e-4, RelTol = 1e-4)
       workprec_matlab.csv         (run_workprec.m)
       sundman_matlab.csv          (run_sundman.m; includes the MATLAB version of each run)
       recip_matlab.csv            (run_recip.m; summarised by code/gen_recip_matlab.py)
       delay_profile_matlab.csv    (run_delay_profile.m; used by code/make_fig_delay_profile.py)
       tol_sweep_matlab.csv        (run_tol_sweep_par.m; analysed by code/gen_tol_sweep.py)
  Copy these files to ../results/matlab/; the tables and figures are regenerated from them by the
  Python scripts listed in ../README.md.

All results in the paper were computed with MATLAB R2023b Update 3.

Solvers: ode45, ode23, ode113, ode23s, ode15s (default NDF), ode15s with 'BDF','on',
ode23t, ode23tb.  Tolerances: RelTol/AbsTol = 1e-4/1e-6 (Moler 2014), 1e-8/1e-12, 1e-12/1e-20.
No toolboxes are required (run_tol_sweep_par.m uses parfor, which runs serially without the
Parallel Computing Toolbox).
