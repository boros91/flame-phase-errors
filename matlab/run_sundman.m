%RUN_SUNDMAN  Time as a dependent variable with the MATLAB ODE suite.
%
%   Problem:  y' = y^q (1-y), y(0) = delta, 0 <= t <= 2/((q-1) delta^(q-1)).
%   Same grid as the LSODA table of the paper (code/exp_sundman_table.py):
%     q = 2: delta = 1e-2, 1e-4, 1e-6, 1e-8;  q = 3: delta = 1e-2, 1e-3, 1e-4;
%     tol = 1e-4, 1e-8.
%   Formulations (see run_one_sundman.m):
%     std        original equation, RelTol = tol, AbsTol = tol/100
%     sund       [y; t], rho = y^(q-1), RelTol = tol, AbsTol = [tol/100; tol]
%                (MATLAB's own interface: the tolerance on t is relative)
%     sundlog    [log y; t], rho = y^(q-1), RelTol = 100*eps, AbsTol = [tol; tol]
%                (relative control of y, ABSOLUTE control of t, unmodified solvers)
%     sundloggen as sundlog with rho = |f|/|y| + |f'|
%   Solvers: ode45, ode113, ode23s, ode15s, ode23t, ode23tb.  The explicit solvers are
%   run only when t_end <= 2e6 (their cost in the stiff phase is ~ t_end/3 steps).
%
%   Needs run_one_sundman.m, flame_theta.m, flame_master.m, flame_exact.m (same folder).
%   Output: results_matlab/sundman_matlab.csv (one row per run, with the MATLAB version).
%   Expected run time: several minutes.

clear; clc;
here   = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'results_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fprintf('MATLAB %s\n', version);

solvers  = {'ode45', 'ode113', 'ode23s', 'ode15s', 'ode23t', 'ode23tb'};
explicit = {'ode45', 'ode113'};
forms    = {'std', 'sund', 'sundlog', 'sundloggen'};
cases    = {2, [1e-2 1e-4 1e-6 1e-8];
            3, [1e-2 1e-3 1e-4]};
tols     = [1e-4 1e-8];

rows = {};
warning('off', 'all');
for c = 1:size(cases, 1)
    q = cases{c, 1};
    for d = cases{c, 2}
        tend = 2 / ((q - 1) * d^(q - 1));
        for tol = tols
            for is = 1:numel(solvers)
                sname = solvers{is};
                if any(strcmp(sname, explicit)) && tend > 2e6, continue; end
                for iform = 1:numel(forms)
                    form = forms{iform};
                    try
                        row = run_one_sundman(sname, q, d, tol, form);
                    catch err
                        fprintf('q=%d d=%.0e tol=%.0e %-8s %-10s FAILED: %s\n', ...
                            q, d, tol, sname, form, err.message);
                        continue
                    end
                    rows{end + 1} = row; %#ok<SAGROW>
                    fprintf('q=%d d=%.0e tol=%.0e %-8s %-10s steps=%7d fevals=%8d maxerr=%.2e D(1/2)=% .2e done=%d %.2fs\n', ...
                        q, d, tol, sname, form, row.nsteps, row.nfevals, row.maxerr, ...
                        row.D_half, row.completed, row.time_s);
                end
            end
        end
    end
end
warning('on', 'all');

T = struct2table([rows{:}]);
writetable(T, fullfile(outdir, 'sundman_matlab.csv'));
fprintf('Results written to %s\n', fullfile(outdir, 'sundman_matlab.csv'));
