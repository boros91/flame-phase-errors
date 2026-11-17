%RUN_WORKPREC  Work-precision study for the flame problem y' = y^2 - y^3 (q = 2).
%
%   For every solver, delta in {1e-4, 1e-6} and tol = 1e-3, ..., 1e-12, three strategies:
%     standard : RelTol = tol,        AbsTol = tol/100       (the benchmark ratio)
%     scaled   : RelTol = tol*delta,  AbsTol = tol*delta^2/100
%     warm     : start on the exact translate at y = 0.01, RelTol = tol, AbsTol = tol/100
%   Records steps, function evaluations, Jacobians, LU decompositions, the maximal error at
%   the computed points and the phase delay at y = 1/2.
%
%   Needs run_one.m, flame_theta.m, flame_master.m, flame_exact.m (same folder).
%   Output: results_matlab/workprec_matlab.csv.  Run time: about 10-20 minutes
%   (explicit solvers are run for delta = 1e-4 only).
%   MATLAB raises RelTol below 100*eps to 100*eps and issues a warning; such runs are kept
%   and the effective RelTol is recorded.

clear; clc;
here   = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'results_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end

solvers  = {'ode45', 'ode113', 'ode23s', 'ode15s', 'ode23t', 'ode23tb'};
explicit = {'ode45', 'ode113'};
strategies = {'standard', 'scaled', 'warm'};
tols   = 10.^(-(3:12));
deltas = [1e-4 1e-6];
q = 2;

rows = {};
warning('off', 'all');
for d = deltas
    for is = 1:numel(solvers)
        sname = solvers{is};
        if any(strcmp(sname, explicit)) && d < 1e-4, continue; end
        for ist = 1:numel(strategies)
            strat = strategies{ist};
            for tol = tols
                switch strat
                    case 'standard', rt = tol;     at = tol / 100;        warm = false;
                    case 'scaled',   rt = tol * d; at = tol * d^2 / 100;  warm = false;
                    case 'warm',     rt = tol;     at = tol / 100;        warm = true;
                end
                rt = max(rt, 100 * eps);
                try
                    row = run_one(sname, q, d, rt, at, warm);
                catch err
                    fprintf('%s %s tol=%.0e failed: %s\n', sname, strat, tol, err.message);
                    continue
                end
                row.strategy = strat;
                row.tol = tol;
                rows{end + 1} = row; %#ok<SAGROW>
                fprintf('d=%.0e %-8s %-8s tol=%.0e steps=%7d fevals=%8d maxerr=%.2e D(1/2)=% .2e\n', ...
                    d, sname, strat, tol, row.nsteps, row.nfevals, row.maxerr, row.D_half);
            end
        end
    end
end
warning('on', 'all');

T = struct2table([rows{:}]);
writetable(T, fullfile(outdir, 'workprec_matlab.csv'));
fprintf('Results written to %s\n', fullfile(outdir, 'workprec_matlab.csv'));
