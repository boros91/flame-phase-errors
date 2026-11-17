%RUN_RECIP  The flame family with the unknown w = y^(1-q) (w = 1/y for q = 2).
%
%   Problem:  y' = y^q (1-y), y(0) = delta, 0 <= t <= 2/((q-1) delta^(q-1)).
%   With w = y^(1-q) it becomes  w' = -(q-1) (1 - w^(-1/(q-1))),  w(0) = delta^(1-q).
%   In the slow phase w ~ delta^(1-q) - (q-1) t, so w is a clock: an ABSOLUTE error eps in w
%   is a time shift eps/(q-1), while a RELATIVE error in w is a relative error in y.
%   Same grid as run_sundman.m:
%     q = 2: delta = 1e-2, 1e-4, 1e-6, 1e-8;  q = 3: delta = 1e-2, 1e-3, 1e-4;  tol = 1e-4, 1e-8.
%   Formulations (see run_one_recip.m):
%     std       original equation, RelTol = tol, AbsTol = tol/100
%     reciprel  w-equation, RelTol = tol, AbsTol = tol      (relative control of w)
%     recip     w-equation, RelTol = 100*eps, AbsTol = tol  (absolute control of w)
%   Solvers: ode45, ode113, ode23s, ode15s, ode23t, ode23tb (explicit ones only for t_end <= 2e6).
%
%   Needs run_one_recip.m, flame_theta.m, flame_master.m, flame_exact.m (same folder).
%   Output: results_matlab/recip_matlab.csv (one row per run, with the MATLAB version).
%   Expected run time: a few minutes.

clear; clc;
here   = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'results_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fprintf('MATLAB %s\n', version);

solvers  = {'ode45', 'ode113', 'ode23s', 'ode15s', 'ode23t', 'ode23tb'};
explicit = {'ode45', 'ode113'};
forms    = {'std', 'reciprel', 'recip'};
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
                        row = run_one_recip(sname, q, d, tol, form);
                    catch err
                        fprintf('q=%d d=%.0e tol=%.0e %-8s %-9s FAILED: %s\n', ...
                            q, d, tol, sname, form, err.message);
                        continue
                    end
                    rows{end + 1} = row; %#ok<SAGROW>
                    fprintf('q=%d d=%.0e tol=%.0e %-8s %-9s steps=%7d fevals=%8d maxerr=%.2e D(1/2)=% .2e done=%d %.2fs\n', ...
                        q, d, tol, sname, form, row.nsteps, row.nfevals, row.maxerr, ...
                        row.D_half, row.completed, row.time_s);
                end
            end
        end
    end
end
warning('on', 'all');

T = struct2table([rows{:}]);
writetable(T, fullfile(outdir, 'recip_matlab.csv'));
fprintf('Results written to %s\n', fullfile(outdir, 'recip_matlab.csv'));
