%RUN_MOLER_FLAME  Moler's flame-problem comparison of MATLAB ODE solvers, extended.
%
%   Problem:  y' = y^q (1-y),  y(0) = delta,  0 <= t <= 2/((q-1) delta^(q-1))
%             (q = 2: flame problem of Moler / Shampine; q = 3: second example).
%
%   For every solver, delta and tolerance pair the script records MATLAB's own
%   statistics (sol.stats), the number of steps in the slow phase and in the stiff
%   phase, the maximal error against the exact solution, and the phase delay
%   D_k = t_k - T_delta(y_k) of the computed points.
%
%   Output (folder results_matlab next to this file):
%     moler_matlab.csv          one row per run
%     steps_q2_delta1e-4.csv    accepted step points t_k for delta = 1e-4,
%                               Moler's tolerance (RelTol = 1e-4), every solver
%
%   Expected run time: a few minutes (the explicit solvers at delta = 1e-6 need
%   several hundred thousand steps).

clear; clc;
here   = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'results_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% --- sanity check of the exact solution: T_delta(y_exact(t)) = t ----------------
for q = [2 3]
    d  = 1e-3;
    tt = linspace(0, 2 / ((q - 1) * d^(q - 1)), 1001)';
    yy = flame_exact(tt, d, q);
    ok = yy < 1 - 1e-6;
    res = max(abs(flame_theta(yy(ok), q) - flame_theta(d, q) - tt(ok))) / abs(flame_theta(d, q));
    fprintf('exact-solution check q=%d: max relative time residual %.2e\n', q, res);
    assert(res < 1e-11, 'exact solution check failed');
end

solvers  = {'ode45', 'ode23', 'ode113', 'ode23s', 'ode15s', 'ode15s_BDF', 'ode23t', 'ode23tb'};
explicit = {'ode45', 'ode23', 'ode113'};
warmset  = {'ode45', 'ode23s', 'ode15s'};
tolsets  = [1e-4 1e-6;      % Moler 2014: RelTol = 1e-4, default AbsTol = 1e-6
            1e-8 1e-12;
            1e-12 1e-20];
cases = {2, [2e-2 1e-2 1e-3 1e-4 1e-5 1e-6];
         3, [1e-1 1e-2 1e-3 1e-4 1e-5]};

rows = {};
stepdump = {};
warning('off', 'backtrace');   % keep solver warnings short
for c = 1:size(cases, 1)
    q = cases{c, 1};
    for d = cases{c, 2}
        for it = 1:size(tolsets, 1)
            rt = tolsets(it, 1); at = tolsets(it, 2);
            for is = 1:numel(solvers)
                sname = solvers{is};
                isexp = any(strcmp(sname, explicit));
                % explicit solvers on q = 3 with delta < 1e-3: stiff phase of length
                % ~1/delta^2 would need > 1e7 steps
                if isexp && q == 3 && d < 1e-3, continue; end
                for warm = [false true]
                    if warm && (~any(strcmp(sname, warmset)) || d >= 1e-2), continue; end
                    [row, t] = run_one(sname, q, d, rt, at, warm);
                    rows{end + 1} = row; %#ok<SAGROW>
                    fprintf('q=%d d=%.0e tol=%.0e %-10s warm=%d steps=%7d (slow %5d, stiff %7d) maxerr=%.2e D(1/2)=% .3e  %.2fs\n', ...
                        q, d, rt, sname, warm, row.nsteps, row.steps_slow, row.steps_stiff, ...
                        row.maxerr, row.D_half, row.time_s);
                    if q == 2 && d == 1e-4 && it == 1 && ~warm
                        stepdump{end + 1} = {sname, t}; %#ok<SAGROW>
                    end
                end
            end
        end
    end
end
warning('on', 'backtrace');

T = struct2table([rows{:}]);
writetable(T, fullfile(outdir, 'moler_matlab.csv'));

fid = fopen(fullfile(outdir, 'steps_q2_delta1e-4.csv'), 'w');
fprintf(fid, 'solver,t\n');
for k = 1:numel(stepdump)
    tk = stepdump{k}{2};
    for j = 1:numel(tk)
        fprintf(fid, '%s,%.17g\n', stepdump{k}{1}, tk(j));
    end
end
fclose(fid);

% --- Moler's table (delta = 1e-4, RelTol = 1e-4) -------------------------------
fprintf('\nMoler 2014 setting (q = 2, delta = 1e-4, RelTol = 1e-4):\n');
sel = T.q == 2 & T.delta == 1e-4 & T.RelTol == 1e-4 & T.warm == 0;
disp(T(sel, {'solver', 'nsteps', 'nfailed', 'nfevals', 'npds', 'ndecomps', ...
             'steps_slow', 'steps_stiff', 'maxerr', 'D_half'}));
fprintf('Results written to %s\n', outdir);
