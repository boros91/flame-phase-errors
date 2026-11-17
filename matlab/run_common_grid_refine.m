%RUN_COMMON_GRID_REFINE  Grid-refinement check of run_common_grid.m: the same runs, with four times as many
%   output times (8001 uniform points, 16001 points on each of the two windows around the fronts).
%   Output: results_matlab/common_grid_matlab_refine4.csv. Run time: a few minutes.
%
%   Description of run_common_grid.m:
%   Nodal versus common-grid error for the standard and the clock formulation
%   (Sect. 4.5 and Table S6 of the paper). For each run the error is measured
%     (a) at the computed points (nodal error, as in the paper), and
%     (b) on a common set of output times, evaluated with deval (the solver's own interpolant):
%         2001 uniform points on [0, tend], 4001 points on [c-40, c+40] around the exact front
%         c = T_delta(1/2), and 4001 points on [cn-40, cn+40] around the computed front cn
%         (first computed point with y >= 1/2).
%   Formulations: 'std' (RelTol = tol, AbsTol = tol/100) and 'recip' = clock w = y^(1-q)
%   (RelTol = 100*eps, AbsTol = tol), as in run_one_recip.m.
%   Problems: q = 2, delta = 1e-4, 1e-6; q = 3, delta = 1e-3, 1e-4; tol = 1e-4, 1e-8.
%   Solvers: ode45 (only for tend <= 2e4), ode23s, ode23t, ode15s.
%   Output: results_matlab/common_grid_matlab.csv (read by code/gen_common_grid.py).
%   Run time: a few minutes.

outdir = fullfile(fileparts(mfilename('fullpath')), 'results_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
cases = {2, 1e-4; 2, 1e-6; 3, 1e-3; 3, 1e-4};
tols = [1e-4, 1e-8];
solvers = {'ode45', 'ode23s', 'ode23t', 'ode15s'};
forms = {'std', 'recip'};
fid = fopen(fullfile(outdir, 'common_grid_matlab_refine4.csv'), 'w');
fprintf(fid, 'q,delta,tol,solver,form,nsteps,nfevals,completed,nodal_err,grid_err,matlab\n');
ver = version;
for c = 1:size(cases, 1)
    q = cases{c, 1}; delta = cases{c, 2};
    tend = 2 / ((q - 1) * delta^(q - 1));
    for tol = tols
        for s = 1:numel(solvers)
            sname = solvers{s};
            if strcmp(sname, 'ode45') && tend > 2e4, continue; end
            for fo = 1:numel(forms)
                form = forms{fo};
                toy = @(w) max(w, realmin).^(-1 / (q - 1));
                if strcmp(form, 'std')
                    F = @(t, y) y.^q .* (1 - y); u0 = delta; rt = tol; at = tol / 100;
                    conv = @(u) u;
                    J = @(t, y) q * y.^(q - 1) - (q + 1) * y.^q;
                else
                    F = @(t, w) -(q - 1) * (1 - toy(w)); u0 = delta^(1 - q); rt = 100 * eps; at = tol;
                    conv = toy;
                    J = @(t, w) -max(w, realmin).^(-q / (q - 1));
                end
                opts = odeset('RelTol', rt, 'AbsTol', at, 'Jacobian', J);
                lastwarn('');
                sol = feval(sname, F, [0 tend], u0, opts);
                t = sol.x(:); y = conv(sol.y(:));
                completed = abs(t(end) - tend) <= 1e-10 * tend;
                nodal = max(abs(y - flame_exact(t, delta, q)));
                cex = flame_theta(0.5, q) - flame_theta(delta, q);
                k = find(y >= 0.5, 1);
                if isempty(k), cn = cex; else, cn = t(k); end
                tg = unique([linspace(0, tend, 8001), linspace(cex - 40, cex + 40, 16001), ...
                             linspace(cn - 40, cn + 40, 16001)]);
                tg = tg(tg >= 0 & tg <= t(end));
                yg = conv(deval(sol, tg));
                grid = max(abs(yg(:) - flame_exact(tg(:), delta, q)));
                fprintf(fid, '%d,%.3g,%.3g,%s,%s,%d,%d,%d,%.6e,%.6e,%s\n', q, delta, tol, sname, form, ...
                        sol.stats.nsteps, sol.stats.nfevals, completed, nodal, grid, ver);
                fprintf('q=%d delta=%.0e tol=%.0e %-6s %-5s nodal %.3e grid %.3e\n', q, delta, tol, ...
                        sname, form, nodal, grid);
            end
        end
    end
end
fclose(fid);
fprintf('written: %s\n', fullfile(outdir, 'common_grid_matlab_refine4.csv'));
