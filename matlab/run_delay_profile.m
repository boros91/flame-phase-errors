%RUN_DELAY_PROFILE  Accepted steps (t_k, y_k) of the MATLAB solvers in Moler's setting,
%   y' = y^2 - y^3, y(0) = 1e-4, 0 <= t <= 2e4, RelTol = 1e-4, AbsTol = 1e-6,
%   for the figure showing how the phase delay D_k = t_k - T(y_k) accumulates.
%   Also ode15s and ode23s with time as a dependent variable, unknowns [log y; t],
%   rho = y, RelTol = 100*eps, AbsTol = [1e-4; 1e-4] (form 'sundlog' of run_one_sundman.m).
%
%   Output: results_matlab/delay_profile_matlab.csv with columns solver, form, k, t, y.
%   Run time: a few seconds.

clear; clc;
here   = fileparts(mfilename('fullpath'));
outdir = fullfile(here, 'results_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fprintf('MATLAB %s\n', version);

delta = 1e-4; tend = 2 / delta; tol = 1e-4;
f = @(y) y.^2 - y.^3;
solvers = {'ode45', 'ode23', 'ode113', 'ode23s', 'ode15s', 'ode15s_BDF', 'ode23t', 'ode23tb'};

S = {}; F = {}; K = []; T = []; Y = [];
for i = 1:numel(solvers)
    s = solvers{i};
    opts = odeset('RelTol', 1e-4, 'AbsTol', 1e-6);
    if strcmp(s, 'ode15s_BDF'), fun = @ode15s; opts = odeset(opts, 'BDF', 'on');
    else, fun = str2func(s); end
    sol = fun(@(t, y) f(y), [0 tend], delta, opts);
    n = numel(sol.x);
    S = [S; repmat({s}, n, 1)]; F = [F; repmat({'std'}, n, 1)]; %#ok<AGROW>
    K = [K; (0:n - 1)']; T = [T; sol.x(:)]; Y = [Y; sol.y(:)]; %#ok<AGROW>
    fprintf('%-11s std      steps = %d\n', s, n - 1);
end

for s = {'ode15s', 'ode23s'}
    fun = str2func(s{1});
    G = @(tau, u) [f(exp(u(1))) ./ (exp(u(1)) .* exp(u(1))); 1 ./ exp(u(1))];
    opts = odeset('RelTol', 100 * eps, 'AbsTol', [tol; tol], 'Events', @(tau, u) ev(tau, u, tend));
    sol = fun(G, [0 10 * tend + 100], [log(delta); 0], opts);
    n = numel(sol.x);
    S = [S; repmat(s, n, 1)]; F = [F; repmat({'sundlog'}, n, 1)]; %#ok<AGROW>
    K = [K; (0:n - 1)']; T = [T; sol.y(2, :)']; Y = [Y; exp(sol.y(1, :))']; %#ok<AGROW>
    fprintf('%-11s sundlog  steps = %d\n', s{1}, n - 1);
end

tab = table(S, F, K, T, Y, 'VariableNames', {'solver', 'form', 'k', 't', 'y'});
tab.matlab = repmat({version}, height(tab), 1);
writetable(tab, fullfile(outdir, 'delay_profile_matlab.csv'));
fprintf('Results written to %s\n', fullfile(outdir, 'delay_profile_matlab.csv'));

function [value, isterminal, direction] = ev(~, u, tend)
value = u(2) - tend; isterminal = 1; direction = 1;
end
