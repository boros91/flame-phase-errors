function [row, t] = run_one(sname, q, delta, rtol, atol, warm)
%RUN_ONE  Solve y' = y^q (1-y), y(0) = delta on [0, 2/((q-1) delta^(q-1))] with one
%   MATLAB solver and return statistics, accuracy and phase delay.
%
%   sname : 'ode45','ode23','ode113','ode23s','ode15s','ode15s_BDF','ode23t','ode23tb'
%   warm  : true -> start at y = ywarm at the exact time T_delta(ywarm) (translation
%           property); the slow phase below ywarm is skipped exactly.
%
%   Phase delay of a computed point (t_k, y_k):  D_k = t_k - T_delta(y_k),
%   so that y_k = y_delta(t_k - D_k) exactly.

f    = @(t, y) y.^q .* (1 - y);
tend = 2 / ((q - 1) * delta^(q - 1));
if q == 2, ywarm = 0.01; else, ywarm = 0.05; end

t0 = 0; y0 = delta;
if warm
    y0 = ywarm;
    t0 = flame_theta(ywarm, q) - flame_theta(delta, q);
end

opts = odeset('RelTol', rtol, 'AbsTol', atol);
if strcmp(sname, 'ode15s_BDF')
    fun  = @ode15s;
    opts = odeset(opts, 'BDF', 'on');
else
    fun = str2func(sname);
end

lastwarn('');
tic;
sol = fun(f, [t0 tend], y0, opts);
el = toc;
[wmsg, ~] = lastwarn;

t = sol.x(:);
y = sol.y(:);
completed = abs(t(end) - tend) <= 1e-10 * tend;

ye     = flame_exact(t, delta, q);
maxerr = max(abs(y - ye));

ok = (y > delta * (1 + 1e-12)) & (y < 0.99);
D  = t(ok) - (flame_theta(y(ok), q) - flame_theta(delta, q));
yo = y(ok);
k  = find(yo >= 0.5, 1);
if isempty(k), Dhalf = NaN; else, Dhalf = D(k); end
if isempty(D), maxD = NaN; else, maxD = max(abs(D)); end

th = flame_theta(0.5, q) - flame_theta(delta, q);   % exact time of y = 1/2
ts = t(2:end);

row = struct();
row.q          = q;
row.delta      = delta;
row.solver     = sname;
row.warm       = double(warm);
row.RelTol     = rtol;
row.AbsTol     = atol;
row.nsteps     = getstat(sol, 'nsteps');
row.nfailed    = getstat(sol, 'nfailed');
row.nfevals    = getstat(sol, 'nfevals');
row.npds       = getstat(sol, 'npds');
row.ndecomps   = getstat(sol, 'ndecomps');
row.steps_slow  = sum(ts < th - 20);
row.steps_stiff = sum(ts > th + 20);
row.maxerr     = maxerr;
row.D_half     = Dhalf;
row.maxD       = maxD;
row.time_s     = el;
row.completed  = double(completed);
row.warning    = wmsg;
end

function v = getstat(sol, name)
%GETSTAT  Field of sol.stats, or the obvious default if the solver does not report it
%   (explicit solvers report no Jacobians/LU decompositions -> 0; anything else
%   missing -> NaN; nsteps falls back to the number of accepted points).
v = NaN;
if isfield(sol, 'stats') && isfield(sol.stats, name)
    v = sol.stats.(name);
elseif any(strcmp(name, {'npds', 'ndecomps'}))
    v = 0;
elseif strcmp(name, 'nsteps')
    v = numel(sol.x) - 1;
end
end
