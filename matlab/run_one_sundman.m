function row = run_one_sundman(sname, q, delta, tol, form)
%RUN_ONE_SUNDMAN  One MATLAB solver run for y' = y^q (1-y), y(0) = delta,
%   0 <= t <= tend = 2/((q-1) delta^(q-1)), in one of four formulations:
%
%   'std'      : the original equation, RelTol = tol, AbsTol = tol/100.
%   'sund'     : time as a dependent variable (Sundman transformation dt/dtau = 1/rho(y),
%                rho = y^(q-1)), unknowns [y; t], MATLAB's own tolerance interface:
%                RelTol = tol (one scalar for both components), AbsTol = [tol/100; tol].
%                The tolerance on t is then max(tol*|t|, tol), i.e. relative.
%   'sundlog'  : the same transformation, unknowns [z; t] with z = log(y),
%                RelTol = 100*eps (the smallest value MATLAB accepts), AbsTol = [tol; tol].
%                An absolute tolerance on z is a relative tolerance on y, so this gives
%                relative control of y and ABSOLUTE control of t without changing the solver.
%   'sundloggen': as 'sundlog' with the generic rate rho = |f(y)|/|y| + |f'(y)|.
%
%   The transformed runs stop at t = tend (Events). Accuracy is measured at the computed
%   points: maximal error against the exact solution, and the exact phase delay
%   D_k = t_k - T_delta(y_k) at the first point with y_k >= 1/2.

f    = @(y) y.^q .* (1 - y);
fp   = @(y) q * y.^(q - 1) - (q + 1) * y.^q;
tend = 2 / ((q - 1) * delta^(q - 1));

switch form
    case 'std'
        rt = tol; at = tol / 100;
        F  = @(t, y) f(y);
        u0 = delta; span = [0 tend];
        opts = odeset('RelTol', rt, 'AbsTol', at);
    case 'sund'
        rt = tol; at = [tol / 100; tol];
        F  = @(tau, u) [f(u(1)) ./ u(1).^(q - 1); 1 ./ u(1).^(q - 1)];
        u0 = [delta; 0]; span = [0 10 * tend + 100];
        opts = odeset('RelTol', rt, 'AbsTol', at, 'Events', @(tau, u) ev_tend(tau, u, tend));
    case {'sundlog', 'sundloggen'}
        rt = 100 * eps; at = [tol; tol];
        if strcmp(form, 'sundlog')
            rho = @(y) y.^(q - 1);
        else
            rho = @(y) abs(f(y)) ./ abs(y) + abs(fp(y));
        end
        F  = @(tau, u) [f(exp(u(1))) ./ (exp(u(1)) .* rho(exp(u(1)))); 1 ./ rho(exp(u(1)))];
        u0 = [log(delta); 0]; span = [0 10 * tend + 100];
        opts = odeset('RelTol', rt, 'AbsTol', at, 'Events', @(tau, u) ev_tend(tau, u, tend));
    otherwise
        error('unknown form %s', form);
end

if strcmp(sname, 'ode15s_BDF')
    fun  = @ode15s;
    opts = odeset(opts, 'BDF', 'on');
else
    fun = str2func(sname);
end

lastwarn('');
tic;
sol = fun(F, span, u0, opts);
el = toc;
[wmsg, ~] = lastwarn;

switch form
    case 'std'
        t = sol.x(:); y = sol.y(:);
    case 'sund'
        t = sol.y(2, :)'; y = sol.y(1, :)';
    otherwise
        t = sol.y(2, :)'; y = exp(sol.y(1, :)');
end
completed = t(end) >= tend * (1 - 1e-10);

ye     = flame_exact(t, delta, q);
maxerr = max(abs(y - ye));
ok = (y > delta * (1 + 1e-12)) & (y < 0.99);
D  = t(ok) - (flame_theta(y(ok), q) - flame_theta(delta, q));
yo = y(ok);
k  = find(yo >= 0.5, 1);
if isempty(k), Dhalf = NaN; else, Dhalf = D(k); end
if isempty(D), maxD = NaN; else, maxD = max(abs(D)); end

row = struct();
row.q         = q;
row.delta     = delta;
row.tol       = tol;
row.solver    = sname;
row.form      = form;
row.RelTol    = rt;
row.AbsTol_y  = at(1);
row.AbsTol_t  = at(end);
row.nsteps    = getstat(sol, 'nsteps');
row.nfailed   = getstat(sol, 'nfailed');
row.nfevals   = getstat(sol, 'nfevals');
row.npds      = getstat(sol, 'npds');
row.ndecomps  = getstat(sol, 'ndecomps');
row.maxerr    = maxerr;
row.D_half    = Dhalf;
row.maxD      = maxD;
row.time_s    = el;
row.completed = double(completed);
row.warning   = wmsg;
row.matlab    = version;
end

function [value, isterminal, direction] = ev_tend(~, u, tend)
value = u(2) - tend; isterminal = 1; direction = 1;
end

function v = getstat(sol, name)
v = NaN;
if isfield(sol, 'stats') && isfield(sol.stats, name)
    v = sol.stats.(name);
elseif any(strcmp(name, {'npds', 'ndecomps'}))
    v = 0;
elseif strcmp(name, 'nsteps')
    v = numel(sol.x) - 1;
end
end
