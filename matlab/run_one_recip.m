function row = run_one_recip(sname, q, delta, tol, form)
%RUN_ONE_RECIP  One MATLAB solver run for y' = y^q (1-y), y(0) = delta,
%   0 <= t <= tend = 2/((q-1) delta^(q-1)), in one of three formulations:
%
%   'std'      : the original equation, RelTol = tol, AbsTol = tol/100.
%   'reciprel' : unknown w = y^(1-q), w' = -(q-1)(1 - w^(-1/(q-1))), RelTol = tol, AbsTol = tol.
%                The test is max(tol*|w|, tol) = tol*|w|: relative control of w, i.e. of y.
%   'recip'    : the same equation, RelTol = 100*eps (the smallest value MATLAB accepts),
%                AbsTol = tol: absolute control of w. Since w ~ delta^(1-q) - (q-1) t in the slow
%                phase, this is absolute control of the time at which y is reached.
%
%   Accuracy is measured at the computed points: maximal error against the exact solution, and
%   the exact phase delay D_k = t_k - T_delta(y_k) at the first point with y_k >= 1/2.

f    = @(y) y.^q .* (1 - y);
tend = 2 / ((q - 1) * delta^(q - 1));
toy  = @(w) max(w, realmin).^(-1 / (q - 1));   % guard: w must stay positive

switch form
    case 'std'
        rt = tol; at = tol / 100;
        F  = @(t, y) f(y);
        u0 = delta;
    case 'reciprel'
        rt = tol; at = tol;
        F  = @(t, w) -(q - 1) * (1 - toy(w));
        u0 = delta^(1 - q);
    case 'recip'
        rt = 100 * eps; at = tol;
        F  = @(t, w) -(q - 1) * (1 - toy(w));
        u0 = delta^(1 - q);
    otherwise
        error('unknown form %s', form);
end
opts = odeset('RelTol', rt, 'AbsTol', at);
fun  = str2func(sname);

lastwarn('');
tic;
sol = fun(F, [0 tend], u0, opts);
el = toc;
[wmsg, ~] = lastwarn;

t = sol.x(:);
if strcmp(form, 'std'), y = sol.y(:); else, y = toy(sol.y(:)); end
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
row.AbsTol    = at;
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
