%RUN_TOL_SWEEP_PAR  Parallel, resumable version of run_tol_sweep.m (same runs, same output file).
%
%   Can the phase error be removed by tightening the tolerances?
%   Part A: solvers ode45, ode23, ode23s, ode23t, ode15s; delta = 1e-2 ... 1e-6;
%           RelTol = 1e-3, ..., 1e-13, 100*eps; AbsTol = RelTol/100.
%   Part B: delta = 1e-4, solvers ode45, ode23s, ode15s; RelTol = 1e-4 ... 1e-12 x AbsTol = 1e-4 ... 1e-20.
%
%   * Runs in parallel with parfor if the Parallel Computing Toolbox is installed (otherwise parfor
%     runs as an ordinary loop).
%   * Every run is stopped after TMAX seconds (run_one_capped.m); such runs are kept and marked
%     'timeout' (completed = 0).
%   * Every finished run is saved at once in results_matlab/tol_sweep_parts/; if the script is
%     interrupted and started again, the runs already done are skipped.
%   * At the end all parts are collected in results_matlab/tol_sweep_matlab.csv.
%
%   Needs run_one_capped.m, flame_theta.m, flame_master.m, flame_exact.m (same folder).

clear; clc;
TMAX = 300;                                   % seconds per run
here    = fileparts(mfilename('fullpath'));
outdir  = fullfile(here, 'results_matlab');
partdir = fullfile(outdir, 'tol_sweep_parts');
if ~exist(partdir, 'dir'), mkdir(partdir); end
fprintf('MATLAB %s\n', version);
q = 2;

% ---------------- the list of runs: {part, solver, delta, RelTol, AbsTol}
jobs = {};
for d = [1e-2 1e-3 1e-4 1e-5 1e-6]
    for s = {'ode45', 'ode23', 'ode23s', 'ode23t', 'ode15s'}
        for rt = [10.^(-(3:13)), 100 * eps]
            jobs(end+1, :) = {'A', s{1}, d, rt, rt / 100}; %#ok<SAGROW>
        end
    end
end
for s = {'ode45', 'ode23s', 'ode15s'}
    for rt = 10.^(-(4:2:12))
        for at = 10.^(-(4:2:20))
            jobs(end+1, :) = {'B', s{1}, 1e-4, rt, at}; %#ok<SAGROW>
        end
    end
end
njobs = size(jobs, 1);
fprintf('%d runs, %d already done\n', njobs, numel(dir(fullfile(partdir, 'job_*.csv'))));

% ---------------- run (in parallel if possible)
parfor j = 1:njobs
    file = fullfile(partdir, sprintf('job_%04d.csv', j));
    if exist(file, 'file'), continue; end
    [part, sname, d, rt, at] = jobs{j, :};
    warning('off', 'all');
    try
        row = run_one_capped(sname, q, d, rt, at, false, TMAX);
    catch err
        row = struct('q', q, 'delta', d, 'solver', sname, 'warm', 0, 'RelTol', rt, 'AbsTol', at, ...
            'nsteps', NaN, 'nfailed', NaN, 'nfevals', NaN, 'npds', NaN, 'ndecomps', NaN, ...
            'steps_slow', NaN, 'steps_stiff', NaN, 'maxerr', NaN, 'D_half', NaN, 'maxD', NaN, ...
            'time_s', NaN, 'completed', 0, 'warning', ['error: ' err.message]);
    end
    row.part = part; row.job = j;
    if isempty(row.warning), row.warning = 'none'; end
    row.warning = strrep(strrep(row.warning, ',', ';'), newline, ' ');
    write_part(file, row);
    fprintf('%4d %s %-7s delta %.0e RelTol %.1e AbsTol %.0e: %8d steps, D = %11.4g, max err %.2e  %s\n', ...
            j, part, sname, d, rt, at, row.nsteps, row.D_half, row.maxerr, row.warning);
end

% ---------------- collect
files = dir(fullfile(partdir, 'job_*.csv'));
T = table();
for k = 1:numel(files)
    Tk = readtable(fullfile(partdir, files(k).name), 'TextType', 'string', 'Delimiter', ',');
    T = [T; Tk]; %#ok<AGROW>
end
T = sortrows(T, 'job');
T.matlab = repmat(string(version), height(T), 1);
writetable(T, fullfile(outdir, 'tol_sweep_matlab.csv'));
fprintf('\nwritten: %s (%d of %d runs)\n', fullfile(outdir, 'tol_sweep_matlab.csv'), height(T), njobs);

function write_part(file, row)
writetable(struct2table(row, 'AsArray', true), file);
end
