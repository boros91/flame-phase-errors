# Flame problem y' = y^2 - y^3, y(0) = delta, 0 <= t <= 2/delta, with time as a dependent variable:
#     dy/dtau = y(1-y),   dt/dtau = 1/y          (Sundman transformation, rho = y)
# and a separate ABSOLUTE tolerance on t. Julia's OrdinaryDiffEq accepts component-wise
# reltol/abstol vectors for all its solvers, explicit and stiff.
#
# Setup (once):   using Pkg; Pkg.add(["OrdinaryDiffEq", "OrdinaryDiffEqBDF", "OrdinaryDiffEqSDIRK"])
# Run:            julia flame_time_tolerance.jl        -> writes time_tolerance_julia.csv
#
# For every solver and delta: the standard formulation, the transformed one with the same
# tolerance on t as on y, and the transformed one with abstol_t = 1e-2 ... 1e-10 (reltol_t = 1e-13).
# Recorded: front delay D at y = 1/2 (D_k = t_k - T_delta(y_k), exact), maximal error, f-evaluations.

using OrdinaryDiffEq, OrdinaryDiffEqBDF, OrdinaryDiffEqSDIRK, Printf
# (recent OrdinaryDiffEq versions export only a few solvers; QNDF is in OrdinaryDiffEqBDF,
#  TRBDF2 and KenCarp4 in OrdinaryDiffEqSDIRK)

# ---------------------------------------------------------------- exact solution (q = 2)
Θ(y) = log(y) - log1p(-y) - 1 / y
Tδ(y, δ) = Θ(y) - Θ(δ)
θlogit(l) = l - (1 + exp(-l))                       # Θ(y) with y = 1/(1+e^{-l})
function master(s)
    lo, hi = -60.0, 60.0
    s < -1 && (lo = min(lo, log(1 / (-s)) - 2))
    for _ in 1:200
        mid = (lo + hi) / 2
        θlogit(mid) < s ? (lo = mid) : (hi = mid)
    end
    l = (lo + hi) / 2
    for _ in 1:3
        y = 1 / (1 + exp(-l))
        l -= (θlogit(l) - s) * y                     # dΘ/dl = 1/y
    end
    1 / (1 + exp(-l))
end
exact(t, δ) = master(t + Θ(δ))

function measure(t, y, δ)
    err = maximum(abs.(y .- exact.(t, δ)))
    ok = (y .> δ * (1 + 1e-12)) .& (y .< 0.99)
    tk, yk = t[ok], y[ok]
    k = findfirst(>=(0.5), yk)
    D = k === nothing ? NaN : tk[k] - Tδ(yk[k], δ)
    D, err
end

# ---------------------------------------------------------------- the two formulations
function standard(δ, alg; reltol = 1e-6, abstol = 1e-8)
    prob = ODEProblem((y, p, t) -> y^2 - y^3, δ, (0.0, 2 / δ))
    sol = solve(prob, alg; reltol, abstol)
    D, err = measure(sol.t, sol.u, δ)
    D, err, sol.stats.nf, string(sol.retcode)
end

function transformed(δ, alg; reltol_t, abstol_t, reltol_y = 1e-6, abstol_y = 1e-8)
    tend = 2 / δ
    function rhs!(du, u, p, τ)
        du[1] = u[1] * (1 - u[1])
        du[2] = 1 / u[1]
    end
    cb = ContinuousCallback((u, τ, integ) -> u[2] - tend, terminate!)
    prob = ODEProblem(rhs!, [δ, 0.0], (0.0, 1e3 * tend))
    sol = solve(prob, alg; reltol = [reltol_y, reltol_t], abstol = [abstol_y, abstol_t], callback = cb)
    t = [u[2] for u in sol.u]; y = [u[1] for u in sol.u]
    D, err = measure(t, y, δ)
    D, err, sol.stats.nf, string(sol.retcode)
end

# ---------------------------------------------------------------- experiment
solvers = [("Tsit5", Tsit5(), :explicit), ("Vern7", Vern7(), :explicit),
           ("Rosenbrock23", Rosenbrock23(), :stiff), ("Rodas5P", Rodas5P(), :stiff), ("FBDF", FBDF(), :stiff),
           ("QNDF", QNDF(), :stiff), ("TRBDF2", TRBDF2(), :stiff), ("KenCarp4", KenCarp4(), :stiff)]

open("time_tolerance_julia.csv", "w") do io
    println(io, "delta,solver,form,abstol_t,D_half,maxerr,nf,retcode")
    for δ in (1e-4, 1e-6), (name, alg, kind) in solvers
        kind == :explicit && δ < 1e-4 && continue
        runs = Any[("standard", NaN, () -> standard(δ, alg)),
                   ("tau_same_tol", 1e-8, () -> transformed(δ, alg; reltol_t = 1e-6, abstol_t = 1e-8))]
        for at in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
            push!(runs, ("tau_abs_tol_t", at, () -> transformed(δ, alg; reltol_t = 1e-13, abstol_t = at)))
        end
        for (form, at, run) in runs
            D, err, nf, rc = try
                run()
            catch e
                NaN, NaN, -1, "error: " * replace(sprint(showerror, e)[1:min(end, 60)], "," => ";")
            end
            @printf("δ=%.0e %-9s %-14s abstol_t=%-7.0e D=% .2e err=%.2e nf=%6d %s\n",
                    δ, name, form, at, D, err, nf, rc)
            @printf(io, "%.0e,%s,%s,%.0e,%.6e,%.6e,%d,%s\n", δ, name, form, at, D, err, nf, rc)
            flush(io)
        end
    end
end
println("written time_tolerance_julia.csv")
