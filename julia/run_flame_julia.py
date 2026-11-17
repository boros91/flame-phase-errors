"""Run flame_time_tolerance.jl from Python, using a Julia installed inside the current conda env.

    conda create -n julia python=3.11 -y
    conda activate julia
    pip install juliacall
    python run_flame_julia.py

The first run downloads Julia (~200 MB) and OrdinaryDiffEq and precompiles them: 10-20 minutes.
Later runs start in under a minute. Output: time_tolerance_julia.csv next to this file."""
import os
import juliapkg

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)

# Julia packages needed by the script (name, UUID from the General registry)
juliapkg.add("OrdinaryDiffEq", "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed")
juliapkg.add("OrdinaryDiffEqBDF", "6ad6398a-0878-4a85-9266-38940aa047c8")
juliapkg.add("OrdinaryDiffEqSDIRK", "2d112036-d095-4a1e-ab9a-08536f3ecdbf")
juliapkg.resolve()
print("Julia executable:", juliapkg.executable())

from juliacall import Main as jl  # noqa: E402  (starts Julia)
print("Julia version:", jl.seval("string(VERSION)"))
jl.include(os.path.join(here, "flame_time_tolerance.jl"))
