"""Rerun of flame_time_tolerance.jl, recording the Julia and package versions.

Use the same conda environment as before (juliacall installed), then, inside this folder:

    python run_flame_julia_v24.py

Output in the subfolder out_v24/ (nothing existing is overwritten):
    out_v24/time_tolerance_julia.csv   the runs (same format as before)
    out_v24/julia_versions.txt         julia=..., OrdinaryDiffEq=..., OrdinaryDiffEqBDF=..., OrdinaryDiffEqSDIRK=...
Send back the folder out_v24."""
import os
import juliapkg

here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, 'out_v24')
os.makedirs(out, exist_ok=True)

juliapkg.add("OrdinaryDiffEq", "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed")
juliapkg.add("OrdinaryDiffEqBDF", "6ad6398a-0878-4a85-9266-38940aa047c8")
juliapkg.add("OrdinaryDiffEqSDIRK", "2d112036-d095-4a1e-ab9a-08536f3ecdbf")
juliapkg.resolve()

from juliacall import Main as jl  # noqa: E402  (starts Julia)
jl.seval("using Pkg")
vers = {"julia": str(jl.seval("string(VERSION)"))}
for name in ["OrdinaryDiffEq", "OrdinaryDiffEqBDF", "OrdinaryDiffEqSDIRK"]:
    vers[name] = str(jl.seval(
        f'string(only(v.version for v in values(Pkg.dependencies()) if v.name == "{name}"))'))
with open(os.path.join(out, 'julia_versions.txt'), 'w') as fh:
    for k, v in vers.items():
        fh.write(f'{k}={v}\n')
print(vers)

os.chdir(out)                      # the .jl script writes time_tolerance_julia.csv into the current folder
jl.include(os.path.join(here, "flame_time_tolerance.jl"))
print("written:", os.path.join(out, "time_tolerance_julia.csv"), "and julia_versions.txt")
