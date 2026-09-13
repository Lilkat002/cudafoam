# Benchmarks

Meshes are the OpenFOAM `pitzDaily` tutorial geometry scaled 25× with graded
refinement:

- `blockMeshDict_large` — 305,625 cells
- `blockMeshDict_huge` — 1,222,500 cells (same geometry, 2× cells in x and y)

`results/` holds the raw solver logs behind the numbers in the top-level
README (`v4_*.log`; the `log.*` files are from the earlier v3 solver, kept
for the before/after record).

## Reproducing

All runs: OpenFOAM 12, `incompressibleFluid/pitzDaily` tutorial as the base
case, single rank. For each mesh size:

```sh
cp -r $FOAM_TUTORIALS/incompressibleFluid/pitzDaily case
cd case
cp /path/to/blockMeshDict_large system/blockMeshDict   # or _huge
blockMesh

# identical, step-for-step comparable runs:
foamDictionary -entry adjustTimeStep -set no  system/controlDict
foamDictionary -entry deltaT        -set 5e-07    system/controlDict  # 2.5e-07 for huge
foamDictionary -entry endTime       -set 2.5e-06  system/controlDict  # 1.25e-06 for huge
foamDictionary -entry writeControl  -set runTime  system/controlDict
foamDictionary -entry writeInterval -set 2.5e-06  system/controlDict  # 1.25e-06 for huge

# make sure every solve actually converges (a truncated solve makes the
# timing comparison meaningless and the fields diverge):
foamDictionary -entry solvers/p/maxIter -set 5000 system/fvSolution

# CPU PCG reference:
foamDictionary -entry solvers/p/solver -set PCG system/fvSolution
foamDictionary -entry solvers/p/preconditioner -set DIC system/fvSolution
foamRun

# GPU:
foamDictionary -entry solvers/p/solver -set cudaPCG system/fvSolution
foamDictionary -entry libs -set '("libcudaPCG.so")' system/controlDict
foamRun

# CPU GAMG baseline:
foamDictionary -entry solvers/p/solver -set GAMG system/fvSolution
foamDictionary -entry solvers/p/smoother -set DICGaussSeidel system/fvSolution
foamRun
```

The small-mesh benchmark is simply the unmodified tutorial (12,225 cells,
`endTime 0.1`, 1000 steps) with the solver switched as above.

Compare wall time via the final `ExecutionTime` line, and validate
correctness by diffing the written pressure fields between the CPU and GPU
runs at the same output time (they should agree to a small fraction of the
field range).
