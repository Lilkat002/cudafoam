# cudaFoam

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![OpenFOAM 12](https://img.shields.io/badge/OpenFOAM-12-darkgreen.svg)](https://openfoam.org)
[![CUDA 12/13](https://img.shields.io/badge/CUDA-12%20%7C%2013-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

Native NVIDIA GPU acceleration for OpenFOAM's pressure solver — a drop-in
`lduMatrix` solver plugin, no changes to OpenFOAM itself.

**[How it works](#how-it-works)** ·
**[Build](#build)** ·
**[Use](#use)** ·
**[Benchmarks](#benchmarks)** ·
**[Limitations](#limitations)** ·
**[Roadmap](#roadmap)**

On a single A10G (AWS g5.xlarge), the GPU solver is faster than **every** CPU
option in stock OpenFOAM 12, including GAMG, at identical tolerances:

| Mesh (pitzDaily, incompressible) | CPU PCG-DIC | CPU GAMG | **cudaPCG (GPU)** |
|---|---|---|---|
| 12,225 cells, 1000 steps | 93.1 s | — | **65.1 s** |
| 305,625 cells, 5 steps | 56.0 s | 17.2 s | **13.3 s** |
| 1,222,500 cells, 5 steps | 498 s | 95.0 s | **88.3 s** |

That is **5.6×** over the equivalent CPU solver (PCG-DIC) at 1.2M cells, and
still ahead of GAMG — the strongest CPU baseline. Accuracy is validated on
every run: pressure fields match the CPU reference to 0.017–0.06 % of the
field range on the small/medium meshes; at 1.2M cells the GPU–CPU difference
(0.56 %) is the same magnitude as the GAMG–PCG difference (0.46 %), i.e.
within the scatter between any two converged solvers at the same tolerance.
Raw logs for every number above are in [`benchmark/results/`](benchmark/results/).

## How it works

`cudaPCG` registers itself with OpenFOAM's runtime solver-selection table, so
it is selected from `fvSolution` like any built-in solver. Internally:

- **LDU → CSR once.** OpenFOAM's LDU addressing is converted to CSR on the
  first call and cached; subsequent solves only re-gather coefficient values
  on the GPU (the mesh structure doesn't change, the values do).
- **SpMV via cuSPARSE** for all matrix–vector products.
- **Multicolor symmetric Gauss-Seidel preconditioner.** Cells are greedily
  graph-colored at setup (2–3 colors on typical FV meshes); each triangular
  sweep is then one fully parallel kernel launch per color. This avoids
  cuSPARSE's level-scheduled triangular solves, which serialize into
  wavefronts and were measured *slower* than the CPU. Sweeps run in **FP32**
  (the preconditioner only needs to be approximate — CG itself stays FP64);
  measured iteration-count cost of FP32: zero. Set `CUDAPCG_FP64_PRECOND=1`
  to force FP64 sweeps.
- **One host synchronization per iteration.** All PCG scalars (α, β, dot
  products, residual) live on the device: dot products are
  block-reduce + atomicAdd kernels, α/β are computed by single-thread device
  kernels, and the solution/residual update is fused with the |r| reduction.
  The host reads back 16 bytes per iteration for the convergence test.
- **Exact OpenFOAM semantics.** The iteration, normFactor, convergence tests
  (`tolerance`, `relTol`, `minIter`, `maxIter`) and the singularity guard
  replicate `PCG.C` / `lduMatrixSolver.C`, so residuals and iteration counts
  are directly comparable with the CPU solver.

## Requirements

- OpenFOAM 12 (openfoam.org packaging; other versions likely need minor
  adjustments to the solver-table registration)
- CUDA toolkit 12.x or 13.x with cuSPARSE
- An NVIDIA GPU (default build targets `sm_86` — Ampere; override with
  `ARCH=sm_XX ./Allwmake`)

## Build

```sh
source /opt/openfoam12/etc/bashrc
cd cudaPCG
./Allwmake            # ARCH=sm_90 ./Allwmake for Hopper, etc.
```

This compiles `cudaKernels.cu` with nvcc into `$FOAM_USER_LIBBIN`, then links
the plugin with wmake. Note: wmake does not track the CUDA object as a
dependency — after editing `cudaKernels.cu`, remove
`$FOAM_USER_LIBBIN/libcudaPCG.so` (or `touch cudaPCG.C`) so the library is
relinked; `Allwmake` recompiles the kernels every time regardless.

## Use

In `system/controlDict`:

```
libs ("libcudaPCG.so");
```

In `system/fvSolution`, select it like any other solver:

```
p
{
    solver          cudaPCG;
    tolerance       1e-07;
    relTol          0.01;
}
```

No `preconditioner` entry is needed (the GPU preconditioner is built in).
Everything else about the case is unchanged.

## Limitations

- **Symmetric matrices only** (it registers as a symmetric-matrix solver) —
  i.e. pressure/Laplacian-type equations. That is where the solver time goes
  in incompressible CFD, so this is the case that matters.
- **Cyclic (periodic) boundaries are fully supported on the GPU**: their
  coupling is folded into the sparse matrix as extra off-diagonal entries
  (validated on the periodic `channel395` tutorial — the first solve's
  initial residual matches CPU PCG exactly, and final fields agree within
  the same scatter as GAMG-vs-PCG).
- **Single rank, single GPU.** On decomposed runs (`decomposePar`) or with
  non-conformal couplings, the solver detects the interfaces it cannot
  apply and automatically falls back to OpenFOAM's own PCG — results are
  always correct, there is just no GPU speedup. MPI support is the natural
  next step.
- Solves at tiny mesh sizes (≲10k cells) are launch-latency bound; the GPU
  advantage grows with mesh size.

## Benchmarks

`benchmark/` contains the scaled pitzDaily meshes used for the numbers above
(`blockMeshDict_large` ≈ 306k cells, `blockMeshDict_huge` ≈ 1.22M cells) and
the raw logs (`results/`). To reproduce: copy the OpenFOAM 12
`incompressibleFluid/pitzDaily` tutorial, drop one of these dicts in as
`system/blockMeshDict`, run `blockMesh`, set the time controls listed at the
top of the corresponding log file (and `adjustTimeStep no;` so runs are
step-for-step comparable), then run `foamRun` once with the CPU settings and
once with `cudaPCG`.

## Roadmap

- MPI / decomposed-case support (coupled interface exchange on the GPU)
- GPU algebraic multigrid — GAMG converges in ~40 iterations where PCG needs
  thousands; a GPU AMG would stack that algorithmic advantage on top of the
  hardware advantage (estimated further 3–5× on the pressure solve)
- Multi-GPU via the MPI path

## License

GPL-3.0 — same as OpenFOAM, whose headers and libraries this plugin builds
against. See [LICENSE](LICENSE).

OPENFOAM® is a registered trademark of OpenCFD Ltd. This project is not
affiliated with or endorsed by OpenCFD Ltd or the OpenFOAM Foundation.
