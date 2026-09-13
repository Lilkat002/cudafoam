// Standalone CUDA translation unit: no OpenFOAM headers, raw pointers only.
// Implements a preconditioned Conjugate Gradient solve for a symmetric matrix
// handed over in OpenFOAM's LDU (lower/diagonal/upper) sparse format,
// mirroring the algorithm in src/OpenFOAM/matrices/lduMatrix/solvers/PCG/PCG.C
// and the normFactor() in lduMatrixSolver.C.
//
// v4 design (throughput-oriented):
//  - LDU is converted once to CSR (structure cached across calls; only values
//    are re-gathered each solve) and SpMV runs through cuSPARSE.
//  - Preconditioner: multicolor symmetric Gauss-Seidel,
//    M = (D+L) D^-1 (D+U) under a greedy graph coloring (typically 2-6
//    colors on FV meshes) — one fully-parallel kernel per color per sweep,
//    no analysis phase. Sweeps run in FP32 by default (the preconditioner
//    only needs to be approximate; CG itself stays FP64) which halves the
//    matrix traffic of the dominant kernels. Set CUDAPCG_FP64_PRECOND=1 to
//    force FP64 sweeps.
//  - All PCG scalars (alpha, beta, dot products, residual) live on the
//    device. Dot products are custom block-reduce+atomicAdd kernels feeding
//    device scalars; alpha/beta are computed by 1-thread kernels; the
//    psi/r update is fused with the |r| reduction. The host performs exactly
//    ONE synchronisation per iteration: a 16-byte read of
//    {residualSum, breakdownFlag} for the convergence test. Divide-by-zero
//    breakdown sets alpha=0 on device (update becomes a no-op) and raises
//    the flag, matching OpenFOAM's pre-update singularity exit.
//
// Cyclic (periodic) boundary coupling is supported: the caller passes the
// couplings as extra COO entries (row, col, value) which are folded into the
// CSR structure — the coloring is built from the CSR, so the preconditioner
// respects periodicity automatically. Processor (MPI) interfaces are not
// handled here; the caller falls back to the CPU solver for those.

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                             \
            exit(1);                                                       \
        }                                                                   \
    } while (0)

#define CUSPARSE_CHECK(call)                                                \
    do {                                                                    \
        cusparseStatus_t st_ = (call);                                     \
        if (st_ != CUSPARSE_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuSPARSE error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cusparseGetErrorString(st_));      \
            exit(1);                                                       \
        }                                                                   \
    } while (0)

namespace {

constexpr int BS = 256;
inline int nb(int n) { return (n + BS - 1) / BS; }

// device scalar slots
enum
{
    S_WARA = 0,     // w.r (current)
    S_WARAOLD,      // w.r (previous iteration)
    S_WAPA,         // w.p
    S_ALPHA,
    S_BETA,
    S_RESID,        // sum|r| accumulator (also reused for setup reductions)
    S_FLAG,         // breakdown flag (0/1, stored as double)
    S_COUNT
};

__global__ void k_gather(
    int nnz, const double* src, const int* perm, double* val, float* valf)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < nnz)
    {
        double v = src[perm[k]];
        val[k] = v;
        if (valf) valf[k] = static_cast<float>(v);
    }
}

__global__ void k_fill(int n, double v, double* x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ void k_sub(int n, const double* a, const double* b, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

// p = w + beta*p, beta read from device scalars
__global__ void k_combineP(
    int n, const double* w, const double* scal, double* p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = w[i] + scal[S_BETA] * p[i];
}

// tmpField terms of lduMatrixSolver::normFactor():
// |A*psi - sumA*avg(psi)| + |source - sumA*avg(psi)|
__global__ void k_normTerm(
    int n, const double* Apsi, const double* source, const double* rowSum,
    double avgPsi, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        double t = rowSum[i] * avgPsi;
        out[i] = fabs(Apsi[i] - t) + fabs(source[i] - t);
    }
}

__device__ void blockReduceAdd(double v, double* out)
{
    __shared__ double sh[BS];
    int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1)
    {
        if (t < s) sh[t] += sh[t + s];
        __syncthreads();
    }
    if (t == 0) atomicAdd(out, sh[0]);
}

// out += a.b  (out must be zeroed beforehand)
__global__ void k_dot(int n, const double* a, const double* b, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    blockReduceAdd(i < n ? a[i] * b[i] : 0.0, out);
}

// out += sum(a)
__global__ void k_sum(int n, const double* a, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    blockReduceAdd(i < n ? a[i] : 0.0, out);
}

// out += sum|a|
__global__ void k_absSum(int n, const double* a, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    blockReduceAdd(i < n ? fabs(a[i]) : 0.0, out);
}

// stash previous w.r and clear the accumulator for this iteration's dot
__global__ void k_prep(double* scal)
{
    scal[S_WARAOLD] = scal[S_WARA];
    scal[S_WARA] = 0.0;
}

// beta from fresh w.r; clear accumulators for the rest of the iteration
__global__ void k_beta(int firstIter, double* scal)
{
    scal[S_BETA] = firstIter ? 0.0 : scal[S_WARA] / scal[S_WARAOLD];
    scal[S_WAPA] = 0.0;
    scal[S_RESID] = 0.0;
    scal[S_FLAG] = 0.0;
}

// alpha with OpenFOAM's singularity guard: on breakdown alpha=0 (the
// subsequent update is then a no-op) and the flag tells the host to exit.
__global__ void k_alpha(double normFactor, double* scal)
{
    if (fabs(scal[S_WAPA]) / normFactor < 1e-20)
    {
        scal[S_ALPHA] = 0.0;
        scal[S_FLAG] = 1.0;
    }
    else
    {
        scal[S_ALPHA] = scal[S_WARA] / scal[S_WAPA];
    }
}

// fused: psi += alpha*p, r -= alpha*w, and reduce sum|r| into resid
__global__ void k_updateFused(
    int n, const double* p, const double* w, const double* scal,
    double* psi, double* r, double* resid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double a = scal[S_ALPHA];
    double ri = 0.0;
    if (i < n)
    {
        psi[i] += a * p[i];
        ri = r[i] - a * w[i];
        r[i] = ri;
    }
    blockReduceAdd(fabs(ri), resid);
}

// One color of a Gauss-Seidel triangular sweep under the coloring
// permutation. forward: y_i = (rhs_i - sum_{color(j)<color(i)} a_ij y_j)/d_i
// (colors processed ascending, so those y_j are final); backward: mirror
// with color(j)>color(i), colors processed descending. T=float runs the
// sweep in reduced precision; the surrounding CG stays FP64.
template<class T, class TR>
__global__ void k_gsSweep(
    int nColorCells, const int* cells,
    const int* rowPtr, const int* colInd, const T* val,
    const int* color, int myColor, int forward,
    const TR* rhs, T* y)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nColorCells) return;
    int i = cells[t];
    T s = static_cast<T>(rhs[i]);
    T d = static_cast<T>(1);
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k)
    {
        int j = colInd[k];
        if (j == i)
        {
            d = val[k];
        }
        else if (forward ? (color[j] < myColor) : (color[j] > myColor))
        {
            s -= val[k] * y[j];
        }
    }
    y[i] = s / d;
}

// tmp = diag .* y   (diag stays double; y/tmp in sweep precision T)
template<class T>
__global__ void k_diagMul(int n, const double* diag, const T* y, T* tmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tmp[i] = static_cast<T>(diag[i]) * y[i];
}

__global__ void k_f2d(int n, const float* a, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = static_cast<double>(a[i]);
}

struct Ctx
{
    int nCells = -1, nFaces = -1, nExtra = 0, nnz = 0;
    bool fp32Precond = true;

    // CSR structure (constant while addressing is constant)
    int *d_rowPtr = nullptr, *d_colInd = nullptr, *d_perm = nullptr;
    double* d_valA = nullptr;
    float* d_valAf = nullptr;   // FP32 copy for preconditioner sweeps

    // multicolor ordering
    int nColors = 0;
    std::vector<int> colorPtr;              // host: [nColors+1] into d_cells
    int *d_cells = nullptr, *d_color = nullptr;

    // staging: [diag | upper | lower | cyclic extras]
    double* d_src = nullptr;

    // work vectors
    double *d_psi = nullptr, *d_source = nullptr, *d_r = nullptr,
           *d_p = nullptr, *d_w = nullptr, *d_y = nullptr,
           *d_tmp = nullptr, *d_ones = nullptr;
    float *d_yf = nullptr, *d_tmpf = nullptr, *d_wf = nullptr; // FP32 sweeps

    double* d_scal = nullptr;   // S_COUNT device scalars
    double* h_pin = nullptr;    // pinned host mirror for per-iter readback

    cudaStream_t stream = nullptr;
    cusparseHandle_t sp = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vPsi = nullptr, vP = nullptr, vW = nullptr,
                         vY = nullptr, vOnes = nullptr;
    void* d_bufMV = nullptr;
};

Ctx ctx;

void freeCtx()
{
    if (ctx.nCells < 0) return;
    cudaFree(ctx.d_rowPtr); cudaFree(ctx.d_colInd); cudaFree(ctx.d_perm);
    cudaFree(ctx.d_valA); cudaFree(ctx.d_valAf);
    cudaFree(ctx.d_cells); cudaFree(ctx.d_color);
    cudaFree(ctx.d_src);
    cudaFree(ctx.d_psi); cudaFree(ctx.d_source); cudaFree(ctx.d_r);
    cudaFree(ctx.d_p); cudaFree(ctx.d_w); cudaFree(ctx.d_y);
    cudaFree(ctx.d_tmp); cudaFree(ctx.d_ones);
    cudaFree(ctx.d_yf); cudaFree(ctx.d_tmpf); cudaFree(ctx.d_wf);
    cudaFree(ctx.d_scal);
    cudaFreeHost(ctx.h_pin);
    cudaFree(ctx.d_bufMV);
    if (ctx.matA) cusparseDestroySpMat(ctx.matA);
    if (ctx.vPsi) cusparseDestroyDnVec(ctx.vPsi);
    if (ctx.vP) cusparseDestroyDnVec(ctx.vP);
    if (ctx.vW) cusparseDestroyDnVec(ctx.vW);
    if (ctx.vY) cusparseDestroyDnVec(ctx.vY);
    if (ctx.vOnes) cusparseDestroyDnVec(ctx.vOnes);
    if (ctx.sp) cusparseDestroy(ctx.sp);
    if (ctx.stream) cudaStreamDestroy(ctx.stream);
    ctx = Ctx();
    ctx.nCells = -1;
}

// Build the CSR structure and the greedy graph coloring on the host.
// CSR value k gathers from
// [diag(0..n) | upper(n..n+nF) | lower(n+nF..n+2nF) | extras(n+2nF..)].
// Extras are cyclic (periodic) couplings handed over as COO entries; they
// participate in the coloring like any other off-diagonal, so the GS
// preconditioner respects periodicity.
void setup(int nCells, int nFaces, const int* uAddr, const int* lAddr,
           int nExtra, const int* eRow, const int* eCol)
{
    freeCtx();

    const int n = nCells;
    const int nnz = nCells + 2 * nFaces + nExtra;

    const char* env = getenv("CUDAPCG_FP64_PRECOND");
    ctx.fp32Precond = !(env && env[0] == '1');

    std::vector<int> rowCount(n, 1);
    for (int f = 0; f < nFaces; ++f)
    {
        rowCount[lAddr[f]]++;
        rowCount[uAddr[f]]++;
    }
    for (int e = 0; e < nExtra; ++e) rowCount[eRow[e]]++;
    std::vector<int> rowPtr(n + 1, 0);
    for (int i = 0; i < n; ++i) rowPtr[i + 1] = rowPtr[i] + rowCount[i];

    std::vector<int> colInd(nnz), perm(nnz), fill(n, 0);
    auto put = [&](int row, int col, int src)
    {
        int k = rowPtr[row] + fill[row]++;
        colInd[k] = col;
        perm[k] = src;
    };
    for (int i = 0; i < n; ++i) put(i, i, i);
    for (int f = 0; f < nFaces; ++f)
    {
        put(lAddr[f], uAddr[f], nCells + f);          // upper coeff
        put(uAddr[f], lAddr[f], nCells + nFaces + f); // lower coeff
    }
    for (int e = 0; e < nExtra; ++e)
    {
        put(eRow[e], eCol[e], nCells + 2 * nFaces + e);
    }
    for (int i = 0; i < n; ++i)
    {
        int b = rowPtr[i], e = rowPtr[i + 1];
        std::vector<std::pair<int, int>> ents(e - b);
        for (int k = b; k < e; ++k) ents[k - b] = {colInd[k], perm[k]};
        std::sort(ents.begin(), ents.end());
        for (int k = b; k < e; ++k)
        {
            colInd[k] = ents[k - b].first;
            perm[k] = ents[k - b].second;
        }
    }

    // greedy coloring: smallest color unused by any already-colored neighbour
    std::vector<int> color(n, -1);
    int nColors = 0;
    {
        std::vector<int> used;
        for (int i = 0; i < n; ++i)
        {
            used.assign(nColors + 1, 0);
            for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k)
            {
                int j = colInd[k];
                if (j != i && color[j] >= 0 && color[j] <= nColors)
                {
                    used[color[j]] = 1;
                }
            }
            int c = 0;
            while (c < static_cast<int>(used.size()) && used[c]) ++c;
            color[i] = c;
            nColors = std::max(nColors, c + 1);
        }
    }

    std::vector<int> colorCount(nColors, 0);
    for (int i = 0; i < n; ++i) colorCount[color[i]]++;
    ctx.colorPtr.assign(nColors + 1, 0);
    for (int c = 0; c < nColors; ++c)
    {
        ctx.colorPtr[c + 1] = ctx.colorPtr[c] + colorCount[c];
    }
    std::vector<int> cells(n), cfill(nColors, 0);
    for (int i = 0; i < n; ++i)
    {
        cells[ctx.colorPtr[color[i]] + cfill[color[i]]++] = i;
    }

    ctx.nCells = nCells;
    ctx.nFaces = nFaces;
    ctx.nExtra = nExtra;
    ctx.nnz = nnz;
    ctx.nColors = nColors;

    CUDA_CHECK(cudaStreamCreate(&ctx.stream));

    CUDA_CHECK(cudaMalloc(&ctx.d_rowPtr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_colInd, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_perm, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_valA, nnz * sizeof(double)));
    if (ctx.fp32Precond)
    {
        CUDA_CHECK(cudaMalloc(&ctx.d_valAf, nnz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_yf, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_tmpf, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_wf, n * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&ctx.d_cells, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_color, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_src, nnz * sizeof(double)));
    for (double** v : {&ctx.d_psi, &ctx.d_source, &ctx.d_r, &ctx.d_p,
                       &ctx.d_w, &ctx.d_y, &ctx.d_tmp, &ctx.d_ones})
    {
        CUDA_CHECK(cudaMalloc(v, n * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&ctx.d_scal, S_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx.h_pin, S_COUNT * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(ctx.d_rowPtr, rowPtr.data(), (n + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_colInd, colInd.data(), nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_perm, perm.data(), nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_cells, cells.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_color, color.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));
    k_fill<<<nb(n), BS, 0, ctx.stream>>>(n, 1.0, ctx.d_ones);

    CUSPARSE_CHECK(cusparseCreate(&ctx.sp));
    CUSPARSE_CHECK(cusparseSetStream(ctx.sp, ctx.stream));
    CUSPARSE_CHECK(cusparseCreateCsr(&ctx.matA, n, n, nnz,
        ctx.d_rowPtr, ctx.d_colInd, ctx.d_valA,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx.vPsi, n, ctx.d_psi, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx.vP, n, ctx.d_p, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx.vW, n, ctx.d_w, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx.vY, n, ctx.d_y, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx.vOnes, n, ctx.d_ones, CUDA_R_64F));

    const double one = 1.0, zero = 0.0;
    size_t s1 = 0, s2 = 0, s3 = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.sp,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, ctx.matA, ctx.vPsi, &zero,
        ctx.vW, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &s1));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.sp,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, ctx.matA, ctx.vP, &zero,
        ctx.vW, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &s2));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx.sp,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &one, ctx.matA, ctx.vOnes, &zero,
        ctx.vY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &s3));
    CUDA_CHECK(cudaMalloc(&ctx.d_bufMV, std::max({s1, s2, s3, size_t(8)})));

    printf("cudaPCG: %d cells, %d faces, %d cyclic couplings, %d colors, "
           "%s preconditioner\n",
           nCells, nFaces, nExtra, nColors,
           ctx.fp32Precond ? "FP32" : "FP64");
}

void spmv(cusparseDnVecDescr_t x, cusparseDnVecDescr_t y)
{
    const double one = 1.0, zero = 0.0;
    CUSPARSE_CHECK(cusparseSpMV(ctx.sp, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one, ctx.matA, x, &zero, y, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, ctx.d_bufMV));
}

// w = M^-1 r with M = (D+L)D^-1(D+U) under the coloring permutation:
// forward sweep (colors ascending) solves (D+L) y = r, then tmp = D y,
// backward sweep (colors descending) solves (D+U) w = tmp.
template<class T>
void preconditionT(T* y, T* tmp, T* w, const T* val)
{
    const int n = ctx.nCells;
    cudaStream_t st = ctx.stream;
    for (int c = 0; c < ctx.nColors; ++c)
    {
        int b = ctx.colorPtr[c], nc = ctx.colorPtr[c + 1] - b;
        k_gsSweep<<<nb(nc), BS, 0, st>>>(nc, ctx.d_cells + b,
            ctx.d_rowPtr, ctx.d_colInd, val,
            ctx.d_color, c, 1, ctx.d_r, y);
    }
    k_diagMul<<<nb(n), BS, 0, st>>>(n, ctx.d_src, y, tmp); // D = diag
    for (int c = ctx.nColors - 1; c >= 0; --c)
    {
        int b = ctx.colorPtr[c], nc = ctx.colorPtr[c + 1] - b;
        k_gsSweep<<<nb(nc), BS, 0, st>>>(nc, ctx.d_cells + b,
            ctx.d_rowPtr, ctx.d_colInd, val,
            ctx.d_color, c, 0, tmp, w);
    }
}

void precondition()
{
    if (ctx.fp32Precond)
    {
        preconditionT<float>(ctx.d_yf, ctx.d_tmpf, ctx.d_wf, ctx.d_valAf);
        k_f2d<<<nb(ctx.nCells), BS, 0, ctx.stream>>>(
            ctx.nCells, ctx.d_wf, ctx.d_w);
    }
    else
    {
        preconditionT<double>(ctx.d_y, ctx.d_tmp, ctx.d_w, ctx.d_valA);
    }
}

// synchronising scalar read via the S_RESID slot (setup path, not hot loop)
double readResidSlot()
{
    CUDA_CHECK(cudaMemcpyAsync(ctx.h_pin, ctx.d_scal + S_RESID,
                               sizeof(double), cudaMemcpyDeviceToHost,
                               ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return ctx.h_pin[0];
}

double sumOf(const double* v, int n)
{
    CUDA_CHECK(cudaMemsetAsync(ctx.d_scal + S_RESID, 0, sizeof(double),
                               ctx.stream));
    k_sum<<<nb(n), BS, 0, ctx.stream>>>(n, v, ctx.d_scal + S_RESID);
    return readResidSlot();
}

double sumAbsOf(const double* v, int n)
{
    CUDA_CHECK(cudaMemsetAsync(ctx.d_scal + S_RESID, 0, sizeof(double),
                               ctx.stream));
    k_absSum<<<nb(n), BS, 0, ctx.stream>>>(n, v, ctx.d_scal + S_RESID);
    return readResidSlot();
}

} // namespace

extern "C" void cudaPCG_solve(
    int nCells, int nFaces,
    const double* h_diag, const double* h_upper, const double* h_lower,
    const int* h_upperAddr, const int* h_lowerAddr,
    int nExtra, const int* h_extraRow, const int* h_extraCol,
    const double* h_extraVal,
    const double* h_source,
    double* h_psi,
    double tolerance, double relTol,
    int minIter, int maxIter,
    int* nIterationsOut, double* initialResidualOut, double* finalResidualOut)
{
    if (nCells != ctx.nCells || nFaces != ctx.nFaces
     || nExtra != ctx.nExtra)
    {
        setup(nCells, nFaces, h_upperAddr, h_lowerAddr,
              nExtra, h_extraRow, h_extraCol);
    }

    const int n = nCells;
    cudaStream_t st = ctx.stream;

    // stage [diag | upper | lower | extras], gather into CSR value order
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_src, h_diag, n * sizeof(double),
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_src + n, h_upper,
                               nFaces * sizeof(double),
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_src + n + nFaces, h_lower,
                               nFaces * sizeof(double),
                               cudaMemcpyHostToDevice, st));
    if (nExtra > 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(ctx.d_src + n + 2 * nFaces, h_extraVal,
                                   nExtra * sizeof(double),
                                   cudaMemcpyHostToDevice, st));
    }
    k_gather<<<nb(ctx.nnz), BS, 0, st>>>(ctx.nnz, ctx.d_src, ctx.d_perm,
                                         ctx.d_valA, ctx.d_valAf);

    CUDA_CHECK(cudaMemcpyAsync(ctx.d_psi, h_psi, n * sizeof(double),
                               cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_source, h_source, n * sizeof(double),
                               cudaMemcpyHostToDevice, st));

    // wA = A*psi ; rA = source - wA
    spmv(ctx.vPsi, ctx.vW);
    k_sub<<<nb(n), BS, 0, st>>>(n, ctx.d_source, ctx.d_w, ctx.d_r);

    // normFactor (exact lduMatrixSolver::normFactor formula)
    double avgPsi = sumOf(ctx.d_psi, n) / n;
    spmv(ctx.vOnes, ctx.vY); // rowSum = A*1
    k_normTerm<<<nb(n), BS, 0, st>>>(n, ctx.d_w, ctx.d_source, ctx.d_y,
                                     avgPsi, ctx.d_tmp);
    double normFactor = sumOf(ctx.d_tmp, n) + 1e-20;

    double initialResidual = sumAbsOf(ctx.d_r, n) / normFactor;
    double finalResidual = initialResidual;

    int nIter = 0;

    bool converged =
        initialResidual < tolerance
     || (relTol > 0.0 && initialResidual < relTol * initialResidual);

    if (minIter > 0 || !converged)
    {
        do
        {
            k_prep<<<1, 1, 0, st>>>(ctx.d_scal);

            precondition();
            k_dot<<<nb(n), BS, 0, st>>>(n, ctx.d_w, ctx.d_r,
                                        ctx.d_scal + S_WARA);
            k_beta<<<1, 1, 0, st>>>(nIter == 0 ? 1 : 0, ctx.d_scal);

            if (nIter == 0)
            {
                CUDA_CHECK(cudaMemcpyAsync(ctx.d_p, ctx.d_w,
                                           n * sizeof(double),
                                           cudaMemcpyDeviceToDevice, st));
            }
            else
            {
                k_combineP<<<nb(n), BS, 0, st>>>(n, ctx.d_w, ctx.d_scal,
                                                 ctx.d_p);
            }

            spmv(ctx.vP, ctx.vW);
            k_dot<<<nb(n), BS, 0, st>>>(n, ctx.d_w, ctx.d_p,
                                        ctx.d_scal + S_WAPA);
            k_alpha<<<1, 1, 0, st>>>(normFactor, ctx.d_scal);
            k_updateFused<<<nb(n), BS, 0, st>>>(n, ctx.d_p, ctx.d_w,
                                                ctx.d_scal, ctx.d_psi,
                                                ctx.d_r,
                                                ctx.d_scal + S_RESID);

            // the single host synchronisation of the iteration:
            // {sum|r|, breakdown flag} land in adjacent slots
            CUDA_CHECK(cudaMemcpyAsync(ctx.h_pin, ctx.d_scal + S_RESID,
                                       2 * sizeof(double),
                                       cudaMemcpyDeviceToHost, st));
            CUDA_CHECK(cudaStreamSynchronize(st));
            if (ctx.h_pin[1] != 0.0) break; // breakdown: psi/r untouched

            finalResidual = ctx.h_pin[0] / normFactor;
            nIter++;
        } while (
            (nIter < maxIter
             && !(finalResidual < tolerance
                  || (relTol > 0.0 && finalResidual < relTol * initialResidual)))
         || nIter < minIter
        );
    }

    CUDA_CHECK(cudaMemcpyAsync(h_psi, ctx.d_psi, n * sizeof(double),
                               cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    *nIterationsOut = nIter;
    *initialResidualOut = initialResidual;
    *finalResidualOut = finalResidual;
}
