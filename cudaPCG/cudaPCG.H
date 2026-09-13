// GPU-accelerated drop-in replacement for OpenFOAM's PCG solver.
// The entire solve runs on the NVIDIA GPU: cuSPARSE SpMV, a multicolor
// symmetric Gauss-Seidel preconditioner (FP32 sweeps by default), and
// on-device reductions/scalars with a single 16-byte host sync per
// iteration. See cudaKernels.cu for the CUDA implementation.

#ifndef cudaPCG_H
#define cudaPCG_H

#include "lduMatrix.H"

namespace Foam
{

class cudaPCG
:
    public lduMatrix::solver
{

public:

    TypeName("cudaPCG");

    cudaPCG
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    virtual ~cudaPCG()
    {}

    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const;
};

}

#endif
