#include "cudaPCG.H"

extern "C" void cudaPCG_solve(
    int nCells, int nFaces,
    const double* diag, const double* upper, const double* lower,
    const int* upperAddr, const int* lowerAddr,
    const double* source,
    double* psi,
    double tolerance, double relTol,
    int minIter, int maxIter,
    int* nIterationsOut, double* initialResidualOut, double* finalResidualOut
);

namespace Foam
{
    defineTypeNameAndDebug(cudaPCG, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<cudaPCG>
        addcudaPCGSymMatrixConstructorToTable_;
}


Foam::cudaPCG::cudaPCG
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


Foam::solverPerformance Foam::cudaPCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    solverPerformance solverPerf(typeName, fieldName_);

    label nCells = psi.size();
    label nFaces = matrix_.upper().size();

    bool hasCoupled = false;
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            hasCoupled = true;
            break;
        }
    }
    if (hasCoupled)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            WarningInFunction
                << "cudaPCG does not apply coupled (processor/cyclic) "
                   "boundary contributions on the GPU. Use it only for "
                   "single-rank runs without cyclic patches; results "
                   "otherwise will not match the CPU solver." << endl;
        }
    }

    int nIterations = 0;
    double initialResidual = 0.0;
    double finalResidual = 0.0;

    cudaPCG_solve
    (
        nCells, nFaces,
        matrix_.diag().begin(),
        matrix_.upper().begin(),
        matrix_.lower().begin(),
        matrix_.lduAddr().upperAddr().begin(),
        matrix_.lduAddr().lowerAddr().begin(),
        source.begin(),
        psi.begin(),
        tolerance_, relTol_,
        minIter_, maxIter_,
        &nIterations, &initialResidual, &finalResidual
    );

    solverPerf.initialResidual() = initialResidual;
    solverPerf.finalResidual() = finalResidual;
    solverPerf.nIterations() = nIterations;

    return solverPerf;
}
