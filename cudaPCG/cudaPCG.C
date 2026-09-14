#include "cudaPCG.H"
#include "PCG.H"
#include "cyclicLduInterface.H"
#include "processorLduInterface.H"

#include <cstdlib>

extern "C" void cudaPCG_solve(
    int nCells, int nFaces,
    const double* diag, const double* upper, const double* lower,
    const int* upperAddr, const int* lowerAddr,
    int nExtra, const int* extraRow, const int* extraCol,
    const double* extraVal,
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

    // Coupled interfaces. Plain cyclic (periodic) coupling is folded into
    // the GPU matrix as extra off-diagonal entries — on a single rank both
    // sides of a cyclic patch are local cells, so in lduMatrix::Amul terms
    // the interface update result[fc] -= bouCoeffs[f]*psi[nbrFc] is just a
    // matrix entry A(fc, nbrFc) = -bouCoeffs[f]. Anything that is not a
    // one-to-one local coupling (processor boundaries, non-conformal/AMI)
    // falls back to OpenFOAM's own PCG so the answer is always correct.
    List<int> extraRow, extraCol;
    List<double> extraVal;
    bool fallback = false;

    forAll(interfaces_, i)
    {
        if (!interfaces_.set(i)) continue;

        const lduInterface& intf = interfaces_[i].interface();

        if (!isA<cyclicLduInterface>(intf))
        {
            fallback = true;
            break;
        }

        const cyclicLduInterface& cyc =
            refCast<const cyclicLduInterface>(intf);

        const labelUList& fc = matrix_.lduAddr().patchAddr(i);
        const labelUList& nbrFc =
#if defined(OPENFOAM) && OPENFOAM >= 1000
            // ESI fork (e.g. v2412). ESI's cyclicAMILduInterface is a
            // standalone class, not derived from cyclicLduInterface, so
            // AMI/non-conformal couplings fail the isA<> test above and
            // take the CPU fallback — as they should.
            matrix_.lduAddr().patchAddr(cyc.neighbPatchID());
#else
            // Foundation fork (OpenFOAM 12)
            matrix_.lduAddr().patchAddr(cyc.nbrPatchIndex());
#endif
        const scalarField& bou = interfaceBouCoeffs_[i];

        if (nbrFc.size() != fc.size())
        {
            fallback = true;
            break;
        }

        label base = extraRow.size();
        extraRow.setSize(base + fc.size());
        extraCol.setSize(base + fc.size());
        extraVal.setSize(base + fc.size());

        forAll(fc, f)
        {
            if (fc[f] == nbrFc[f])
            {
                // degenerate self-coupling (single-cell-thick periodic
                // direction) — leave it to the CPU solver
                fallback = true;
                break;
            }
            extraRow[base + f] = fc[f];
            extraCol[base + f] = nbrFc[f];
            extraVal[base + f] = -bou[f];
        }
        if (fallback) break;
    }

    static const bool verbose = std::getenv("CUDAPCG_VERBOSE") != nullptr;

    if (fallback)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            WarningInFunction
                << "cudaPCG: coupled interfaces present that cannot be "
                   "applied on the GPU; solving on the CPU with PCG instead. "
                   "The coupled interfaces are:" << endl;
            forAll(interfaces_, i)
            {
                if (!interfaces_.set(i)) continue;
                const lduInterface& intf = interfaces_[i].interface();
                const char* what =
                    isA<processorLduInterface>(intf)
                  ? "processor (MPI decomposition)"
                  : isA<cyclicLduInterface>(intf)
                  ? "cyclic (face-count mismatch or self-coupling)"
                  : "non-conformal/AMI or other coupled type";
                Info<< "    interface " << i << ": " << what << ", "
                    << matrix_.lduAddr().patchAddr(i).size()
                    << " faces" << endl;
            }
        }
        if (verbose)
        {
            Info<< "cudaPCG: [" << fieldName_
                << "] CPU fallback (PCG)" << endl;
        }

        dictionary controls(controlDict_);
        if (!controls.found("preconditioner"))
        {
            controls.add("preconditioner", word("DIC"));
        }

        return PCG
        (
            fieldName_,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            controls
        ).solve(psi, source, cmpt);
    }

    if (verbose)
    {
        Info<< "cudaPCG: [" << fieldName_ << "] GPU path, "
            << extraRow.size() << " cyclic couplings" << endl;
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
        extraRow.size(),
        extraRow.begin(),
        extraCol.begin(),
        extraVal.begin(),
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
