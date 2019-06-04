using Laplacians, SparseArrays


SOLVER = nothing

function solve_SDDM(A, b; reuse_solver=false)
    if !reuse_solver || SOLVER == nothing
        A = sparse(A)
        global SOLVER = approxchol_sddm(A, tol=1e-6)
    end

    x = SOLVER(b)
    return x
end

