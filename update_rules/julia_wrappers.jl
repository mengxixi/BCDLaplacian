using Laplacians, SparseArrays


function solve_SDDM(A, b)
    A = sparse(A)
    solver = approxchol_sddm(A, tol=1e-6)
    x = solver(b)
    return x
end

