using Laplacians, SparseArrays, LinearAlgebra, Statistics


SOLVER = nothing
tol = 1e-6

function solve_SDDM(A, b; reuse_solver=false)
    if !reuse_solver || SOLVER == nothing
        A = sparse(A)
        global SOLVER = approxchol_sddm(A, tol=tol)
    end

    x = SOLVER(b)
    return x
end


function reset_solver()
    global SOLVER = nothing
end


function solve_pinv(A, b)
    return pinv(A)*b
end


function solve(A, b)
    return A\b
end