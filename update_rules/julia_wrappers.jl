using Laplacians, SparseArrays, LinearAlgebra, Statistics, Logging


SOLVER = nothing
tol = 1e-16

function solve_SDDM(A, b; reuse_solver=false)
    if reuse_solver && SOLVER != nothing
        x = SOLVER(b)
    else
        B = sparse(A)
        global SOLVER = approxchol_sddm(B, tol=tol)
        x = SOLVER(b)
    end
    return x
end


function reset_solver()
    global SOLVER = nothing
end


function solve(A, b)
    # Hessian should be symmetric
    B = Symmetric(A)

    # Start with cholesky since Hessian is always symmetric
    # and we hope that each block is positive definite
    try
        C = cholesky(B)
        return C\b
    catch e
        println("Cholesky failed")
        @info "Exception " e
    end

    # If not PD try a generic solver
    try 
        return A\b
    catch e
        println("Direct solve failed")
        @info "Exception: " e
    end

    # Last try if singular
    return pinv(A)*b
end