using Laplacians, SparseArrays, LinearAlgebra, Statistics, Logging


SOLVER = nothing
tol = 1e-16

# Warm-up code to get the packages loaded/functions compiled
# so time measurements are more accurate
function warmup()
    n = 1000
    A = subsampleEdges(complete_graph(n), 0.05)
    L = lap(A)
    dval = zeros(n); dval[1] = dval[n] = 1e-3;
    SDDM = L + spdiagm(0=>dval)
    b = randn(n); b .-= mean(b)

    solver = approxchol_sddm(SDDM, tol=tol)
    x = solver(b)
    println("warmup---Relative norm: ", norm(SDDM * x - b) / norm(b))

    B = Symmetric(SDDM)
    Chol = cholesky(B)
    x = Chol\b 
    println("warmup---Relative norm: ", norm(SDDM * x - b) / norm(b))
end

warmup()


function solve_SDDM(A, b; reuse_solver=false)
    if !reuse_solver || SOLVER == nothing
        B = sparse(A)
        global SOLVER = approxchol_sddm(B, tol=tol)
    end
    x = SOLVER(b)

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