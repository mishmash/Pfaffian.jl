# using PyCall
# @pyimport imp
# path = dirname("pfapack/python/pfaffian.py")
# name = basename("pfapack/python/pfaffian.py")
# (file, filename, data) = imp.find_module("pfaffian", [path]);
# pfaffian = imp.load_module(name, file, filename, data)

function Householder(x::Vector{Float64})
    @assert length(x) > 0

    sigma = dot(x[2:end], x[2:end])

    if sigma == 0
        return (zeros(Float64, length(x)), 0, x[1])
    else
        norm_x = sqrt(x[1]^2 + sigma)
        v = copy(x)
        if x[1] <= 0
            v[1] -= norm_x
            alpha = +norm_x
        else
            v[1] += norm_x
            alpha = -norm_x
        end
        normalize!(v)
        return v, 2, alpha
    end
end

Householder(x::Vector{Int}) = Householder(convert(Vector{Float64}, x))

function Householder(x::Vector{Complex128})
    @assert length(x) > 0

    sigma = dot(x[2:end], x[2:end])

    if sigma == 0
        return (zeros(Complex128, length(x)), 0, x[1])
    else
        norm_x = sqrt(abs2(x[1]) + sigma)
        v = copy(x)
        phase = exp(im * atan2(imag(x[1]), real(x[1])))
        v[1] += phase * norm_x
        normalize!(v)
        return v, 2, -phase * norm_x
    end
end

function skew_tridiagonalize(A::Union{Matrix{Float64}, Matrix{Complex128}}; overwrite_A=false, calc_Q=true)
    @assert size(A)[1] == size(A)[2] > 0
    @assert maximum(abs, A + A.') < 1e-14

    T = eltype(A)
    n = size(A)[1]

    # FIXME: not very Julian...
    if !overwrite_A
        A = copy(A)
    end

    if calc_Q
        Q = eye(T, n)
    end

    for i in 1:n-2
        v, tau, alpha = Householder(A[i+1:end, i])
        A[i+1, i] = alpha
        A[i, i+1] = -alpha
        A[i+2:end, i] = 0.
        A[i, i+2:end] = 0.

        # w = tau * A[i+1:end, i+1:end] * conj(v)
        w = tau * @view(A[i+1:end, i+1:end]) * conj(v)
        # A[i+1:end, i+1:end] += v * w.' - w * v.' # most natural way == turtle
        # A[i+1:end, i+1:end] .+= v .* w.' .- w .* v.' # new dot syntax == useless
        # optimize outer product computation with BLAS
        # (see https://www.reddit.com/r/Julia/comments/32qad9/how_to_calculate_an_outer_product_efficiently/)
        BLAS.ger!(T(1.), v, conj(w), @view(A[i+1:end, i+1:end]))
        BLAS.ger!(T(-1.), w, conj(v), @view(A[i+1:end, i+1:end]))

        if calc_Q
            y = tau * @view(Q[:, i+1:end]) * v
            BLAS.ger!(T(-1.), y, v, @view(Q[:, i+1:end]))
        end
    end

    if calc_Q
        return A, Q
    else
        return A
    end
end

function skew_tridiagonalize(A::Matrix{Int}; overwrite_A=false, calc_Q=true)
    return skew_tridiagonalize(convert(Matrix{Float64}, A), overwrite_A=overwrite_A, calc_Q=calc_Q)
end

function Pfaffian_LTL(A::Union{Matrix{Float64}, Matrix{Complex128}}; overwrite_A=false, skewsymtol=1e-6)::Union{Float64, Complex128}
    @assert size(A)[1] == size(A)[2] > 0
    @assert maximum(abs, A + A.') < skewsymtol

    T = eltype(A)
    n = size(A)[1]

    if n%2 == 1
        return T(0.)
    end

    # FIXME: not very Julian...
    if !overwrite_A
        A = copy(A)
    end

    pf_val = T(1.)

    for k in 1:2:n-1
        kp = k + indmax(abs.(A[k+1:end, k]))

        if kp != k+1
            temp = copy(A[k+1, k:end])
            A[k+1, k:end] = @view(A[kp, k:end])
            A[kp, k:end] = temp

            temp = copy(A[k:end, k+1])
            A[k:end, k+1] = @view(A[k:end, kp])
            A[k:end, kp] = temp

            pf_val *= -1.
        end

        if A[k+1, k] != 0.
            tau = copy(A[k, k+2:end])
            tau /= A[k, k+1]

            pf_val *= A[k, k+1]

            if k+2 <= n
                BLAS.ger!(T(1.), tau, conj(@view(A[k+2:end, k+1])), @view(A[k+2:end, k+2:end]))
                BLAS.ger!(T(-1.), @view(A[k+2:end, k+1]), conj(tau), @view(A[k+2:end, k+2:end]))
            end
        else
            return T(0.)
        end
    end

    return pf_val
end

function Pfaffian_LTL(A::Matrix{Int}; overwrite_A=false, skewsymtol=1e-6)::Float64
    return Pfaffian_LTL(convert(Matrix{Float64}, A), overwrite_A=overwrite_A, skewsymtol=skewsymtol)
end

function Pfaffian_Householder(A::Union{Matrix{Float64}, Matrix{Complex128}}; overwrite_A=false, skewsymtol=1e-6)::Union{Float64, Complex128}
    @assert size(A)[1] == size(A)[2] > 0
    @assert maximum(abs, A + A.') < skewsymtol

    T = eltype(A)
    n = size(A)[1]

    if n%2 == 1
        return T(0.)
    end

    # FIXME: not very Julian...
    if !overwrite_A
        A = copy(A)
    end

    pf_val = T(1.)

    for i in 1:n-2
        v, tau, alpha = Householder(A[i+1:end, i])
        A[i+1, i] = alpha
        A[i, i+1] = -alpha
        A[i+2:end, i] = 0.
        A[i, i+2:end] = 0.

        w = tau * @view(A[i+1:end, i+1:end]) * conj(v)
        BLAS.ger!(T(1.), v, conj(w), @view(A[i+1:end, i+1:end]))
        BLAS.ger!(T(-1.), w, conj(v), @view(A[i+1:end, i+1:end]))

        if tau != 0
            pf_val *= 1. - tau
        end
        if i%2 == 1
            pf_val *= -alpha
        end
    end

    pf_val *= A[n-1, n]

    return pf_val
end

function Pfaffian_Householder(A::Matrix{Int}; overwrite_A=false, skewsymtol=1e-6)::Float64
    return Pfaffian_Householder(convert(Matrix{Float64}, A), overwrite_A=overwrite_A, skewsymtol=skewsymtol)
end

# default to LTL for speed
Pfaffian(A::Matrix; overwrite_A=false, skewsymtol=1e-6)::Union{Float64, Complex128} = Pfaffian_LTL(A, overwrite_A=overwrite_A, skewsymtol=skewsymtol)
# Pfaffian(A::Matrix; overwrite_A=false, skewsymtol=1e-6)::Union{Float64, Complex128} = Pfaffian_Householder(A, overwrite_A=overwrite_A, skewsymtol=skewsymtol)

# Wimmer's version
# Pfaffian(A::Matrix; overwrite_A=false, skewsymtol=1e-6)::Union{Float64, Complex128} = pfaffian[:pfaffian](A, overwrite_a=overwrite_A, method="P")
