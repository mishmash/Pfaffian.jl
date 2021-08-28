# test against Wimmer's PFAPACK Python implementation: https://michaelwimmer.org/downloads.html
# (PFAPACK must be installed in the same directory as Pfaffian.jl and this script)
# run the present script in a REPL opened in the same folder as pfapack folder downloaded above
# note that the Julia Python installation requires scipy to be installed. Do this using Conda.jl before
# running the test.
include("Pfaffian.jl")

using PyCall
pf = pyimport("pfapack.pfaffian")


N = 100

println()
@show N
println()
println("Testing real methods:")
println()

x = rand(Float64, N)

v, tau, alpha = Householder(x)
v_w, tau_w, alpha_w = pf.householder_real(x)

@show maximum(abs, v - v_w)
@show abs(tau - tau_w)
@show abs(alpha - alpha_w)
println()

A = rand(Float64, (N, N))
A = A - A'

T, Q = skew_tridiagonalize(A)
T_w, Q_w = pf.skew_tridiagonalize(A)

@show maximum(abs, T - T_w)
@show maximum(abs, Q - Q_w)
println()

pf_ltl = Pfaffian_LTL(A)
pf_ltl_w = pf.pfaffian(A, method="P")

@show abs((pf_ltl - pf_ltl_w)/pf_ltl_w)
println()

pf_h = Pfaffian_Householder(A)
pf_h_w = pf.pfaffian(A, method="H")

@show abs((pf_h - pf_h_w)/pf_h_w)
println()

println()
println("Testing integer methods:")
println()

x = rand(-3:3, N)

v, tau, alpha = Householder(x)
v_w, tau_w, alpha_w = pf.householder_real(convert(Vector{Float64}, x))

@show maximum(abs, v - v_w)
@show abs(tau - tau_w)
@show abs(alpha - alpha_w)
println()



A = rand(-10:10, (N, N))
A = A - A'

T, Q = skew_tridiagonalize(A)
T_w, Q_w = pf.skew_tridiagonalize(convert(Matrix{Float64}, A))

@show maximum(abs, T - T_w)
@show maximum(abs, Q - Q_w)
println()

pf_ltl = Pfaffian_LTL(A)
pf_ltl_w = pf.pfaffian(A, method="P")

@show abs((pf_ltl - pf_ltl_w)/pf_ltl_w)
println()

pf_h = Pfaffian_Householder(A)
pf_h_w = pf.pfaffian(A, method="H")

@show abs((pf_h - pf_h_w)/pf_h_w)
println()

println()
println("Testing complex methods:")
println()

x = rand(ComplexF64, N)

v, tau, alpha = Householder(x)
v_w, tau_w, alpha_w = pf.householder_complex(x)

@show maximum(abs, v - v_w)
@show abs(tau - tau_w)
@show abs(alpha - alpha_w)
println()

A = rand(ComplexF64, (N, N))
A = A - A'

T, Q = skew_tridiagonalize(A)
T_w, Q_w = pf.skew_tridiagonalize(A)

@show maximum(abs, T - T_w)
@show maximum(abs, Q - Q_w)
println()

pf_ltl = Pfaffian_LTL(A)
pf_ltl_w = pf.pfaffian(A, method="P")

@show abs((pf_ltl - pf_ltl_w)/pf_ltl_w)
println()

pf_h = Pfaffian_Householder(A)
pf_h_w = pf.pfaffian(A, method="H")

@show abs((pf_h - pf_h_w)/pf_h_w)
println()
