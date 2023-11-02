using Zygote, LinearAlgebra, Random, BenchmarkTools

p = rand(4)
m = 10000

# Arbitrarily chosen diagonal diffusion. Note this isn't a diagonal matrix
# explicitly, but a vector representing a diagonal matrix
function g(u, p, t)
    [p[1] * x - p[2] * t + p[3] * p[4] * t * x * x for x in u]
end

# Current point of the SDE u(t)
u = rand(m)
# Propagated gradient 
v = rand(m)
# Time t
t = rand()
# Sample from white noise
beta = rand(m)

# Compute the VJP without the optimization
dy1, back1 = Zygote.pullback(u, p) do u, p
    g(u, p, t)
end
out = @btime [back1(x) for x in eachcol(Diagonal(v))]

dv = stack(first.(out))
dparams = stack(last.(out))
println(dparams)

# Compute the VJP with the optimization
dy2, back2 = Zygote.pullback(u, p) do u, p
    g(u, p, t) .* beta
end
out2 = @btime back2(v)

resv = first(out2)
resparams = last(out2)

println(isapprox(dv * beta, resv)) # => true
println(isapprox(dparams * beta, resparams)) # => true