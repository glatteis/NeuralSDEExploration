using Lux, Zygote, DifferentialEquations, ComponentArrays, Random, SciMLSensitivity, LinearAlgebra

p = [1.5]
m = 2

function f(u, p, t)
    [p[1] * u[1], u[2]]
end

Random.seed!(434988934)
y = rand(m)
λ = rand(m)
t = rand()

dgrad = zeros(length(p),m)
dλ = zeros(m,m)
dy = zeros(m)

dy, back = Zygote.pullback(y, p) do u, p
    f(u, p, t)
end
[back(x) for x in eachcol(Diagonal(λ))]
@time out = [back(x) for x in eachcol(Diagonal(λ))]
    
println(out)

dλ1 = stack(first.(out))
dgrad1 = stack(last.(out))

dW = rand(m)

println("Computed with a vjp for each dimension: $(dλ1 * dW)")
println("Computed with a vjp for each dimension: $(dgrad1 * dW)")

dy2, back = Zygote.pullback(y, p) do u, p
    f(u, p, t) .* dW
end

back(λ)
@time out2 = back(λ)

resλ = first(out2)
resgrad = last(out2)

println("Computed with a single vjp: $resλ")
println("Computed with a single vjp: $resgrad")

println(dλ1 * dW ≈ resλ) # true
println(dgrad1 * dW ≈ resgrad) # true
