using Random
using Lux
using Enzyme

rng = Random.default_rng()
Random.seed!(rng, 42)

f(x) = x .* x ./ 2
x = randn(rng, Float32, 5)
v = ones(Float32, 5)

