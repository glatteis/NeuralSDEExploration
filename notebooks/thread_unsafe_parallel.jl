### A Pluto.jl notebook ###
# v0.19.24

using Markdown
using InteractiveUtils

# ╔═╡ ba6497fa-ddc2-11ed-0d5a-15b552751796
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end

# ╔═╡ d3575102-5982-4812-8f17-0bdb8f481fe5
using Lux, LuxCore, DifferentialEquations, Random, ComponentArrays, Plots

# ╔═╡ 7693891d-cabf-47a7-8396-58fdab672234
normal = Lux.Dense(2 => 2)

# ╔═╡ 68a56641-22eb-4b09-ad0a-651a1bbc467c
parallel = Lux.Parallel(nothing, [Lux.Chain(Lux.Dense(1 => 4, tanh), Lux.Dense(4 => 1, tanh, init_bias=ones), Lux.Scale(1)) for i in 1:2]...)

# ╔═╡ cf947cd1-e5d9-461f-b3ba-fe2278771024
println("Running on $(Threads.nthreads()) threads")

# ╔═╡ ebaf488e-33fa-4649-86e3-f4eb7fc7cf4d
struct Model{N1,N2} <: LuxCore.AbstractExplicitContainerLayer{(:normal,:parallel,)}
	normal::N1
	parallel::N2
end

# ╔═╡ 02f622b9-ebe6-4383-9de8-d11dce3e32be
rng = Xoshiro()

# ╔═╡ 25a5213f-4fed-4121-92d2-c639f6aabfd6
model = Model{typeof(normal),typeof(parallel)}(normal, parallel)

# ╔═╡ 9ff0b7d8-5693-488f-be8e-6385e01dac47
ps_, st = Lux.setup(rng, model)

# ╔═╡ 3cfff749-0211-441b-8eae-e10bfa31c688
ps = ComponentArray(ps_)

# ╔═╡ 6a0185aa-6f6f-4ed2-b00b-c4791f5ccf05
function drift(batch)
	return function(u::Vector{Float64}, p::ComponentVector, t::Float64)
		one = model.normal(u, p.normal, st.normal)[1]
		two = reduce(vcat, parallel(([[x] for x in one]...,), p.parallel, st.parallel)[1])
		return two
	end
end

# ╔═╡ fcd8ad4e-240c-4c34-b142-8f597759a7ea
diffusion(u, p, t) = fill(10, size(u))

# ╔═╡ 25b42893-5a7d-419b-b523-7678f34e065a
function prob_func(prob, batch, repeat)
    return SDEProblem{false}(drift(batch), diffusion, fill(Float64(batch), 2), (0e0,1e0), ps,seed=batch,noise=nothing)
end

# ╔═╡ 7fb6f5ef-d7ed-4a40-9184-637e1ce225a0
ensemble = EnsembleProblem(nothing, output_func=(sol, i) -> (sol, false), prob_func=prob_func)

# ╔═╡ 243454bd-bf10-42cd-ad03-6f216a2a7e65
solution = solve(ensemble,EulerHeun(),EnsembleThreads();trajectories=100,dt=0.001)

# ╔═╡ 694aad49-1664-4ac5-9404-6b3840dc6847
solution.u[40][:, 100]

# ╔═╡ 1552d4ca-b0e4-45a7-a728-daf863684590
plot(solution)

# ╔═╡ Cell order:
# ╠═ba6497fa-ddc2-11ed-0d5a-15b552751796
# ╠═d3575102-5982-4812-8f17-0bdb8f481fe5
# ╠═7693891d-cabf-47a7-8396-58fdab672234
# ╠═68a56641-22eb-4b09-ad0a-651a1bbc467c
# ╠═cf947cd1-e5d9-461f-b3ba-fe2278771024
# ╠═ebaf488e-33fa-4649-86e3-f4eb7fc7cf4d
# ╠═02f622b9-ebe6-4383-9de8-d11dce3e32be
# ╠═25a5213f-4fed-4121-92d2-c639f6aabfd6
# ╠═9ff0b7d8-5693-488f-be8e-6385e01dac47
# ╠═3cfff749-0211-441b-8eae-e10bfa31c688
# ╠═6a0185aa-6f6f-4ed2-b00b-c4791f5ccf05
# ╠═fcd8ad4e-240c-4c34-b142-8f597759a7ea
# ╠═25b42893-5a7d-419b-b523-7678f34e065a
# ╠═7fb6f5ef-d7ed-4a40-9184-637e1ce225a0
# ╠═243454bd-bf10-42cd-ad03-6f216a2a7e65
# ╠═694aad49-1664-4ac5-9404-6b3840dc6847
# ╠═1552d4ca-b0e4-45a7-a728-daf863684590
