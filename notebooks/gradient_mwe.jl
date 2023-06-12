### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 0df239e6-0829-11ee-157e-a9b7ac54d1db
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ 21242c14-afcd-44b8-94ca-f8604efbdee7
begin
	using Lux, DifferentialEquations, Random, Zygote, FiniteDiff, ComponentArrays
	
	rng = Xoshiro()
	
	context_net = Lux.Dense(1, 1)
	dudt_net = Lux.Dense(1, 1)
	
	ps_context, st_context = Lux.setup(rng, context_net)
	ps_dudt, st_dudt = Lux.setup(rng, dudt_net)
	
	p = ComponentArray((context_net=ps_context, dudt_net=ps_dudt))
	
	tspan = (0f0, 1f0)
	tsteps = 10

	function loss(p)
		# context is computed outside of dudt
		context = context_net([1f0], p.context_net, st_context)[1]

		dudt(u, p, t) = dudt_net(context, p.dudt_net, st_dudt)[1]

		solution = solve(ODEProblem(dudt, [0f0], (0f0, 1f0), p))
		only(solution.u[end])
	end

	println(loss(p))
	println(FiniteDiff.finite_difference_gradient(loss, p))
	println(Zygote.gradient(loss, p)[1])	
end

# ╔═╡ Cell order:
# ╠═0df239e6-0829-11ee-157e-a9b7ac54d1db
# ╠═21242c14-afcd-44b8-94ca-f8604efbdee7
