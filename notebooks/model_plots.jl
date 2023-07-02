### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b0febea6-fa47-11ed-18db-513dc85d0e01
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end

# ╔═╡ 47fc2b1b-08c4-44fa-a919-9f5083e06929
using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs, PGFPlotsX, Random, Random123, SciMLSensitivity, DiffEqNoiseProcess

# ╔═╡ 16246076-d2a8-4b7e-96e8-359a6e092993
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
		println(join(ARGS, " "))
	end
end

# ╔═╡ 0a3db747-8d05-44e7-bfdd-c5d2fccb389b
pgfplotsx(size=(500, 250))

# ╔═╡ 204b6855-f793-4d0a-a269-f486b5cf859b
md"""
Used model: $(@bind model_name Arg("model", Select(["sun", "sunode", "fhn", "ou"]), short_name="m")), CLI arg: `--model`, `-m` (required!)
"""

# ╔═╡ af84a5af-604f-4eaa-a5d5-5d5d18c649bc
md"""
Timestep size: $(@bind dt Arg("dt", NumberField(0.0:1.0, 0.05), required=false)), CLI arg: `--dt`
"""

# ╔═╡ 36dc1a29-ee0a-4e69-8a08-8cff818d6688
tspan_data = (0e0, 5e0)

# ╔═╡ 3f39c872-5e04-4e61-9c53-9f3fd2824760
n = 100

# ╔═╡ 41cfa896-7e03-4ea7-9ed7-3c16b3c2ebe1
seed = 40

# ╔═╡ 497c096a-d3b8-4a3a-a28b-121d3f2bcbb1
md"""
Use brownian tree: $(@bind use_tree Arg("use-tree", CheckBox(), required=false))
CLI arg: `--use-tree`
"""

# ╔═╡ 51b63289-0414-4958-9bbc-227d3f5e0696
noise = if use_tree
	function(seed)
		rng_tree = Xoshiro(seed)
		VirtualBrownianTree(0e0, 0e0, tend=tspan_data[2]*1.5e0; tree_depth=2, rng=Threefry4x((rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32))))
	end
else
	(seed) -> nothing
end

# ╔═╡ a7aace61-142e-4b67-a315-c2cf85177158
(initial_condition, model) = if model_name == "sun"
	(
		range(210e0, 350e0, n),
		NeuralSDEExploration.ZeroDEnergyBalanceModel()
	)
elseif model_name == "sunode"
	(
		range(210e0, 350e0, n),
		NeuralSDEExploration.ZeroDEnergyBalanceModelNonStochastic()
	)
elseif model_name == "fhn"
	(
		[[0e0, 0e0] for i in 1:n],
		NeuralSDEExploration.FitzHughNagumoModel()
	)
elseif model_name == "ou"
	(
		[0e0 for i in 1:n],
		NeuralSDEExploration.OrnsteinUhlenbeck(2.0, 1.0, 1.0)
	)
else
	@error "Invalid model name!"
	nothing
end

# ╔═╡ f92d721f-1061-418c-b0cf-6da0045b6ec3
function steps(tspan, dt)
	return Int(ceil((tspan[2] - tspan[1]) / dt))
end

# ╔═╡ f76cecd4-bd35-4111-8822-be02ea4b0c78
solution_full_1 = NeuralSDEExploration.series(model, initial_condition, tspan_data, steps(tspan_data, dt), noise=noise)

# ╔═╡ 4ea56ef7-fe58-4bda-b882-3881303fddc8
solution_full_2 = NeuralSDEExploration.series(model, initial_condition, tspan_data, steps(tspan_data, dt), seed=0)

# ╔═╡ 7d424318-658b-4f63-aaa1-7c552c83a93f
solution_1 = [(solution_full_1.t, map(first, u)) for u in solution_full_1.u]

# ╔═╡ 23e48eeb-8bae-4b58-b3a9-5571f9479e07
solution_2 = [(solution_full_2.t, map(first, u)) for u in solution_full_2.u]

# ╔═╡ 74c9ae6e-86c0-49b6-bf11-4c5f80283aa3
exp = plot(select_ts(1:50, select_tspan((tspan_data[2] / 2, tspan_data[2]), solution_full_1)), color=:black, axis=([], false), ticks=false, grid=false, background=RGBA{Float64}(1.0,1.0,1.0,0.0))

# ╔═╡ de01645c-ef21-4037-a667-5f7a6a868993
savefig(exp, "~/Downloads/flowchart_data.tikz")

# ╔═╡ 12e33fad-6664-493b-9514-e12257b9197d
plot(solution_1, legend=false)

# ╔═╡ f1613986-b7af-433b-8df9-afd2350f3a2e
plot(solution_2, legend=false)

# ╔═╡ 68b29d7a-5ef1-4272-ae42-5a6052493c1c
# ╠═╡ disabled = true
#=╠═╡
savefig(p, "~/Downloads/p.pdf")
  ╠═╡ =#

# ╔═╡ 49b6f9a8-9934-4fc3-b253-7b72c8a586b2
# ╠═╡ disabled = true
#=╠═╡
savefig(p, "~/Downloads/plot.tikz")
  ╠═╡ =#

# ╔═╡ 8bdd7b6c-c545-47ee-9172-43677a0d6b4b
# ╠═╡ disabled = true
#=╠═╡
PGFPlotsX.DEFAULT_PREAMBLE
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═b0febea6-fa47-11ed-18db-513dc85d0e01
# ╠═16246076-d2a8-4b7e-96e8-359a6e092993
# ╠═47fc2b1b-08c4-44fa-a919-9f5083e06929
# ╠═0a3db747-8d05-44e7-bfdd-c5d2fccb389b
# ╟─204b6855-f793-4d0a-a269-f486b5cf859b
# ╟─af84a5af-604f-4eaa-a5d5-5d5d18c649bc
# ╠═36dc1a29-ee0a-4e69-8a08-8cff818d6688
# ╠═3f39c872-5e04-4e61-9c53-9f3fd2824760
# ╠═41cfa896-7e03-4ea7-9ed7-3c16b3c2ebe1
# ╟─497c096a-d3b8-4a3a-a28b-121d3f2bcbb1
# ╠═51b63289-0414-4958-9bbc-227d3f5e0696
# ╠═a7aace61-142e-4b67-a315-c2cf85177158
# ╠═f92d721f-1061-418c-b0cf-6da0045b6ec3
# ╠═f76cecd4-bd35-4111-8822-be02ea4b0c78
# ╠═4ea56ef7-fe58-4bda-b882-3881303fddc8
# ╠═7d424318-658b-4f63-aaa1-7c552c83a93f
# ╠═23e48eeb-8bae-4b58-b3a9-5571f9479e07
# ╠═74c9ae6e-86c0-49b6-bf11-4c5f80283aa3
# ╠═de01645c-ef21-4037-a667-5f7a6a868993
# ╠═12e33fad-6664-493b-9514-e12257b9197d
# ╠═f1613986-b7af-433b-8df9-afd2350f3a2e
# ╠═68b29d7a-5ef1-4272-ae42-5a6052493c1c
# ╠═49b6f9a8-9934-4fc3-b253-7b72c8a586b2
# ╠═8bdd7b6c-c545-47ee-9172-43677a0d6b4b
