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

# ╔═╡ b0febea6-fa47-11ed-18db-513dc85d0f01
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end

# ╔═╡ 47fc2b1b-08c4-44fa-a919-9f5083e06929
using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs

# ╔═╡ 16246076-d2a8-4b7e-96e8-359a6f092993
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
		println(join(ARGS, " "))
	end
end

# ╔═╡ 0a3db747-8d05-44e7-bfdd-c5d2fccb389b
gr(size=(600, 200))

# ╔═╡ 204b6855-f793-4d0a-a269-f486b5cf859b
md"""
Used model: $(@bind model_name Arg("model", Select(["sun", "fhn", "ou"]), short_name="m")), CLI arg: `--model`, `-m` (required!)
"""

# ╔═╡ af84a5af-604f-4eaa-a5d5-5d5d18c649bc
md"""
Timestep size: $(@bind dt Arg("dt", NumberField(0.0:1.0, 0.05), required=false)), CLI arg: `--dt`
"""

# ╔═╡ 36dc1a29-ef0a-4e69-8a08-8cff818d6688
tspan_data = (0f0, 50f0)

# ╔═╡ 3f39c872-5f04-4e61-9c53-9f3fd2824760
n = 10

# ╔═╡ a7aace61-142e-4b67-a315-c2cf85177158
(initial_condition, model) = if model_name == "sun"
	(
		range(210f0, 350f0, n),
		NeuralSDEExploration.ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, 0.135)
	)
elseif model_name == "fhn"
	(
		[[0f0, 0f0] for i in 1:n],
		NeuralSDEExploration.FitzHughNagumoModel()
	)
elseif model_name == "ou"
	(
		[0f0 for i in 1:n],
		NeuralSDEExploration.OrnsteinUhlenbeck()
	)
else
	@error "Invalid model name!"
	nothing
end

# ╔═╡ f92d721f-1061-418c-b0cf-6da0045b6ec3
function steps(tspan, dt)
	return Int(ceil((tspan[2] - tspan[1]) / dt))
end

# ╔═╡ f76cecd4-bd35-4111-8822-bf02ea4b0c78
solution_full = NeuralSDEExploration.series(model, initial_condition, tspan_data, steps(tspan_data, dt), seed=40)

# ╔═╡ c8b69529-4192-40c6-882e-c455a65512b2
start_point = 10

# ╔═╡ 7d424318-658b-4f63-aaa1-7c552c83a93f
solution = [(x.t[start_point:end], map(first, x.u[start_point:end])) for x in solution_full]

# ╔═╡ 12e33fad-6664-493b-9514-e12257b9197d
p = plot(solution, legend=false)

# ╔═╡ 68b29d7a-5ef1-4272-ae42-5a6052493c1c
savefig(p, "~/Downloads/p.pdf")

# ╔═╡ Cell order:
# ╠═b0febea6-fa47-11ed-18db-513dc85d0f01
# ╠═16246076-d2a8-4b7e-96e8-359a6f092993
# ╠═47fc2b1b-08c4-44fa-a919-9f5083e06929
# ╠═0a3db747-8d05-44e7-bfdd-c5d2fccb389b
# ╟─204b6855-f793-4d0a-a269-f486b5cf859b
# ╟─af84a5af-604f-4eaa-a5d5-5d5d18c649bc
# ╠═36dc1a29-ef0a-4e69-8a08-8cff818d6688
# ╠═3f39c872-5f04-4e61-9c53-9f3fd2824760
# ╠═a7aace61-142e-4b67-a315-c2cf85177158
# ╠═f92d721f-1061-418c-b0cf-6da0045b6ec3
# ╠═f76cecd4-bd35-4111-8822-bf02ea4b0c78
# ╠═c8b69529-4192-40c6-882e-c455a65512b2
# ╠═7d424318-658b-4f63-aaa1-7c552c83a93f
# ╠═12e33fad-6664-493b-9514-e12257b9197d
# ╠═68b29d7a-5ef1-4272-ae42-5a6052493c1c
