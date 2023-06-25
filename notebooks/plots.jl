### A Pluto.jl notebook ###
# v0.19.22

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

# ╔═╡ c5916de0-c180-11ed-27a8-038cdb9d5940
begin
	import Pkg
	Pkg.activate("..")
	using Revise
end

# ╔═╡ 734441ce-fdae-49c9-8ec2-20a54630abfc
using NeuralSDEExploration, Plots, PlutoUI

# ╔═╡ 18bf2a6e-7f27-44fe-a834-9ce697728c59
ebm1 = NeuralSDEExploration.
ZeroDEnergyBalanceModel(
0.425,
0.4,
1363,
3.4019999999999996e-8,
0.074)

# ╔═╡ 36c67008-f19d-4bd8-b2b4-f4b0d141ed6b
ebm2 = NeuralSDEExploration.ZeroDEnergyBalanceModelNonStochastic()

# ╔═╡ 5e1d1bec-1252-4ceb-827b-b32ff1b7166a
begin
	n = 50
    datasize = 300
    tspan = (0.0e0, 20e0)
end

# ╔═╡ 9e4eb730-a0df-41c6-bfaa-3702e7ed7e13
solution1 = NeuralSDEExploration.series(ebm1, range(210.0e0, 320.0e0, n), tspan, datasize)

# ╔═╡ ab6e853c-9c2f-4d2a-a38d-ac041911d3e4
solution2 = NeuralSDEExploration.series(ebm2, range(210.0e0, 320.0e0, n), (0.0e0, 5e0), datasize)

# ╔═╡ 6bda1827-da30-40f8-a2be-fb7180cd4e17
begin
	p = plot([(x.t, x.u) for x in solution2], legend=nothing, xlabel="time", ylabel="temperature (K)")
	savefig(p, "nonstoch.pdf")
end

# ╔═╡ 751df299-1443-4752-8a78-8f65d4e61e8a
begin
	p2 = plot([(x.t, x.u) for x in solution1], legend=nothing, xlabel="time", ylabel="temperature (K)")
	savefig(p2, "stoch.pdf")
end

# ╔═╡ 320fefc9-5726-478a-8087-a0e334b3205b
@bind i Slider(1:n)

# ╔═╡ 5b022cfd-7c18-4b93-b540-0470c18b1c30
plot((solution1[i].t, solution1[i].u), legend=nothing, xlabel="time", ylabel="temperature (K)")

# ╔═╡ 336ba04d-f129-4f2e-a138-87ae2429b604
begin
	p3 = plot((solution1[18].t, solution1[18].u), legend=nothing, xlabel="time", ylabel="temperature (K)")
	savefig(p3, "stochsingle.pdf")
end

# ╔═╡ Cell order:
# ╠═c5916de0-c180-11ed-27a8-038cdb9d5940
# ╠═734441ce-fdae-49c9-8ec2-20a54630abfc
# ╠═18bf2a6e-7f27-44fe-a834-9ce697728c59
# ╠═36c67008-f19d-4bd8-b2b4-f4b0d141ed6b
# ╠═5e1d1bec-1252-4ceb-827b-b32ff1b7166a
# ╠═9e4eb730-a0df-41c6-bfaa-3702e7ed7e13
# ╠═ab6e853c-9c2f-4d2a-a38d-ac041911d3e4
# ╠═6bda1827-da30-40f8-a2be-fb7180cd4e17
# ╠═751df299-1443-4752-8a78-8f65d4e61e8a
# ╠═320fefc9-5726-478a-8087-a0e334b3205b
# ╠═5b022cfd-7c18-4b93-b540-0470c18b1c30
# ╠═336ba04d-f129-4f2e-a138-87ae2429b604
