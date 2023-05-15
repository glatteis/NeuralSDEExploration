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

# ╔═╡ 68c8b576-e29e-11ed-2ad2-afc5ee52401a
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end


# ╔═╡ 1071dc36-2fb4-4027-8304-cf4b7c6a962e
using NeuralSDEExploration, StateSpaceModels, Plots, PlutoUI

# ╔═╡ 7bf385b4-69cf-4e42-a8e9-2b12e57846ae
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
	end
end


# ╔═╡ ab5de288-e73a-48f2-b8c6-478d74700f6a
ebm = NeuralSDEExploration.FitzHughNagumoModel(0.04,
2.23,
0.82,
-6.98,
0.0,
-1.51)

# ╔═╡ 2ae8f19f-8a70-43ec-b552-d56c1b65d746
begin
	n = 10
    datasize = 1000
    tspan = (0.0, 10.0)
end


# ╔═╡ cbce9923-70fa-4706-ab43-62f2c25ec559
solution = NeuralSDEExploration.series(ebm, [[0.0, 0.0]], tspan, datasize; seed=10)

# ╔═╡ d8809ab0-978e-435a-9213-a1d0fcc35331
solution[1]

# ╔═╡ ddd3c757-80d1-49f9-8c62-8340f1d494e9
plot(solution[1])

# ╔═╡ 4ec721fd-e48f-4597-8b54-ba344392971e
arima = auto_arima(solution[1].u)

# ╔═╡ 7b289fae-0e7b-4193-b465-3d7ea8ba584c
@bind steps_ahead Slider(1:100; show_value=true)

# ╔═╡ Cell order:
# ╠═68c8b576-e29e-11ed-2ad2-afc5ee52401a
# ╠═7bf385b4-69cf-4e42-a8e9-2b12e57846ae
# ╠═1071dc36-2fb4-4027-8304-cf4b7c6a962e
# ╠═ab5de288-e73a-48f2-b8c6-478d74700f6a
# ╠═2ae8f19f-8a70-43ec-b552-d56c1b65d746
# ╠═cbce9923-70fa-4706-ab43-62f2c25ec559
# ╠═d8809ab0-978e-435a-9213-a1d0fcc35331
# ╠═ddd3c757-80d1-49f9-8c62-8340f1d494e9
# ╠═4ec721fd-e48f-4597-8b54-ba344392971e
# ╠═7b289fae-0e7b-4193-b465-3d7ea8ba584c
