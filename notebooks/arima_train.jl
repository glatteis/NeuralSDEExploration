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
ebm = NeuralSDEExploration.ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, 0.14)

# ╔═╡ 2ae8f19f-8a70-43ec-b552-d56c1b65d746
begin
	n = 2
    datasize = 100000
    tspan = (0.0, 100.0)
end


# ╔═╡ cbce9923-70fa-4706-ab43-62f2c25ec559
solution = NeuralSDEExploration.series(ebm, range(210.0e0, 320.0e0, n), tspan, datasize; seed=10)

# ╔═╡ 4ec721fd-e48f-4597-8b54-ba344392971e
arima = auto_arima(solution[1].u)

# ╔═╡ 7b289fae-0e7b-4193-b465-3d7ea8ba584c
@bind steps_ahead Slider(1:100; show_value=true)

# ╔═╡ 2dbac6e2-c574-4456-9dbd-3e9db1ad9a88
f = forecast(arima, steps_ahead)

# ╔═╡ 7a0e66eb-438b-4fc8-b86d-b7ca9c6ed449
plot(arima, f)

# ╔═╡ Cell order:
# ╠═68c8b576-e29e-11ed-2ad2-afc5ee52401a
# ╠═7bf385b4-69cf-4e42-a8e9-2b12e57846ae
# ╠═1071dc36-2fb4-4027-8304-cf4b7c6a962e
# ╠═ab5de288-e73a-48f2-b8c6-478d74700f6a
# ╠═2ae8f19f-8a70-43ec-b552-d56c1b65d746
# ╠═cbce9923-70fa-4706-ab43-62f2c25ec559
# ╠═4ec721fd-e48f-4597-8b54-ba344392971e
# ╠═7b289fae-0e7b-4193-b465-3d7ea8ba584c
# ╠═2dbac6e2-c574-4456-9dbd-3e9db1ad9a88
# ╠═7a0e66eb-438b-4fc8-b86d-b7ca9c6ed449
