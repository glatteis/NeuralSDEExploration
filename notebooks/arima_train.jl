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
using NeuralSDEExploration, StateSpaceModels, Plots, PlutoUI, DataFrames, CSV, StatsBase

# ╔═╡ 7bf385b4-69cf-4e42-a8e9-2b12e57846ae
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
	end
end


# ╔═╡ dd48a285-e40e-468a-b5af-5518351128e9
noise = 0.2

# ╔═╡ ab5de288-e73a-48f2-b8c6-478d74700f6a
# ╠═╡ disabled = true
#=╠═╡
model = NeuralSDEExploration.ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, noise)

  ╠═╡ =#

# ╔═╡ 534b4e0a-ff36-452f-997c-3623f07e877e
model = NeuralSDEExploration.FitzHughNagumoModelGamma()

# ╔═╡ 2ae8f19f-8a70-43ec-b552-d56c1b65d746
begin
	n = 2
    datasize = 10000
    tspan = (0.0, 200.0)
end


# ╔═╡ cbce9923-70fa-4706-ab43-62f2c25ec559
solution = NeuralSDEExploration.series(model, [[0.0, 0.0]], tspan, datasize; seed=11)

# ╔═╡ d8809ab0-978e-435a-9213-a1d0fcc35331
solution[1]

# ╔═╡ ddd3c757-80d1-49f9-8c62-8340f1d494e9
plot(solution[1])

# ╔═╡ 3354ca5f-e9db-454e-b107-b9839817122a
solution[1].u

# ╔═╡ 29818d39-25d1-480c-9c7f-1094af9c4047
airp = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame

# ╔═╡ 58f08340-c39e-433e-ac63-3a3b89d475a7
y = map(first, solution[1].u)

# ╔═╡ 6a941237-b0d1-490d-81d3-07aa15b1d723
plot(y)

# ╔═╡ e816dd26-cf7d-4bcc-abe8-af2fd3fcb9cf
log_air_passengers = log.(airp.passengers)

# ╔═╡ 376c4845-7383-4d0c-8ab6-faecded265c9
m = auto_arima(y; max_p=10, max_q=10)

# ╔═╡ ce690e6c-96e5-4fb3-8426-a52bcec08ceb
scenario_length = 3000

# ╔═╡ 1f11e515-ebc5-4088-b84e-15d770454d7d
StateSpaceModels.fit!(m)

# ╔═╡ 9fdcf593-afb5-43fa-b928-99d5b42ebe81
f = forecast(m, scenario_length)

# ╔═╡ 2f2ae3a7-2386-4db6-a450-6fdd18fd116e
plot(m, f)

# ╔═╡ 95e6c78e-0a0c-43ae-9775-a225b053a426
scenarios = simulate_scenarios(m, 100, 100)

# ╔═╡ cffd7ab6-8eaa-4c4a-a952-d4f51bb10b19
md"""
Histogram span: $(@bind hspan RangeSlider(1:min(datasize, scenario_length)))
"""

# ╔═╡ f7c5e60a-8aaa-48f3-a902-4aaae07b19fc
begin
	#ts = y[hspan]
	histogram_data = fit(Histogram, y, -5.0:0.15:5.0)
end

# ╔═╡ 336bcb98-5f03-4133-a3d0-eba7336febe3
begin
	histogram_arima = fit(Histogram, vcat(scenarios[hspan, :, :]...), -5.0:0.15:5.0)
end

# ╔═╡ 959e21b2-d51d-4ba8-bd26-40cb4aab7f85
plot(scenarios[:, 1, :])

# ╔═╡ 534b6091-9e34-454c-aec8-e5bb354c1404
begin
	p_hist = plot([
		histogram_data,
		histogram_arima,
	], alpha=0.5, labels=["data" "prior"], title="marginal probabilities")
end

# ╔═╡ Cell order:
# ╠═68c8b576-e29e-11ed-2ad2-afc5ee52401a
# ╠═7bf385b4-69cf-4e42-a8e9-2b12e57846ae
# ╠═1071dc36-2fb4-4027-8304-cf4b7c6a962e
# ╠═dd48a285-e40e-468a-b5af-5518351128e9
# ╠═ab5de288-e73a-48f2-b8c6-478d74700f6a
# ╠═534b4e0a-ff36-452f-997c-3623f07e877e
# ╠═2ae8f19f-8a70-43ec-b552-d56c1b65d746
# ╠═cbce9923-70fa-4706-ab43-62f2c25ec559
# ╠═d8809ab0-978e-435a-9213-a1d0fcc35331
# ╠═ddd3c757-80d1-49f9-8c62-8340f1d494e9
# ╠═3354ca5f-e9db-454e-b107-b9839817122a
# ╟─29818d39-25d1-480c-9c7f-1094af9c4047
# ╠═58f08340-c39e-433e-ac63-3a3b89d475a7
# ╠═6a941237-b0d1-490d-81d3-07aa15b1d723
# ╠═e816dd26-cf7d-4bcc-abe8-af2fd3fcb9cf
# ╠═376c4845-7383-4d0c-8ab6-faecded265c9
# ╠═ce690e6c-96e5-4fb3-8426-a52bcec08ceb
# ╠═1f11e515-ebc5-4088-b84e-15d770454d7d
# ╠═9fdcf593-afb5-43fa-b928-99d5b42ebe81
# ╠═2f2ae3a7-2386-4db6-a450-6fdd18fd116e
# ╠═95e6c78e-0a0c-43ae-9775-a225b053a426
# ╟─cffd7ab6-8eaa-4c4a-a952-d4f51bb10b19
# ╠═f7c5e60a-8aaa-48f3-a902-4aaae07b19fc
# ╠═336bcb98-5f03-4133-a3d0-eba7336febe3
# ╠═959e21b2-d51d-4ba8-bd26-40cb4aab7f85
# ╠═534b6091-9e34-454c-aec8-e5bb354c1404
