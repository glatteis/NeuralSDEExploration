### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 68c8b576-e29e-11ed-2ad2-afc5ee52401a
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end


# ╔═╡ 1071dc36-2fb4-4027-8304-cf4b7c6a962e
using NeuralSDEExploration, StateSpaceModels, Plots, PlutoUI, DataFrames, CSV

# ╔═╡ 7bf385b4-69cf-4e42-a8e9-2b12e57846ae
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
	end
end


# ╔═╡ dd48a285-e40e-468a-b5af-5518351128e9
noise = 0.135

# ╔═╡ ab5de288-e73a-48f2-b8c6-478d74700f6a
# ╠═╡ disabled = true
#=╠═╡
model = NeuralSDEExploration.ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, noise)

  ╠═╡ =#

# ╔═╡ 407b38a5-8e2e-4980-9e8c-25bfd3a0aab5
model = NeuralSDEExploration.FitzHughNagumoModel()

# ╔═╡ 2ae8f19f-8a70-43ec-b552-d56c1b65d746
begin
	n = 2
    datasize = 625
    tspan = (0.0, 5.0)
end


# ╔═╡ cbce9923-70fa-4706-ab43-62f2c25ec559
solution = NeuralSDEExploration.series(model, [[0.0, 0.0]], tspan, datasize; seed=10)

# ╔═╡ d8809ab0-978e-435a-9213-a1d0fcc35331
solution[1]

# ╔═╡ ddd3c757-80d1-49f9-8c62-8340f1d494e9
plot(solution[1])

# ╔═╡ 3354ca5f-e9db-454e-b107-b9839817122a
solution[1].u

# ╔═╡ 29818d39-25d1-480c-9c7f-1094af9c4047
airp = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame

# ╔═╡ 58f08340-c39e-433e-ac63-3a3b89d475a7
y = map(first, solution[1].u)[1:400]

# ╔═╡ 6a941237-b0d1-490d-81d3-07aa15b1d723
plot(y)

# ╔═╡ e816dd26-cf7d-4bcc-abe8-af2fd3fcb9cf
log_air_passengers = log.(airp.passengers)

# ╔═╡ 51ecdc84-4603-4994-acd0-2fd6a57861ca
# ╠═╡ disabled = true
#=╠═╡
m = auto_arima(log_air_passengers; seasonal=12)
  ╠═╡ =#

# ╔═╡ 1ee5a51c-c624-4451-8372-f29a8552a19e
# ╠═╡ disabled = true
#=╠═╡
m = SARIMA(log_air_passengers; order = (0, 1, 1), seasonal_order = (0, 1, 1, 12))
  ╠═╡ =#

# ╔═╡ 376c4845-7383-4d0c-8ab6-faecded265c9
m = auto_ets(log_air_passengers)

# ╔═╡ 1f11e515-ebc5-4088-b84e-15d770454d7d
fit!(m)

# ╔═╡ 9fdcf593-afb5-43fa-b928-99d5b42ebe81
f = forecast(m, 200)

# ╔═╡ 2f2ae3a7-2386-4db6-a450-6fdd18fd116e
plot(m, f)

# ╔═╡ Cell order:
# ╠═68c8b576-e29e-11ed-2ad2-afc5ee52401a
# ╠═7bf385b4-69cf-4e42-a8e9-2b12e57846ae
# ╠═1071dc36-2fb4-4027-8304-cf4b7c6a962e
# ╠═dd48a285-e40e-468a-b5af-5518351128e9
# ╠═ab5de288-e73a-48f2-b8c6-478d74700f6a
# ╠═407b38a5-8e2e-4980-9e8c-25bfd3a0aab5
# ╠═2ae8f19f-8a70-43ec-b552-d56c1b65d746
# ╠═cbce9923-70fa-4706-ab43-62f2c25ec559
# ╠═d8809ab0-978e-435a-9213-a1d0fcc35331
# ╠═ddd3c757-80d1-49f9-8c62-8340f1d494e9
# ╠═3354ca5f-e9db-454e-b107-b9839817122a
# ╟─29818d39-25d1-480c-9c7f-1094af9c4047
# ╠═58f08340-c39e-433e-ac63-3a3b89d475a7
# ╠═6a941237-b0d1-490d-81d3-07aa15b1d723
# ╠═e816dd26-cf7d-4bcc-abe8-af2fd3fcb9cf
# ╠═51ecdc84-4603-4994-acd0-2fd6a57861ca
# ╠═1ee5a51c-c624-4451-8372-f29a8552a19e
# ╠═376c4845-7383-4d0c-8ab6-faecded265c9
# ╠═1f11e515-ebc5-4088-b84e-15d770454d7d
# ╠═9fdcf593-afb5-43fa-b928-99d5b42ebe81
# ╠═2f2ae3a7-2386-4db6-a450-6fdd18fd116e
