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

# ╔═╡ 7a6bbfd6-ffb7-11ed-39d7-5b673fe4cdae
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end

# ╔═╡ ae7d6e11-55da-44a2-a5b6-60d11caa9dbf
using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs, JLD2, Random, StatsBase, Random123, DiffEqNoiseProcess, Distributions

# ╔═╡ 6995cd16-0c69-49c7-9523-4c842c0db339
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
		println(join(ARGS, " "))
	end
end

# ╔═╡ 43e6a892-8b2d-4fbb-b49c-681d78f0202c
md"""
Model filename: $(@bind f TextField())
"""

# ╔═╡ 09a13088-a784-4d53-8012-cef948eb796c
dict = f === "" ? (nothing, nothing) : load(f)["data"]

# ╔═╡ 9f89a8d9-05e5-4af0-9dd8-1a528ea7e9de
latent_sde = dict["latent_sde"]

# ╔═╡ 7da4b3b6-3cee-4b93-9359-a2f7e2341da9
ps = dict["ps"]

# ╔═╡ 23bc5697-f1ca-4598-92fb-2d43e94ce310
st = dict["st"]

# ╔═╡ b214a6b2-d430-4d95-9f3a-b31c4ff7bcc7
model = dict["model"]

# ╔═╡ fabc1578-ba35-4c4e-9129-02da3bf43f56
timeseries = Timeseries(dict["timeseries"])

# ╔═╡ af3619b0-f9be-40f2-8027-77e435f8e4e5
md"""
Select timeseries to do some simulations on: $(@bind ti RangeSlider(1:100; default=1:4))
"""

# ╔═╡ 69459a5f-75b7-4c33-a489-bf4d4411c1ec
seed = 378

# ╔═╡ e0afda9e-0b17-4e7e-9d1e-d0f05df6fa4e
viz_batch = select_ts(ti, timeseries)

# ╔═╡ 9c5f37b0-9998-4879-85d0-f540089e1ca8
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(viz_batch, ps, st)

# ╔═╡ 8f38bdbc-8b22-46c5-bbf4-d38d277b000f
plot(logterm_[1, :, :]', title="KL-Divergence", linewidth=2)

# ╔═╡ 2f6d198e-4cae-40cd-8624-6aab867fee0b
plot(posterior_data[1, :,:]',linewidth=2,legend=false,title="projected posterior")

# ╔═╡ 63213503-ab28-4158-b522-efd0b0139b6d
function plot_prior(priorsamples; rng=rng, tspan=latent_sde.tspan, datasize=latent_sde.datasize)
	prior = NeuralSDEExploration.sample_prior_dataspace(latent_sde,ps,st;seed=abs(rand(rng, Int)),b=priorsamples, tspan=tspan, datasize=datasize)
	return plot(prior, linewidth=.5,color=:black,legend=false,title="projected prior")
end

# ╔═╡ 0b47115c-0561-439b-be7b-78195da6215e
function plotmodel()
	n = 5
	rng_plot = Xoshiro()
	nums = sample(rng_plot, 1:length(timeseries.u), n; replace=false)
	
	posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(select_ts(nums, timeseries), ps, st, seed=seed)
	
	priorsamples = 25
	priornums = sample(rng_plot, 1:length(timeseries.u), priorsamples; replace=false)
	priorplot = plot_prior(priorsamples, rng=rng_plot)

	posteriorplot = plot(timeseries.t, posterior_data[1, :,:]', linewidth=2, legend=false, title="projected posterior")
	dataplot = plot(select_ts(nums, timeseries), linewidth=2, legend=false, title="data")
	
	timeseriesplot = plot(select_ts(priornums, timeseries), linewidth=.5, color=:black, legend=false, title="data")
	
	l = @layout [a b ; c d]
	p = plot(dataplot, posteriorplot, timeseriesplot, priorplot, layout=l)
	p
end

# ╔═╡ 6c06ef9d-d3b4-4917-89f6-af0a3e72b4d1
plotmodel()

# ╔═╡ 9542f2b8-e746-4b37-b6f5-b8fd8a9d1876
data_sample = NeuralSDEExploration.sample_prior_dataspace(latent_sde,ps,st;b=500, tspan=latent_sde.tspan, datasize=latent_sde.datasize)

# ╔═╡ 6860e52e-8f51-41e1-b505-5b70ca112051
latent_sample = NeuralSDEExploration.sample_prior(latent_sde,ps,st;b=500, tspan=latent_sde.tspan, datasize=latent_sde.datasize)

# ╔═╡ 851ac161-d44c-47a7-89d5-c6c97d3ac8a6
mean_and_var_model = map_ts((ts) -> [mean(ts), std(ts)], data_sample)

# ╔═╡ 5fc6a274-5185-4d3f-808b-b1326f92081a
p_model = plot(select_ts(1:1, mean_and_var_model), ribbon=2*map(only, mean_and_var_model.u[2]))

# ╔═╡ de7959b4-87bd-4374-8fbd-c7a9f0e57d5a
mean_and_var_data = map_ts((ts) -> [mean(ts), std(ts)], select_ts(1:500, timeseries))

# ╔═╡ c275150e-e90c-4895-99c8-2f6364505dc0
p_data = plot!(p_model, select_ts(1:1, mean_and_var_data), ribbon=2*map(only, mean_and_var_data.u[2]))

# ╔═╡ 2a1ca1d0-163d-41b8-9f2d-8a3a475cc75d
function loss(ps, minibatch, eta, seed)
	_, _, _, kl_divergence, likelihood = latent_sde(minibatch, ps, st, seed=seed)
	return mean(-likelihood .+ (eta * kl_divergence)), mean(kl_divergence), mean(likelihood)
end

# ╔═╡ bf819c97-3ed9-484c-a499-7449244cb840
function kl_loss(ps, minibatch, eta, seed)
	_, _, _, kl_divergence, likelihood = latent_sde(minibatch, ps, st, seed=seed)
	return mean(kl_divergence), mean(kl_divergence), mean(likelihood)
end

# ╔═╡ 0dc46a16-e26d-4ec2-a74e-675e83959ab2
loss(ps, viz_batch, 10.0, seed)

# ╔═╡ 63e1f3d9-185a-4d7a-8535-3d21f2af1ee0
plot(latent_sample)

# ╔═╡ 905a440f-4a56-4205-aba0-558fd28f0bc0
tipping_rate(data_sample)

# ╔═╡ 99b80546-740e-485e-82a1-948f837ed696
tipping_rate(timeseries)

# ╔═╡ 96c0423f-214e-4995-a2e4-fe5c84d5a7c3
md"""
Histogram span: $(@bind hspan RangeSlider(1:20))
"""

# ╔═╡ a09063de-3002-400a-af89-233eb0822041
begin
	prior_latent = NeuralSDEExploration.sample_prior(latent_sde,ps,st;b=length(timeseries),seed=0,noise=noise)
	projected_prior = map(first, vcat([[latent_sde.projector(y[1:end-1], ps.projector, st.projector) for y in x[hspan].u] for x in prior_latent.u]...))
	#projected_prior = vcat(reduce(vcat, [latent_sde.projector(x[hspan], ps.projector, st.projector)[1] for x in prior_latent.u])...)
	histogram_prior = fit(Histogram, projected_prior, 0.0:0.01:1.0)
end

# ╔═╡ 0d74625b-edf2-45a7-9b16-08fc29d83eb0
loss(ps, timeseries[1:1], 30.0, rand(UInt32))

# ╔═╡ fe157d5e-eead-4921-a310-467e56e33fb7
begin
	ts = vcat(reduce(vcat, [s.u[hspan] for s in timeseries])...)
	histogram_data = fit(Histogram, ts, 0.0:0.01:1.0)
end

# ╔═╡ 2812e069-6bf9-4c80-91d7-f7fcf6c338fb
begin
	p_hist = plot([
		histogram_data,
		histogram_prior,
	], alpha=0.5, labels=["data" "prior"], title="marginal probabilities")
end

# ╔═╡ fe1ae4b3-2f1f-4b6c-a076-0d215f222e6c
plot_prior(25, rng=Xoshiro(), tspan=(0f0, 10f0), datasize=100)

# ╔═╡ 72045fd0-2769-4868-9b67-e7a41e3f1d7d
plot(NeuralSDEExploration.sample_prior(latent_sde,ps,st;b=1,tspan=(0f0,10f0),datasize=5000)
)

# ╔═╡ ae0c9dae-e490-4965-9353-c820a3ce3645
plot(NeuralSDEExploration.sample_prior_dataspace(latent_sde,ps,st;b=1,tspan=(0f0,10f0),datasize=5000,seed=1), title="Neural SDE")

# ╔═╡ 63e3e80c-6d03-4e42-a718-5bf30ad7182f
plot(filter_dims(1:1, NeuralSDEExploration.series(model, [[0f0, 0f0]], (0f0, 10f0), 5000,seed=1)), title="FitzHugh-Nagumo")

# ╔═╡ 9f8c49a0-2098-411b-976a-2b43cbb20a44
plot(NeuralSDEExploration.series(model, [[0f0, 0f0]], (0f0, 10f0), 5000))

# ╔═╡ ff5519b9-2a69-41aa-8f55-fc63fa176d3f
 plot(sample(timeseries, 25),linewidth=.5,color=:black,legend=false,title="data")


# ╔═╡ Cell order:
# ╠═7a6bbfd6-ffb7-11ed-39d7-5b673fe4cdae
# ╠═6995cd16-0c69-49c7-9523-4c842c0db339
# ╠═ae7d6e11-55da-44a2-a5b6-60d11caa9dbf
# ╟─43e6a892-8b2d-4fbb-b49c-681d78f0202c
# ╠═09a13088-a784-4d53-8012-cef948eb796c
# ╠═9f89a8d9-05e5-4af0-9dd8-1a528ea7e9de
# ╠═7da4b3b6-3cee-4b93-9359-a2f7e2341da9
# ╠═23bc5697-f1ca-4598-92fb-2d43e94ce310
# ╠═b214a6b2-d430-4d95-9f3a-b31c4ff7bcc7
# ╠═fabc1578-ba35-4c4e-9129-02da3bf43f56
# ╟─af3619b0-f9be-40f2-8027-77e435f8e4e5
# ╠═69459a5f-75b7-4c33-a489-bf4d4411c1ec
# ╠═e0afda9e-0b17-4e7e-9d1e-d0f05df6fa4e
# ╠═9c5f37b0-9998-4879-85d0-f540089e1ca8
# ╠═8f38bdbc-8b22-46c5-bbf4-d38d277b000f
# ╠═2f6d198e-4cae-40cd-8624-6aab867fee0b
# ╠═63213503-ab28-4158-b522-efd0b0139b6d
# ╠═0b47115c-0561-439b-be7b-78195da6215e
# ╠═6c06ef9d-d3b4-4917-89f6-af0a3e72b4d1
# ╠═9542f2b8-e746-4b37-b6f5-b8fd8a9d1876
# ╠═6860e52e-8f51-41e1-b505-5b70ca112051
# ╠═851ac161-d44c-47a7-89d5-c6c97d3ac8a6
# ╠═5fc6a274-5185-4d3f-808b-b1326f92081a
# ╠═de7959b4-87bd-4374-8fbd-c7a9f0e57d5a
# ╠═c275150e-e90c-4895-99c8-2f6364505dc0
# ╠═2a1ca1d0-163d-41b8-9f2d-8a3a475cc75d
# ╠═bf819c97-3ed9-484c-a499-7449244cb840
# ╠═0dc46a16-e26d-4ec2-a74e-675e83959ab2
# ╠═63e1f3d9-185a-4d7a-8535-3d21f2af1ee0
# ╠═905a440f-4a56-4205-aba0-558fd28f0bc0
# ╠═99b80546-740e-485e-82a1-948f837ed696
# ╟─96c0423f-214e-4995-a2e4-fe5c84d5a7c3
# ╠═a09063de-3002-400a-af89-233eb0822041
# ╠═0d74625b-edf2-45a7-9b16-08fc29d83eb0
# ╠═fe157d5e-eead-4921-a310-467e56e33fb7
# ╠═2812e069-6bf9-4c80-91d7-f7fcf6c338fb
# ╠═fe1ae4b3-2f1f-4b6c-a076-0d215f222e6c
# ╠═72045fd0-2769-4868-9b67-e7a41e3f1d7d
# ╟─ae0c9dae-e490-4965-9353-c820a3ce3645
# ╟─63e3e80c-6d03-4e42-a718-5bf30ad7182f
# ╠═9f8c49a0-2098-411b-976a-2b43cbb20a44
# ╠═ff5519b9-2a69-41aa-8f55-fc63fa176d3f
