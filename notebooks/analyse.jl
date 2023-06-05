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
using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs, JLD2, Random, StatsBase, Random123, DiffEqNoiseProcess

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

# ╔═╡ fabc1578-ba35-4c4e-9129-02da3bf43f56
timeseries = dict["timeseries"]

# ╔═╡ b5a4ca1c-0458-4fdb-b2eb-b29967c02158
noise = function(seed)
	rng = Xoshiro(seed)
	VirtualBrownianTree(-1f0, 0f0, tend=timeseries[1].t[end]+1f0; tree_depth=0, rng=Threefry4x((rand(rng, Int64), rand(rng, Int64), rand(rng, Int64), rand(rng, Int64))))
end

# ╔═╡ af3619b0-f9be-40f2-8027-77e435f8e4e5
md"""
Select timeseries to do some simulations on: $(@bind ti RangeSlider(1:100; default=1:4))
"""

# ╔═╡ 69459a5f-75b7-4c33-a489-bf4d4411c1ec
seed = 547842378

# ╔═╡ e0afda9e-0b17-4e7e-9d1e-d0f05df6fa4e
viz_batch = shuffle(Xoshiro(seed), timeseries)[ti]

# ╔═╡ 9c5f37b0-9998-4879-85d0-f540089e1ca8
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(viz_batch, ps, st; seed=seed, noise=noise)

# ╔═╡ 8f38bdbc-8b22-46c5-bbf4-d38d277b000f
plot(logterm_[1, :, :]', title="KL-Divergence")

# ╔═╡ 63213503-ab28-4158-b522-efd0b0139b6d
function plot_prior(priorsamples; rng=rng)
	prior_latent = NeuralSDEExploration.sample_prior(latent_sde,ps,st;seed=abs(rand(rng, Int)),b=priorsamples, noise=noise)
	projected_prior = reduce(hcat, [reduce(vcat, [latent_sde.projector(u, ps.projector, st.projector)[1] for u in batch.u]) for batch in prior_latent.u])
	priorplot = plot(projected_prior, linewidth=.5,color=:black,legend=false,title="projected prior")
	return priorplot
end

# ╔═╡ 0b47115c-0561-439b-be7b-78195da6215e
function plotmodel()
	posteriors = []
	priors = []
	posterior_latent = nothing
	datas = []
	n = 5
	rng = Xoshiro(0)
	nums = sample(rng,1:length(timeseries),n;replace=false)

	posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(timeseries[nums], ps, st, seed=seed, noise=noise)
	
	priorsamples = 25
	priorplot = plot_prior(priorsamples, rng=rng)

	posteriorplot = plot(posterior_data[1, :,:]',linewidth=2,legend=false,title="projected posterior")
	dataplot = plot(timeseries[nums],linewidth=2,legend=false,title="data")
	
	timeseriesplot = plot(sample(rng, timeseries, priorsamples),linewidth=.5,color=:black,legend=false,title="data")
	
	l = @layout [a b ; c d]
	p = plot(dataplot, posteriorplot, timeseriesplot, priorplot, layout=l)
	#p = plot(timeseriesplot)
	#posterior_latent
	p
end

# ╔═╡ 6c06ef9d-d3b4-4917-89f6-af0a3e72b4d1
plotmodel()

# ╔═╡ 2a1ca1d0-163d-41b8-9f2d-8a3a475cc75d
function loss(ps, minibatch, eta, seed)
	_, _, _, kl_divergence, likelihood = latent_sde(minibatch, ps, st, seed=seed, noise=noise)
	return mean(-likelihood .+ (eta * kl_divergence)), mean(kl_divergence), mean(likelihood)
end

# ╔═╡ 0dc46a16-e26d-4ec2-a74e-675e83959ab2
loss(ps, viz_batch, 1.0, seed)

# ╔═╡ 96c0423f-214e-4995-a2e4-fe5c84d5a7c3
md"""
Histogram span: $(@bind hspan RangeSlider(1:datasize))
"""

# ╔═╡ Cell order:
# ╠═7a6bbfd6-ffb7-11ed-39d7-5b673fe4cdae
# ╠═6995cd16-0c69-49c7-9523-4c842c0db339
# ╠═ae7d6e11-55da-44a2-a5b6-60d11caa9dbf
# ╟─43e6a892-8b2d-4fbb-b49c-681d78f0202c
# ╠═09a13088-a784-4d53-8012-cef948eb796c
# ╠═9f89a8d9-05e5-4af0-9dd8-1a528ea7e9de
# ╠═7da4b3b6-3cee-4b93-9359-a2f7e2341da9
# ╠═23bc5697-f1ca-4598-92fb-2d43e94ce310
# ╠═fabc1578-ba35-4c4e-9129-02da3bf43f56
# ╠═b5a4ca1c-0458-4fdb-b2eb-b29967c02158
# ╟─af3619b0-f9be-40f2-8027-77e435f8e4e5
# ╠═69459a5f-75b7-4c33-a489-bf4d4411c1ec
# ╠═e0afda9e-0b17-4e7e-9d1e-d0f05df6fa4e
# ╠═9c5f37b0-9998-4879-85d0-f540089e1ca8
# ╠═8f38bdbc-8b22-46c5-bbf4-d38d277b000f
# ╠═63213503-ab28-4158-b522-efd0b0139b6d
# ╠═0b47115c-0561-439b-be7b-78195da6215e
# ╠═6c06ef9d-d3b4-4917-89f6-af0a3e72b4d1
# ╠═2a1ca1d0-163d-41b8-9f2d-8a3a475cc75d
# ╠═0dc46a16-e26d-4ec2-a74e-675e83959ab2
# ╠═96c0423f-214e-4995-a2e4-fe5c84d5a7c3
