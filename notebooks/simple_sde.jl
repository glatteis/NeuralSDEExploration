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

# ╔═╡ 65947e8c-d1f3-11ed-2067-93e06230d83c
begin
	import Pkg
	Pkg.activate("..")
	using Revise
end

# ╔═╡ 9f89ad4a-4ffc-4cc8-bd7d-916ff6d6fa10
using Optimisers, StatsBase, Zygote, ForwardDiff, Enzyme, Lux, DifferentialEquations, Functors, ComponentArrays, Distributions, ParameterSchedulers, Random

# ╔═╡ 682a8844-d1a8-4919-8a57-41b942b7da25
using NeuralSDEExploration, Plots, PlutoUI, ProfileSVG

# ╔═╡ d86a3b4e-6735-4de2-85d3-b6106ae48444
Revise.retry()

# ╔═╡ 9e4de245-815a-4d38-bb14-7b7b29da24cf
rng = Xoshiro()

# ╔═╡ e6a96c38-cc43-4f1a-8281-a3bd9e7d12c3
function steps(tspan, dt)
	return Int(ceil((tspan[2] - tspan[1]) / dt))
end

# ╔═╡ c56ceab4-6ca7-4ea8-a905-24b5e9d8e0e1
begin
	initial_prior = Lux.Dense(1 => 4; bias=false, init_weight=zeros)
    initial_posterior = Lux.Dense(1 => 4; bias=false, init_weight=ones)
    drift_prior = Lux.Dense(2 => 2; init_weight=ones, bias=false)
    drift_posterior = Lux.Dense(3 => 2; init_weight=ones, bias=false)
	diffusion = Diagonal([Lux.Dense(1 => 1; init_weight=zeros, init_bias=ones) for i in 1:2]...)
    encoder = Lux.Recurrence(Lux.RNNCell(1 => 1, identity); return_sequence=true)
	encoder_net = Lux.Dense(1 => 1; bias=false, init_weight=ones)
    projector = Lux.Dense(2 => 1; bias=false, init_weight=ones)

    tspan = (0f0, 5f0)
    datasize = 20

    latent_sde = LatentSDE(
        initial_prior,
        initial_posterior,
        drift_prior,
        drift_posterior,
        diffusion,
        encoder,
		encoder_net,
        projector,
		EulerHeun(),
        tspan,
		datasize
    )
	ps__, st_ = Lux.setup(rng, latent_sde)
	println(ps__)
	ps_ = (initial_prior = (weight = [0.0; 0.0; 0.0; 0.0;;],), initial_posterior = (weight = [0.0; 0.0; 0.0; 0.0;;],), drift_prior = (weight = [0.8 1.0; 1.0 1.0],), drift_posterior = (weight = [0.0 0.0 1.0; 0.0 0.0 1.0],), diffusion = (layer_1 = (weight = [0.0;;], bias = [0.0001;;]), layer_2 = (weight = [0.0;;], bias = [0.0001;;])), encoder_recurrent = (weight_ih = [1.0;;], weight_hh = [1.0;;], bias = [0.0]), encoder_net = (weight = [1.0;;],), projector = (weight = [1.0 1.0],))
	ps = ComponentArray{Float32}(ps_) |> Lux.gpu
	st = st_ |> Lux.gpu
end

# ╔═╡ 9d2a583f-e285-41d0-8911-19aa8dff73aa
initial_prior(reshape([42f0], 1, 1) |> gpu, ps.initial_prior, st.initial_prior) 

# ╔═╡ ddbf32ad-de72-4d03-88db-4e38f5485a3e
initial_posterior(reshape([42.0], 1, 1) |> gpu, ps.initial_posterior, st.initial_posterior) 

# ╔═╡ 15e4ff79-0753-4c38-9967-07f8ffd09326
drift_prior(reshape([42.0, 11.0], :, 1) |> gpu, ps.drift_prior, st.drift_prior) 

# ╔═╡ 17023016-849b-49b9-a592-1224d2c6ebbc
drift_posterior(reshape([42.0, 11.0, 0.0], :, 1) |> gpu, ps.drift_posterior, st.drift_posterior) 

# ╔═╡ 23601b08-2d78-426c-9702-53232c0db1bc
display(ps_)

# ╔═╡ 1e1100d2-0452-4754-8ab3-cecba9602909
inputs = [(t=range(tspan[1],tspan[end],datasize),u=[f(x) for x in range(tspan[1],tspan[end],datasize)]) for f in [(x)->0f0, (x)->100+x, (x)->-x-100, (x)->100+x]]

# ╔═╡ 24d1e95b-08d1-463c-87c5-b71dc6397624
timeseries = Timeseries(inputs)

# ╔═╡ 0522b60f-3817-424c-bb32-dbc50769a025
plot(inputs)

# ╔═╡ 7b81e46b-c55a-42e4-af34-9d1101de4b9c
@bind seed Slider(1:100)

# ╔═╡ 02bc2a6e-1497-4bf3-ac67-26c770425c22
m1 = reshape([.1, .1, .1, .1], 1, :, 1)

# ╔═╡ ae0c97bf-a307-4e61-83c1-8bf84d2d7e0a
m2 = reshape([.5, 0.0, .7, .8], 1, :, 1)

# ╔═╡ 0903e805-2147-42fd-919f-5bf2103df859
m3 = cat(m1, m2; dims=3)

# ╔═╡ ccad1e15-c7d7-4d66-a637-7108fab46841
plot(timeseries)

# ╔═╡ e8ef1773-8087-4f47-abfe-11e73f28a269
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(timeseries, ps, st)

# ╔═╡ c6e86e18-8499-49b3-bf65-73afa4dfd45a
plot(Timeseries(collect(range(tspan[1],tspan[end],datasize)), posterior_latent), label="posterior")

# ╔═╡ b4caf08a-6a9a-403d-b537-a80a197d8c31
kl_divergence_

# ╔═╡ 1d7ea638-dff8-44c6-9ccf-97c898916834
distance_

# ╔═╡ 0615efd2-c125-48cc-9600-0e2fb6a25a0d
plot(sample_prior(latent_sde, ps, st))

# ╔═╡ 7fabae7c-abef-4d2c-a3b5-b9d2f683dc26
plot(logterm_[1, :, :]', label="KL-Divergence")

# ╔═╡ 445423ba-5b52-438b-afbf-e6399c133779
posterior_latent

# ╔═╡ 7072a7d4-fd83-40a8-87fe-bd47c5a430eb
function loss(ps, minibatch)
	mean(NeuralSDEExploration.loss(latent_sde, ps, minibatch, st, 1.0; ensemblemode=EnsembleSerial()))
end

# ╔═╡ 0ec36eba-8887-4ec2-9893-7ae5774c9337
i = 0

# ╔═╡ 7e231cb4-f76b-49b9-91f5-785603a4afd9
function cb()
	posteriors = []
	priors = []
	posterior_latent = nothing
	datas = []
	n = 4
	rng = Xoshiro(294)
	nums = sample(rng,1:length(timeseries),n;replace=false)

	posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, timeseries[nums], st, seed=seed+i)
	
	priorsamples = 20
	prior_latent = NeuralSDEExploration.sample_prior(latent_sde,ps,st;seed=abs(rand(rng, Int)),b=priorsamples)
	projected_prior = vcat([latent_sde.projector(x, ps.projector, st.projector)[1] for x in prior_latent.u]...)

	posteriorplot = plot(posterior_data[1, :,:]',linewidth=2,legend=false,title="projected posterior")
	dataplot = plot(timeseries[nums],linewidth=2,legend=false,title="data")
	priorplot = plot(projected_prior, linewidth=.5,color=:grey,legend=false,title="projected prior")
	
	timeseriesplot = plot(sample(rng, timeseries, priorsamples),linewidth=.5,color=:grey,legend=false,title="data")
	
	l = @layout [a b ; c d]
	p = plot(dataplot, posteriorplot, timeseriesplot, priorplot, layout=l)
	#p = plot(timeseriesplot)
	#posterior_latent
	p
end

# ╔═╡ ee76f88c-069e-45b1-bce0-6019983b5260
function train(learning_rate, num_steps)
	ar = 20 # annealing rate
	sched = Loop(Sequence([Loop(x -> (50*x)/ar, ar), Loop(x -> 50.0, ar)], [ar, ar]), ar*2)

	opt_state = Optimisers.setup(Optimisers.Adam(), ps)
	for (step, eta) in zip(1:num_steps, sched)
		s = sample(rng, 1:size(inputs)[1], 4, replace=false)
		minibatch = inputs[s]
		l = loss(ps, minibatch)
		println("Loss: $l")
		dps = Zygote.gradient(ps -> loss(ps, minibatch), ps)[1]
		Optimisers.update!(opt_state, ps, dps)
	end
end

# ╔═╡ Cell order:
# ╠═65947e8c-d1f3-11ed-2067-93e06230d83c
# ╠═d86a3b4e-6735-4de2-85d3-b6106ae48444
# ╠═9f89ad4a-4ffc-4cc8-bd7d-916ff6d6fa10
# ╠═682a8844-d1a8-4919-8a57-41b942b7da25
# ╠═9e4de245-815a-4d38-bb14-7b7b29da24cf
# ╠═e6a96c38-cc43-4f1a-8281-a3bd9e7d12c3
# ╠═c56ceab4-6ca7-4ea8-a905-24b5e9d8e0e1
# ╠═9d2a583f-e285-41d0-8911-19aa8dff73aa
# ╠═ddbf32ad-de72-4d03-88db-4e38f5485a3e
# ╠═15e4ff79-0753-4c38-9967-07f8ffd09326
# ╠═17023016-849b-49b9-a592-1224d2c6ebbc
# ╠═23601b08-2d78-426c-9702-53232c0db1bc
# ╠═1e1100d2-0452-4754-8ab3-cecba9602909
# ╠═24d1e95b-08d1-463c-87c5-b71dc6397624
# ╠═0522b60f-3817-424c-bb32-dbc50769a025
# ╠═7b81e46b-c55a-42e4-af34-9d1101de4b9c
# ╠═02bc2a6e-1497-4bf3-ac67-26c770425c22
# ╠═ae0c97bf-a307-4e61-83c1-8bf84d2d7e0a
# ╠═0903e805-2147-42fd-919f-5bf2103df859
# ╠═ccad1e15-c7d7-4d66-a637-7108fab46841
# ╠═e8ef1773-8087-4f47-abfe-11e73f28a269
# ╠═c6e86e18-8499-49b3-bf65-73afa4dfd45a
# ╠═b4caf08a-6a9a-403d-b537-a80a197d8c31
# ╠═1d7ea638-dff8-44c6-9ccf-97c898916834
# ╠═0615efd2-c125-48cc-9600-0e2fb6a25a0d
# ╠═7fabae7c-abef-4d2c-a3b5-b9d2f683dc26
# ╠═445423ba-5b52-438b-afbf-e6399c133779
# ╠═7072a7d4-fd83-40a8-87fe-bd47c5a430eb
# ╠═0ec36eba-8887-4ec2-9893-7ae5774c9337
# ╠═7e231cb4-f76b-49b9-91f5-785603a4afd9
# ╠═ee76f88c-069e-45b1-bce0-6019983b5260
