### A Pluto.jl notebook ###
# v0.19.24

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
using Optimisers, StatsBase, Zygote, ForwardDiff, Enzyme, Lux, DifferentialEquations, Functors, ComponentArrays, Distributions, ParameterSchedulers, Random, Flux

# ╔═╡ 682a8844-d1a8-4919-8a57-41b942b7da25
using NeuralSDEExploration, Plots, PlutoUI, ProfileSVG

# ╔═╡ d86a3b4e-6735-4de2-85d3-b6106ae48444
Revise.retry()

# ╔═╡ 9e4de245-815a-4d38-bb14-7b7b29da24cf
rng = Xoshiro()

# ╔═╡ c56ceab4-6ca7-4ea8-a905-24b5e9d8f0e1
begin
	initial_prior = Lux.Dense(1 => 4; bias=false, init_weight=zeros)
    initial_posterior = Lux.Dense(1 => 4; bias=false, init_weight=ones)
    drift_prior = Lux.Scale(2; init_weight=ones, bias=false)
    drift_posterior = Lux.Dense(3 => 2; init_weight=ones, bias=false)
	diffusion = Lux.Parallel(nothing, [Lux.Dense(1 => 1; init_weight=zeros, init_bias=ones) for i in 1:2]...)
    encoder = Lux.Recurrence(Lux.RNNCell(1 => 1, identity; init_weight=ones, init_state=zeros, init_bias=zeros); return_sequence=true)
    projector = Lux.Dense(2 => 1; bias=false, init_weight=ones)

    tspan = (0.0, 5.0)
    datasize = 4

    latent_sde = LatentSDE(
        initial_prior,
        initial_posterior,
        drift_prior,
        drift_posterior,
        diffusion,
        encoder,
        projector,
		EulerHeun(),
        tspan;
        saveat=range(tspan[1], tspan[end], datasize),
        dt=(tspan[end]/datasize),
    )
	ps__, st = Lux.setup(rng, latent_sde)
	ps_ = (initial_prior = (weight = [0.0; 0.0; 0.0; 0.0;;],), initial_posterior = (weight = [1.0; 1.0; 1.0; 1.0;;],), drift_prior = (weight = [1.0, 1.0],), drift_posterior = (weight = [1.0 1.0 1.0; 1.0 1.0 1.0],), diffusion = (layer_1 = (weight = [0.0;;], bias = [1.0;;]), layer_2 = (weight = [0.0;;], bias = [1.0;;])), encoder = (weight_ih = [1.0;;], weight_hh = [1.0;;], bias = [0.0]), projector = (weight = [1.0 1.0],))
	println(ps_)
	ps = ComponentArray(ps_)
end

# ╔═╡ 02b7abfe-e12a-41d5-a407-a249323ddb3e
begin
	initial_prior_flux = Flux.Dense(1 => 4; bias=false, init=zeros) |> f64
    initial_posterior_flux = Flux.Dense(reshape([1.0, 1.0, 1.0, 1.0], :, 1), false) |> f64
    drift_prior_flux = Flux.Scale([1.0, 1.0], false) |> f64
    drift_posterior_flux = Flux.Dense(ones(2, 3), false) |> f64
    diffusion_flux = [Flux.Dense(1 => 1, bias=[1.0], init=zeros) |> f64 for i in 1:2]
    encoder_flux = Flux.RNN((x) -> x, reshape([1.0], 1, 1), reshape([1.0], 1, 1), reshape([0.0], 1, 1), reshape([0.0], 1, 1)) |> f64
    projector_flux = Flux.Dense(2 => 1; bias=false, init=ones) |> f64

    latent_sde_flux = LatentSDEFlux(
        initial_prior_flux,
        initial_posterior_flux,
        drift_prior_flux,
        drift_posterior_flux,
        diffusion_flux,
        encoder_flux,
        projector_flux,
        tspan,
		EulerHeun();
        saveat=range(tspan[1], tspan[end], datasize),
        dt=(tspan[end]/datasize),
    )
    ps_flux_, re = Functors.functor(latent_sde_flux)
    ps_flux = ComponentArray(ps_flux_)
end

# ╔═╡ 9d2a583f-e285-41d0-8911-19aa8dff73aa
initial_prior(reshape([42.0], 1, 1), ps.initial_prior, st.initial_prior) 

# ╔═╡ 4176d447-5b4a-47eb-aead-1bac0d983aa6
initial_prior_flux(reshape([42.0]))

# ╔═╡ ddbf32ad-de72-4d03-88db-4e38f5485a3e
initial_posterior(reshape([42.0], 1, 1), ps.initial_posterior, st.initial_posterior) 

# ╔═╡ 54dccf1c-42ab-4f93-bc0e-37c0ab39c541
initial_posterior_flux(reshape([42.0]))

# ╔═╡ 15e4ff79-0753-4c38-9967-07f8ffd09326
drift_prior(reshape([42.0, 11.0], :, 1), ps.drift_prior, st.drift_prior) 

# ╔═╡ b045daee-e966-4c0c-97cb-60d7bd41ac38
drift_prior_flux(reshape([42.0, 11.0], :, 1))

# ╔═╡ 17023016-849b-49b9-a592-1224d2c6ebbc
drift_posterior(reshape([42.0, 11.0, 0.9], :, 1), ps.drift_posterior, st.drift_posterior) 

# ╔═╡ 40c71faf-7477-4be2-87a5-79f9333f8398
drift_posterior_flux(reshape([42.0, 11.0, 0.9], :, 1))

# ╔═╡ 23601b08-2d78-426c-9702-53232c0db1bc
display(ps_)

# ╔═╡ 63af22d4-65cd-40c8-8ff6-9c1321d15bb1
display(ps_flux_)

# ╔═╡ af673c70-c7bf-4fe6-92c0-b5e09fd99195
inputs = [(t=range(tspan[1],tspan[end],datasize),u=[f(x) for x in range(tspan[1],tspan[end],datasize)]) for f in [(x)->-x,(x)->x]]

# ╔═╡ 24d1e95b-08d1-463c-87c5-b71dc6397624
timeseries = inputs

# ╔═╡ 0522b60f-3817-424c-bb32-dbc50769a025
plot(inputs)

# ╔═╡ 7b81e46b-c55a-42e4-af34-9d1101de4b9c
@bind seed Slider(1:100)

# ╔═╡ 02bc2a6e-1497-4bf3-ac67-26c770425c22
m1 = reshape([.1, .1, .1, .1], 1, :, 1)

# ╔═╡ ae0c97bf-a307-4e61-83c1-8bf84d2d7f0a
m2 = reshape([.5, 0.0, .7, .8], 1, :, 1)

# ╔═╡ 0903e805-2147-42fd-919f-5bf2103df859
m3 = cat(m1, m2; dims=3)

# ╔═╡ 6b826904-f134-40cf-a5a8-0f2aaf89e562
ps.encoder

# ╔═╡ 78ad30c4-38b9-4f1a-8cf8-9174708286e1
encoder(m1, ps.encoder, st.encoder)

# ╔═╡ 6c872660-6f3d-44c0-a302-44238d8b14a7
begin
	Flux.reset!(encoder_flux)
	encoder_flux(permutedims(m2, (1, 3, 2)))
end

# ╔═╡ 7dfb3923-c0fc-4f88-9104-7d619ff415cd
encoder(m2, ps.encoder, st.encoder)

# ╔═╡ 1bcb6fba-8286-4008-98ff-d581760ffff6
encoder(m3, ps.encoder, st.encoder)

# ╔═╡ e8ef1773-8087-4f47-abfe-11e73f28a269
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, inputs, st; ensemblemode=EnsembleThreads(), seed=10)

# ╔═╡ 5769cf41-397e-47a3-bc2a-1e69f15df2f5
posterior_latent_flux, posterior_data_flux, logterm_flux_, kl_divergence_flux_, distance_flux_ = NeuralSDEExploration.pass(latent_sde_flux, ps_flux, inputs; ensemblemode=EnsembleThreads(), seed=10)

# ╔═╡ 7fabae7c-abef-4d2c-a3b5-b9d2f683dc26
plot(logterm_[1, :, :]', label="KL-Divergence")

# ╔═╡ 298f9785-adbe-44de-8c04-e77fac73c312
plot(logterm_flux_[1, :, :]', label="KL-Divergence")

# ╔═╡ 937d5963-eddc-4296-9b6e-9532eb57bdf2
plot(posterior_data[1, :, :]', label="posterior")

# ╔═╡ 0e217ea5-463f-4366-8bf8-672a08cfe15c
plot(posterior_data_flux[1, :, :]', label="posterior")

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

# ╔═╡ 04d8ee06-9872-4481-8df5-26d47862261b


# ╔═╡ 6bf6a59c-549c-495b-a574-caa12c87e055
dps = Zygote.gradient(ps -> mean(NeuralSDEExploration.loss(latent_sde, ps, inputs, st, 1.0; seed=1, ensemblemode=EnsembleSerial())), ps)[1]

# ╔═╡ ada03e49-be82-418a-a915-efbc78a81368
println(dps)

# ╔═╡ e99f0f6a-950b-4ccc-8798-a3a10730b4f5
println(ps_flux)

# ╔═╡ 83e1b7df-9cc4-4504-936e-69028ed7ee02
dps_flux = Zygote.gradient(ps -> mean(NeuralSDEExploration.loss(latent_sde_flux, ps, inputs, 1.0; seed=1, ensemblemode=EnsembleSerial())), ps_flux)[1]

# ╔═╡ 07e32ac3-f419-46a5-91cb-4f88eb57c7e3
println(dps_flux)

# ╔═╡ Cell order:
# ╠═65947e8c-d1f3-11ed-2067-93e06230d83c
# ╠═d86a3b4e-6735-4de2-85d3-b6106ae48444
# ╠═9f89ad4a-4ffc-4cc8-bd7d-916ff6d6fa10
# ╠═682a8844-d1a8-4919-8a57-41b942b7da25
# ╠═9e4de245-815a-4d38-bb14-7b7b29da24cf
# ╠═c56ceab4-6ca7-4ea8-a905-24b5e9d8f0e1
# ╠═02b7abfe-e12a-41d5-a407-a249323ddb3e
# ╠═9d2a583f-e285-41d0-8911-19aa8dff73aa
# ╠═4176d447-5b4a-47eb-aead-1bac0d983aa6
# ╠═ddbf32ad-de72-4d03-88db-4e38f5485a3e
# ╠═54dccf1c-42ab-4f93-bc0e-37c0ab39c541
# ╠═15e4ff79-0753-4c38-9967-07f8ffd09326
# ╠═b045daee-e966-4c0c-97cb-60d7bd41ac38
# ╠═17023016-849b-49b9-a592-1224d2c6ebbc
# ╠═40c71faf-7477-4be2-87a5-79f9333f8398
# ╠═23601b08-2d78-426c-9702-53232c0db1bc
# ╠═63af22d4-65cd-40c8-8ff6-9c1321d15bb1
# ╠═af673c70-c7bf-4fe6-92c0-b5e09fd99195
# ╠═24d1e95b-08d1-463c-87c5-b71dc6397624
# ╠═0522b60f-3817-424c-bb32-dbc50769a025
# ╠═7b81e46b-c55a-42e4-af34-9d1101de4b9c
# ╠═02bc2a6e-1497-4bf3-ac67-26c770425c22
# ╠═ae0c97bf-a307-4e61-83c1-8bf84d2d7f0a
# ╠═0903e805-2147-42fd-919f-5bf2103df859
# ╠═6b826904-f134-40cf-a5a8-0f2aaf89e562
# ╠═78ad30c4-38b9-4f1a-8cf8-9174708286e1
# ╠═6c872660-6f3d-44c0-a302-44238d8b14a7
# ╠═7dfb3923-c0fc-4f88-9104-7d619ff415cd
# ╠═1bcb6fba-8286-4008-98ff-d581760ffff6
# ╠═e8ef1773-8087-4f47-abfe-11e73f28a269
# ╠═5769cf41-397e-47a3-bc2a-1e69f15df2f5
# ╠═7fabae7c-abef-4d2c-a3b5-b9d2f683dc26
# ╠═298f9785-adbe-44de-8c04-e77fac73c312
# ╠═937d5963-eddc-4296-9b6e-9532eb57bdf2
# ╠═0e217ea5-463f-4366-8bf8-672a08cfe15c
# ╠═7072a7d4-fd83-40a8-87fe-bd47c5a430eb
# ╠═0ec36eba-8887-4ec2-9893-7ae5774c9337
# ╠═7e231cb4-f76b-49b9-91f5-785603a4afd9
# ╠═ee76f88c-069e-45b1-bce0-6019983b5260
# ╠═04d8ee06-9872-4481-8df5-26d47862261b
# ╠═6bf6a59c-549c-495b-a574-caa12c87e055
# ╠═ada03e49-be82-418a-a915-efbc78a81368
# ╠═e99f0f6a-950b-4ccc-8798-a3a10730b4f5
# ╠═83e1b7df-9cc4-4504-936e-69028ed7ee02
# ╠═07e32ac3-f419-46a5-91cb-4f88eb57c7e3
