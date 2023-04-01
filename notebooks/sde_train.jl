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

# ╔═╡ 67cb574d-7bd6-40d9-9dc3-d57f4226cc83
begin
	import Pkg
	Pkg.activate("..")
	using Revise
end

# ╔═╡ b6abba94-db07-4095-98c9-443e31832e7d
using Optimisers, StatsBase, Zygote, ForwardDiff, Enzyme, Flux, DifferentialEquations, Functors, ComponentArrays, Distributions, ParameterSchedulers, Random

# ╔═╡ d1440209-78f7-4a9a-9101-a06ad2534e5d
using NeuralSDEExploration, Plots, PlutoUI, ProfileSVG

# ╔═╡ 080d6dd1-0aa6-4bee-bf50-20d8ca890145
begin
	using SciMLBase
	methods(SciMLBase.is_diagonal_noise)
end

# ╔═╡ 32be3e35-a529-4d16-8ba0-ec4e223ae401
md"""
Let's train a Neural SDE from a modified form of the simple zero-dimensional energy balance model. First, let's just instantiate the predefined model from the package...
"""

# ╔═╡ f74dd752-485b-4203-9d72-c56e55a3ef76
ebm = NeuralSDEExploration.ZeroDEnergyBalanceModel()

# ╔═╡ cc2418c2-c355-4291-b5d7-d9019787834f
md"Let's generate the data and plot a quick example:"

# ╔═╡ dd03f851-2e26-4850-a7d4-a64f154d2872
begin
	n = 1000
    datasize = 200
    tspan = (0.0f0, 10f0)
end

# ╔═╡ c00a97bf-5e10-4168-8d58-f4f9270258ac
solution = NeuralSDEExploration.series(ebm, range(210.0f0, 320.0f0, n), tspan, datasize)

# ╔═╡ 1502612c-1489-4abf-8a8b-5b2d03a68cb1
md"""
Let's also plot a single example trajectory:
"""

# ╔═╡ f4651b27-135e-45f1-8647-64ab08c2e8e8
md"""
Let's normalize our data for training:
"""

# ╔═╡ aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
begin
    datamax = max([max(x.u...) for x in solution]...)
    datamin = min([min(x.u...) for x in solution]...)

    function normalize(x)
        return ((x - datamin) / (datamax - datamin)) + 1.0
    end

    function rescale(x)
        return ((x - 1.0) * (datamax - datamin)) + datamin
    end
end

# ╔═╡ 9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
timeseries = [(t=ts.t, u=map(normalize, ts.u)) for ts in solution]

# ╔═╡ da11fb69-a8a1-456d-9ce7-63180ef27a83
md"""
We are going to build a simple latent SDE. Define a few constants...
"""

# ╔═╡ 22453ba0-81a7-43f6-bb80-91d3c991530d
context_size = 4 # The size of the context given to the posterior SDE.

# ╔═╡ d81ccb5f-de1c-4a01-93ce-3e7302caedc0
hidden_size = 10 # The hidden layer size for all ANNs.

# ╔═╡ b5721107-7cf5-4da3-b22a-552e3d56bcfa
latent_dims = 3 # Dimensions of the latent space.

# ╔═╡ 9767a8ea-bdda-43fc-b636-8681d150d29f
data_dims = 1 # Dimensions of our input data.

# ╔═╡ 39154e76-01f7-4710-8f21-6f3a7d3fcfcd
md"""
The encoder takes a timeseries and outputs context that can be passed to the posterior SDE, that is, the SDE that has information about the data and encodes $p_\theta(z \mid x)$.
"""

# ╔═╡ 847587ec-9297-471e-b788-7b776c05851e
encoder = Flux.Chain(Flux.GRU(data_dims => 4), Flux.Dense(4 => context_size, tanh)) |> f64

# ╔═╡ 95f42c13-4a06-487c-bd35-a8d6bac9e02e
@bind i Slider(1:length(timeseries))

# ╔═╡ 63632673-4374-450c-a48d-493ae7a6b1ce
begin
	Flux.reset!(encoder)
	example_context = encoder(reshape(timeseries[i].u, 1, 1, :))[:, 1, :]
end

# ╔═╡ 016ab596-424b-4c50-b5c0-46d08aa47909
plot(example_context')

# ╔═╡ bdf4c32c-9a2b-4814-8513-c6e16ebee69c
plot(timeseries[i].u, legend=nothing)

# ╔═╡ 0d09a662-78d3-4202-b2be-843a0669fc9f
md"""
The `initial_posterior` net is the posterior for the initial state. It takes the context and outputs a mean and standard devation for the position zero of the posterior. The `initial_prior` is a fixed gaussian distribution.
"""

# ╔═╡ cdebb87f-9759-4e6b-a00a-d764b3c7fbf8
initial_posterior = Flux.Dense(context_size => latent_dims + latent_dims) |> f64

# ╔═╡ 3ec28482-2d6c-4125-8d85-4fb46b130677
initial_prior = Flux.Dense(1 => latent_dims + latent_dims, init=Flux.glorot_uniform) |> f64

# ╔═╡ 8b283d5e-1ce7-4deb-b382-6fa8e5612ef1
md"""
Drift of prior. This is just an SDE drift in the latent space
"""

# ╔═╡ c14806bd-42cf-480b-b618-bfe72183feb3
drift_prior = Flux.Chain(
	Flux.Dense(latent_dims => hidden_size, tanh),
	Flux.Dense(hidden_size => hidden_size, tanh),
	Flux.Dense(hidden_size => hidden_size, tanh),
	Flux.Dense(hidden_size => latent_dims, tanh),
	Flux.Scale(latent_dims)
) |> f64

# ╔═╡ 64dc2da0-48cc-4242-bb17-449a300688c7
md"""
Drift of posterior. This is the term of an SDE when fed with the context.
"""

# ╔═╡ df2034fd-560d-4529-836a-13745f976c1f
drift_posterior = Flux.Chain(
	Flux.Dense(latent_dims + context_size => hidden_size, tanh),
	Flux.Dense(hidden_size => hidden_size, tanh),
	Flux.Dense(hidden_size => latent_dims, tanh),
	Flux.Scale(latent_dims)
) |> f64

# ╔═╡ 4a97576c-96de-458e-b881-3f5dd140fa6a
md"""
Diffusion. Prior and posterior share the same diffusion (they are not actually evaluated seperately while training, only their KL divergence). This is a diagonal diffusion, i.e. every term in the latent space has its own independent Wiener process.
"""

# ╔═╡ a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9

diffusion = [Flux.Chain(Flux.Dense(1 => 4, softplus), Flux.Dense(4 => 1, sigmoid)) |> f64 for i in 1:latent_dims]

# ╔═╡ bfabcd80-fb62-410f-8710-f577852c77df
md"""
The projector will transform the latent space back into data space.
"""

# ╔═╡ f0486891-b8b3-4a39-91df-1389d6f799e1
projector = Flux.Dense(latent_dims => data_dims) |> f64

# ╔═╡ 001c318e-b7a6-48a5-bfd5-6dd0368873ac
latent_sde = LatentSDE(
	initial_prior,
	initial_posterior,
	drift_prior,
	drift_posterior,
	diffusion,
	encoder,
	projector,
    tspan,
	EulerHeun();
	#SOSRA2();
	saveat=range(tspan[1], tspan[end], datasize),
	dt=(tspan[end]/1e3),
)

# ╔═╡ 05568880-f931-4394-b31e-922850203721
ps_, re = Functors.functor(latent_sde)

# ╔═╡ 5b073f11-08bd-4128-85c6-7f043b60bdb8
ps = ComponentArray(ps_)

# ╔═╡ 9ea12ddb-ff8a-4c16-b2a5-8b7603f262a3
md"""
Integrate the prior SDE in latent space to see what a run looks like:
"""

# ╔═╡ 9346f569-d5f9-43cd-9302-1ee64ef9a030
plot(NeuralSDEExploration.sample_prior(latent_sde, ps, b=5))

# ╔═╡ b98200a8-bf73-42a2-a357-af56812d01c3
md"""
Integrate the posterior SDE in latent space given context:
"""

# ╔═╡ ce6a7357-0dd7-4581-b786-92d1eddd114d
plot(NeuralSDEExploration.sample_posterior(latent_sde, timeseries[1:1]))

# ╔═╡ 26885b24-df80-4fbf-9824-e175688f1322
@bind seed Slider(1:1000)

# ╔═╡ 5f56cc7c-861e-41a4-b783-848623b94bf9
@bind ti Slider(1:length(timeseries))

# ╔═╡ 88fa1b08-f0d4-4fcf-89c2-8a9f33710d4c
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, timeseries[ti:ti+10]; seed=seed)

# ╔═╡ dabf2a1f-ec78-4942-973f-4dbf9037ee7b
plot(logterm_[1, :, :]', label="KL-Divergence")

# ╔═╡ 38324c42-e5c7-4b59-8129-0e4c17ab5bf1
plot(posterior_data[1, :, :]', label="posterior")

# ╔═╡ 08021ed6-ac31-4829-9f21-f046af73d5a3
plot(posterior_latent[:, 1, :]')

# ╔═╡ 92c138d3-05f6-4a57-9b6b-08f41cb4ea55
plot(NeuralSDEExploration.sample_posterior(latent_sde, timeseries[1:1], seed=seed))

# ╔═╡ 3d889727-ae6d-4fa0-98ae-d3ae73fb6a3c
plot(NeuralSDEExploration.sample_prior(latent_sde, ps, seed=seed))

# ╔═╡ b5c6d43c-8252-4602-8232-b3d1b0bcee33
function cb()
	posteriors = []
	priors = []
	posterior_latent = nothing
	datas = []
	n = 5
	rng = Xoshiro(1235)
	nums = sample(rng,1:length(timeseries),n;replace=false)

	posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, timeseries[nums], seed=seed+i)
	
	priorsamples = 100
	prior_latent = NeuralSDEExploration.sample_prior(latent_sde,ps,seed=seed+i;b=priorsamples)
	projected_prior = vcat([latent_sde.projector_re(ps.projector_p)(x) for x in prior_latent.u]...)

	posteriorplot = plot(posterior_data[1, :,:]',linewidth=2,legend=false,title="projected posterior")
	dataplot = plot(timeseries[nums],linewidth=2,legend=false,title="data")
	priorplot = plot(projected_prior, linewidth=.2,color=:grey,legend=false,title="projected prior")
	
	timeseriesplot = plot(sample(rng, timeseries, priorsamples),linewidth=.2,color=:grey,legend=false,title="data")
	
	l = @layout [a b ; c d]
	p = plot(dataplot, posteriorplot, timeseriesplot, priorplot, layout=l)
	#p = plot(timeseriesplot)
	#posterior_latent
	p
end

# ╔═╡ 025b33d9-7473-4a54-a3f1-787a8650f9e7
begin
	pl = cb()
	savefig(pl, "train.pdf")
	pl
end

# ╔═╡ f4a16e34-669e-4c93-bd83-e3622a747a3a
function train(learning_rate, num_steps)
	
	ar = 20 # annealing rate
	sched = Loop(Sequence([Loop(x -> (50*x)/ar, ar), Loop(x -> 50.0, ar)], [ar, ar]), ar*2)

	
	opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)
	anim = @animate for (step, eta) in zip(1:num_steps, sched)
		minibatch = timeseries[sample(1:size(timeseries)[1], 20, replace=false)]
		if step % 10 == 0
			cb()
		end
		function loss(ps)
			sum(NeuralSDEExploration.loss(latent_sde, ps, minibatch, 30.0))
		end
		l = loss(ps)
		println("Loss: $l")
		grads = Zygote.gradient(loss, ps)[1]
		#return Vector(grads)
		#grads = ForwardDiff.gradient(loss, ps)
		#println("Grads: $grads")
		Optimisers.update!(opt_state, ps, grads)
	end
	return gif(anim)
end

# ╔═╡ 2ada3ddd-b47a-46ff-abba-17cbb94041a2
md"Enable training $(@bind enabletraining CheckBox())"

# ╔═╡ 1a958c8e-b2eb-4ed2-a7d6-2f1a52c5ac7a
begin
	if enabletraining
		train(0.035, 1)
	end
end

# ╔═╡ Cell order:
# ╠═67cb574d-7bd6-40d9-9dc3-d57f4226cc83
# ╠═b6abba94-db07-4095-98c9-443e31832e7d
# ╠═d1440209-78f7-4a9a-9101-a06ad2534e5d
# ╟─32be3e35-a529-4d16-8ba0-ec4e223ae401
# ╠═f74dd752-485b-4203-9d72-c56e55a3ef76
# ╟─cc2418c2-c355-4291-b5d7-d9019787834f
# ╠═dd03f851-2e26-4850-a7d4-a64f154d2872
# ╠═c00a97bf-5e10-4168-8d58-f4f9270258ac
# ╟─1502612c-1489-4abf-8a8b-5b2d03a68cb1
# ╟─f4651b27-135e-45f1-8647-64ab08c2e8e8
# ╠═aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
# ╠═9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
# ╟─da11fb69-a8a1-456d-9ce7-63180ef27a83
# ╠═22453ba0-81a7-43f6-bb80-91d3c991530d
# ╠═d81ccb5f-de1c-4a01-93ce-3e7302caedc0
# ╠═b5721107-7cf5-4da3-b22a-552e3d56bcfa
# ╠═9767a8ea-bdda-43fc-b636-8681d150d29f
# ╟─39154e76-01f7-4710-8f21-6f3a7d3fcfcd
# ╠═847587ec-9297-471e-b788-7b776c05851e
# ╠═95f42c13-4a06-487c-bd35-a8d6bac9e02e
# ╠═63632673-4374-450c-a48d-493ae7a6b1ce
# ╠═016ab596-424b-4c50-b5c0-46d08aa47909
# ╠═bdf4c32c-9a2b-4814-8513-c6e16ebee69c
# ╟─0d09a662-78d3-4202-b2be-843a0669fc9f
# ╠═cdebb87f-9759-4e6b-a00a-d764b3c7fbf8
# ╠═3ec28482-2d6c-4125-8d85-4fb46b130677
# ╠═8b283d5e-1ce7-4deb-b382-6fa8e5612ef1
# ╠═c14806bd-42cf-480b-b618-bfe72183feb3
# ╟─64dc2da0-48cc-4242-bb17-449a300688c7
# ╠═df2034fd-560d-4529-836a-13745f976c1f
# ╟─4a97576c-96de-458e-b881-3f5dd140fa6a
# ╠═a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9
# ╟─bfabcd80-fb62-410f-8710-f577852c77df
# ╠═f0486891-b8b3-4a39-91df-1389d6f799e1
# ╠═001c318e-b7a6-48a5-bfd5-6dd0368873ac
# ╠═05568880-f931-4394-b31e-922850203721
# ╠═5b073f11-08bd-4128-85c6-7f043b60bdb8
# ╟─9ea12ddb-ff8a-4c16-b2a5-8b7603f262a3
# ╠═9346f569-d5f9-43cd-9302-1ee64ef9a030
# ╟─b98200a8-bf73-42a2-a357-af56812d01c3
# ╠═ce6a7357-0dd7-4581-b786-92d1eddd114d
# ╠═7a3e92cb-51b6-455f-b6dd-ff90896a9ffb
# ╠═26885b24-df80-4fbf-9824-e175688f1322
# ╠═5f56cc7c-861e-41a4-b783-848623b94bf9
# ╟─c5e2a097-7433-4b06-9549-86ecdb0e5c12
# ╠═557b968f-b409-4bd9-8460-edef0ee8f2e6
# ╠═2b90fe00-a530-44a2-b70c-0eb6cce156a6
# ╠═88fa1b08-f0d4-4fcf-89c2-8a9f33710d4c
# ╠═dabf2a1f-ec78-4942-973f-4dbf9037ee7b
# ╠═38324c42-e5c7-4b59-8129-0e4c17ab5bf1
# ╠═08021ed6-ac31-4829-9f21-f046af73d5a3
# ╠═92c138d3-05f6-4a57-9b6b-08f41cb4ea55
# ╠═3d889727-ae6d-4fa0-98ae-d3ae73fb6a3c
# ╠═b5c6d43c-8252-4602-8232-b3d1b0bcee33
# ╠═025b33d9-7473-4a54-a3f1-787a8650f9e7
# ╠═f4a16e34-669e-4c93-bd83-e3622a747a3a
# ╟─2ada3ddd-b47a-46ff-abba-17cbb94041a2
# ╠═1a958c8e-b2eb-4ed2-a7d6-2f1a52c5ac7a
# ╠═080d6dd1-0aa6-4bee-bf50-20d8ca890145
