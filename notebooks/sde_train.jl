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
using Optimisers, StatsBase, Zygote, ForwardDiff, Enzyme, Flux

# ╔═╡ d1440209-78f7-4a9a-9101-a06ad2534e5d
using NeuralSDEExploration, Plots, PlutoUI

# ╔═╡ cf890cbc-a4ba-418c-b245-817fbabf7797
using ComponentArrays

# ╔═╡ 72399050-91e6-4c5e-b5fd-af74de97b1e8
using Functors

# ╔═╡ e0d7d514-ff1e-4446-81ab-8559d65be448
import CairoMakie

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
    datasize = 300
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
        return (x - datamin) / (datamax - datamin)
    end

    function rescale(x)
        return (x * (datamax - datamin)) + datamin
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
hidden_size = 16 # The hidden layer size for all ANNs.

# ╔═╡ b5721107-7cf5-4da3-b22a-552e3d56bcfa
latent_dims = 2 # Dimensions of the latent space.

# ╔═╡ 9767a8ea-bdda-43fc-b636-8681d150d29f
data_dims = 1 # Dimensions of our input data.

# ╔═╡ 39154e76-01f7-4710-8f21-6f3a7d3fcfcd
md"""
The encoder takes a timeseries and outputs context that can be passed to the posterior SDE, that is, the SDE that has information about the data and encodes $p_\theta(z \mid x)$.
"""

# ╔═╡ 847587ec-9297-471e-b788-7b776c05851e
encoder = Flux.Chain(Flux.GRU(data_dims => 4), Flux.Dense(4 => context_size, tanh))

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
initial_posterior = Flux.Dense(context_size => latent_dims + latent_dims)

# ╔═╡ 3ec28482-2d6c-4125-8d85-4fb46b130677
initial_prior = Flux.Dense(0 => latent_dims + latent_dims, init=Flux.glorot_uniform)

# ╔═╡ 8b283d5e-1ce7-4deb-b382-6fa8e5612ef1
md"""
Drift of prior. This is just an SDE drift in the latent space
"""

# ╔═╡ c14806bd-42cf-480b-b618-bfe72183feb3
drift_prior = Flux.Chain(
	Flux.Dense(latent_dims => hidden_size, tanh),
	Flux.Dense(hidden_size => hidden_size, tanh),
	Flux.Dense(hidden_size => latent_dims, tanh),
	#Flux.Scale(latent_dims)
)

# ╔═╡ 64dc2da0-48cc-4242-bb17-449a300688c7
md"""
Drift of posterior. This is the term of an SDE when fed with the context.
"""

# ╔═╡ df2034fd-560d-4529-836a-13745f976c1f
drift_posterior = Flux.Chain(
	Flux.Dense(latent_dims + context_size => hidden_size, tanh),
	Flux.Dense(hidden_size => hidden_size, tanh),
	Flux.Dense(hidden_size => latent_dims, tanh),
	#Flux.Scale(latent_dims)
)

# ╔═╡ 4a97576c-96de-458e-b881-3f5dd140fa6a
md"""
Diffusion. Prior and posterior share the same diffusion (they are not actually evaluated seperately while training, only their KL divergence). This is a diagonal diffusion, i.e. every term in the latent space has its own independent Wiener process.
"""

# ╔═╡ a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9
#diffusion = Lux.Chain(Lux.Dense(latent_dims => hidden_size, tanh; bias=false), Lux.Dense(hidden_size => latent_dims, tanh; bias=false), Lux.Scale(latent_dims; bias=false))
diffusion = Flux.Chain(Flux.Scale(latent_dims, tanh; bias=false))

# ╔═╡ bfabcd80-fb62-410f-8710-f577852c77df
md"""
The projector will transform the latent space back into data space.
"""

# ╔═╡ f0486891-b8b3-4a39-91df-1389d6f799e1
projector = Flux.Dense(latent_dims => data_dims)

# ╔═╡ 001c318e-b7a6-48a5-bfd5-6dd0368873ac
latent_sde = LatentSDE(
	initial_prior,
	initial_posterior,
	drift_prior,
	drift_posterior,
	diffusion,
	encoder,
	projector,
    tspan;
	saveat=range(tspan[1], tspan[end], datasize),
	#abstol=1e-3,
	#reltol=1e-3
)

# ╔═╡ 9ea12ddb-ff8a-4c16-b2a5-8b7603f262a3
md"""
Integrate the prior SDE in latent space to see what a run looks like:
"""

# ╔═╡ 9346f569-d5f9-43cd-9302-1ee64ef9a030
plot(NeuralSDEExploration.sample_prior(latent_sde))

# ╔═╡ b98200a8-bf73-42a2-a357-af56812d01c3
md"""
Integrate the posterior SDE in latent space given context:
"""

# ╔═╡ ce6a7357-0dd7-4581-b786-92d1eddd114d
plot(NeuralSDEExploration.sample_posterior(latent_sde, timeseries[1]))

# ╔═╡ 7a3e92cb-51b6-455f-b6dd-ff90896a9ffb
CairoMakie.streamplot((x, y) -> CairoMakie.Point2(latent_sde.drift_prior_re(latent_sde.drift_prior_p)([Float32(x), Float32(y)])), (-4, 4), (-4, 4))

# ╔═╡ 26885b24-df80-4fbf-9824-e175688f1322
@bind seed Slider(1:1000)

# ╔═╡ 5f56cc7c-861e-41a4-b783-848623b94bf9
@bind ti Slider(1:length(timeseries))

# ╔═╡ 557b968f-b409-4bd9-8460-edef0ee8f2e6
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, timeseries[ti], seed=seed)

# ╔═╡ dabf2a1f-ec78-4942-973f-4dbf9037ee7b
plot(hcat(logterm_...)')

# ╔═╡ 38324c42-e5c7-4b59-8129-0e4c17ab5bf1
begin
	p = plot(hcat(map(only, posterior_data)), label="posterior")
	plot!(p, timeseries[1].u, label="data")
end

# ╔═╡ 08021ed6-ac31-4829-9f21-f046af73d5a3
plot(hcat(posterior_latent...)')

# ╔═╡ 92c138d3-05f6-4a57-9b6b-08f41cb4ea55
plot(NeuralSDEExploration.sample_posterior(latent_sde, timeseries[1], seed=seed))

# ╔═╡ 3d889727-ae6d-4fa0-98ae-d3ae73fb6a3c
plot(NeuralSDEExploration.sample_prior(latent_sde, seed=seed))

# ╔═╡ 49fa3b60-7488-4ec0-b394-702dbfd50d65
function cb()
	data = timeseries[1]
	p = plot(NeuralSDEExploration.sample_posterior(latent_sde, data, seed=1)[1], label="posterior")
	plot!(p, data.u, label="data")
end

# ╔═╡ d0c965c9-30ef-4233-8f72-19c69c3a92fd
ps_, re = Flux.destructure(latent_sde)

# ╔═╡ 0ce4fa52-ffed-4ee1-9134-a406fa9f4ae0
begin
	# import TruncatedStacktraces
	# TruncatedStacktraces.VERBOSE[] = true
end

# ╔═╡ 10f65b85-1a37-4890-8508-b8f33c325206
ps = ComponentArray(ps_)

# ╔═╡ 8be36057-42b3-4a0c-b312-d92c1c06522b
ComponentVector(Functors.functor(latent_sde)[1])

# ╔═╡ 0e979c84-7c36-4928-9bfb-005afc3f4e30
NeuralSDEExploration.loss(re(ps), timeseries[ti], seed=seed)

# ╔═╡ f4a16e34-669e-4c93-bd83-e3622a747a3a
function train(learning_rate, num_steps)
	opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)
	anim = @animate for step in 1:num_steps
		minibatch = timeseries[sample(1:size(timeseries)[1], 1, replace=false)]
		#if step % 10 == 0
		#cb()
		#end
		function loss(ps)
			sum(NeuralSDEExploration.loss(re(ps), ts) for ts in minibatch)
		end
		l = loss(ps)
		println("Loss: $l")
		grads = Zygote.gradient(loss, ps)[1]
		#grads = ForwardDiff.gradient(loss, ps)
		#println("Grads: $grads")
		Optimisers.update!(opt_state, ps, grads)
	end
	return gif(anim)
end

# ╔═╡ 660d714d-5cff-4890-b626-b3a6f8d1af9d
train(0.05, 10)

# ╔═╡ Cell order:
# ╠═67cb574d-7bd6-40d9-9dc3-d57f4226cc83
# ╠═b6abba94-db07-4095-98c9-443e31832e7d
# ╠═d1440209-78f7-4a9a-9101-a06ad2534e5d
# ╠═e0d7d514-ff1e-4446-81ab-8559d65be448
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
# ╟─8b283d5e-1ce7-4deb-b382-6fa8e5612ef1
# ╠═c14806bd-42cf-480b-b618-bfe72183feb3
# ╟─64dc2da0-48cc-4242-bb17-449a300688c7
# ╠═df2034fd-560d-4529-836a-13745f976c1f
# ╟─4a97576c-96de-458e-b881-3f5dd140fa6a
# ╠═a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9
# ╟─bfabcd80-fb62-410f-8710-f577852c77df
# ╟─f0486891-b8b3-4a39-91df-1389d6f799e1
# ╠═001c318e-b7a6-48a5-bfd5-6dd0368873ac
# ╟─9ea12ddb-ff8a-4c16-b2a5-8b7603f262a3
# ╠═9346f569-d5f9-43cd-9302-1ee64ef9a030
# ╟─b98200a8-bf73-42a2-a357-af56812d01c3
# ╠═ce6a7357-0dd7-4581-b786-92d1eddd114d
# ╠═7a3e92cb-51b6-455f-b6dd-ff90896a9ffb
# ╠═26885b24-df80-4fbf-9824-e175688f1322
# ╠═5f56cc7c-861e-41a4-b783-848623b94bf9
# ╠═557b968f-b409-4bd9-8460-edef0ee8f2e6
# ╠═dabf2a1f-ec78-4942-973f-4dbf9037ee7b
# ╠═38324c42-e5c7-4b59-8129-0e4c17ab5bf1
# ╠═08021ed6-ac31-4829-9f21-f046af73d5a3
# ╠═92c138d3-05f6-4a57-9b6b-08f41cb4ea55
# ╠═3d889727-ae6d-4fa0-98ae-d3ae73fb6a3c
# ╠═49fa3b60-7488-4ec0-b394-702dbfd50d65
# ╠═d0c965c9-30ef-4233-8f72-19c69c3a92fd
# ╠═0ce4fa52-ffed-4ee1-9134-a406fa9f4ae0
# ╠═cf890cbc-a4ba-418c-b245-817fbabf7797
# ╠═10f65b85-1a37-4890-8508-b8f33c325206
# ╠═72399050-91e6-4c5e-b5fd-af74de97b1e8
# ╠═8be36057-42b3-4a0c-b312-d92c1c06522b
# ╠═0e979c84-7c36-4928-9bfb-005afc3f4e30
# ╠═f4a16e34-669e-4c93-bd83-e3622a747a3a
# ╠═660d714d-5cff-4890-b626-b3a6f8d1af9d
