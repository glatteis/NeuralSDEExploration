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

# ╔═╡ 67cb574d-7bd6-40d9-9dc3-d57f4226cc83
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end

# ╔═╡ b6abba94-db07-4095-98c9-443e31832e7d
using Optimisers, StatsBase, Zygote, Lux, DifferentialEquations, ComponentArrays, ParameterSchedulers, Random, Distributed, ForwardDiff, LuxCore

# ╔═╡ d1440209-78f7-4a9a-9101-a06ad2534e5d
using NeuralSDEExploration, Plots, PlutoUI

# ╔═╡ db557c9a-24d6-4008-8225-4b8867ee93db
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
	end
end

# ╔═╡ 13ef3cd9-7f58-459e-a659-abc35b550326
begin
	if @isdefined PlutoRunner  # running inside Pluto
		md"Enable training $(@bind enabletraining CheckBox())"		
	else
		enabletraining = true
	end
end


# ╔═╡ 32be3e35-a529-4d16-8ba0-ec4e223ae401
md"""
Let's train a Neural SDE from a modified form of the simple zero-dimensional energy balance model. First, let's just instantiate the predefined model from the package...
"""

# ╔═╡ f74dd752-485b-4203-9d72-c56e55a3ef76
ebm = NeuralSDEExploration.ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, 0.14)

# ╔═╡ c799a418-d85e-4f9b-af7a-ed667fab21b6
println("Running on $(Threads.nthreads()) threads")

# ╔═╡ cc2418c2-c355-4291-b5d7-d9019787834f
md"Let's generate the data and plot a quick example:"

# ╔═╡ dd03f851-2e26-4850-a7d4-a64f154d2872
begin
	n = 1000
    datasize = 100
    tspan = (0.0, 5.0)
end

# ╔═╡ d052d6c0-2065-4ae1-acf7-fbe90ff1cb02
begin
	struct BroadcastLayer{T <: NamedTuple} <: 	LuxCore.AbstractExplicitContainerLayer{(:layers,)}
    layers::T
	end
	function BroadcastLayer(layers...)
	    for l in layers
	        if !iszero(LuxCore.statelength(l))
	            throw(ArgumentError("Stateful layer `$l` are not supported for `BroadcastLayer`."))
	        end
	    end
	    names = ntuple(i -> Symbol("layer_$i"), length(layers))
	    return BroadcastLayer(NamedTuple{names}(layers))
	end
	
	BroadcastLayer(; kwargs...) = BroadcastLayer(connection, (; kwargs...))
	
	function (m::BroadcastLayer)(x, ps, st::NamedTuple{names}) where {names}
		println(values(ps))
		println(ps)
		println(values(collect(ps)))
		println(values(m.layers))

	    results = (first ∘ Lux.apply).(values(m.layers), x, values(collect(ps)), values(st))
	    return results, st
	end
		
	Base.keys(m::BroadcastLayer) = Base.keys(getfield(m, :layers))
end

# ╔═╡ 5d020072-8a2e-438d-8e7a-330cca97964b
LuxCore.initialparameters(rng::Random.AbstractRNG, l::BroadcastLayer) = NamedTuple{keys(l.layers)}(LuxCore.initialparameters.((rng, ), (values(l.layers))),)

# ╔═╡ 1a5b7d12-d972-45cf-a874-27202eb97975
# ╠═╡ disabled = true
#=╠═╡
encoder = Lux.Chain(Lux.Recurrence(Lux.GRUCell(data_dims => context_size); return_sequence=true), BroadcastLayer(Lux.Dense(context_size => context_size)))
  ╠═╡ =#

# ╔═╡ c00a97bf-5e10-4168-8d58-f4f9270258ac
solution = NeuralSDEExploration.series(ebm, range(210.0e0, 320.0e0, n), tspan, datasize; seed=10)

# ╔═╡ 1502612c-1489-4abf-8a8b-5b2d03a68cb1
md"""
Let's also plot some example trajectories:
"""

# ╔═╡ 455263ef-2f94-4f3e-8401-f0da7fb3e493
plot(reduce(hcat, [solution[i].u for i in 1:10]))

# ╔═╡ f4651b27-135e-45f1-8647-64ab08c2e8e8
md"""
Let's normalize our data for training:
"""

# ╔═╡ aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
begin
    datamax = max([max(x.u...) for x in solution]...)
    datamin = min([min(x.u...) for x in solution]...)

    function normalize(x)
        return ((x - datamin) / (datamax - datamin))
    end
end

# ╔═╡ 9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
timeseries = [(t=ts.t, u=map(normalize, ts.u)) for ts in solution]

# ╔═╡ da11fb69-a8a1-456d-9ce7-63180ef27a83
md"""
We are going to build a simple latent SDE. Define a few constants...
"""

# ╔═╡ 22453ba0-81a7-43f6-bb80-91d3c991530d
context_size = 8 # The size of the context given to the posterior SDE.

# ╔═╡ d81ccb5f-de1c-4a01-93ce-3e7302caedc0
hidden_size = 64 # The hidden layer size for all ANNs.

# ╔═╡ b5721107-7cf5-4da3-b22a-552e3d56bcfa
latent_dims = 2 # Dimensions of the latent space.

# ╔═╡ 9767a8ea-bdda-43fc-b636-8681d150d29f
data_dims = 1 # Dimensions of our input data.

# ╔═╡ e97cd020-d180-4db3-8e19-3efc2529aad2
encoder = Lux.Recurrence(Lux.LSTMCell(data_dims => context_size); return_sequence=true)

# ╔═╡ 39154e76-01f7-4710-8f21-6f3a7d3fcfcd
md"""
The encoder takes a timeseries and outputs context that can be passed to the posterior SDE, that is, the SDE that has information about the data and encodes $p_\theta(z \mid x)$.
"""

# ╔═╡ 0d09a662-78d3-4202-b2be-843a0669fc9f
md"""
The `initial_posterior` net is the posterior for the initial state. It takes the context and outputs a mean and standard devation for the position zero of the posterior. The `initial_prior` is a fixed gaussian distribution.
"""

# ╔═╡ cdebb87f-9759-4e6b-a00a-d764b3c7fbf8
initial_posterior = Lux.Dense(context_size => latent_dims + latent_dims; init_weight=zeros)

# ╔═╡ 3ec28482-2d6c-4125-8d85-4fb46b130677
initial_prior = Lux.Dense(1 => latent_dims + latent_dims) 

# ╔═╡ 8b283d5e-1ce7-4deb-b382-6fa8e5612ef1
md"""
Drift of prior. This is just an SDE drift in the latent space
"""

# ╔═╡ c14806bd-42cf-480b-b618-bfe72183feb3
drift_prior = Lux.Chain(
	Lux.Dense(latent_dims => hidden_size, tanh),
	Lux.Dense(hidden_size => hidden_size, tanh),
	Lux.Dense(hidden_size => latent_dims, tanh),
	Lux.Scale(latent_dims)
)

# ╔═╡ 64dc2da0-48cc-4242-bb17-449a300688c7
md"""
Drift of posterior. This is the term of an SDE when fed with the context.
"""

# ╔═╡ df2034fd-560d-4529-836a-13745f976c1f
drift_posterior = Lux.Chain(
	Lux.Dense(latent_dims + context_size => hidden_size, tanh),
	Lux.Dense(hidden_size => hidden_size, tanh),
	Lux.Dense(hidden_size => latent_dims, tanh),
	Lux.Scale(latent_dims)
)

# ╔═╡ 4a97576c-96de-458e-b881-3f5dd140fa6a
md"""
Diffusion. Prior and posterior share the same diffusion (they are not actually evaluated seperately while training, only their KL divergence). This is a diagonal diffusion, i.e. every term in the latent space has its own independent Wiener process.
"""

# ╔═╡ a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9
diffusion = Lux.Parallel(nothing, [Lux.Chain(Lux.Dense(1 => 32, tanh), Lux.Dense(32 => 1, tanh, init_bias=ones), Lux.Scale(1, init_weight=ones)) for i in 1:latent_dims]...)

# ╔═╡ 08892d82-da8e-4fbd-be18-0e0b3a3fb498
# ╠═╡ disabled = true
#=╠═╡
diffusion = Lux.Parallel(nothing, [Lux.Scale(1; bias=true, init_weight=zeros, init_bias=ones) for i in 1:latent_dims]...)
  ╠═╡ =#

# ╔═╡ bfabcd80-fb62-410f-8710-f577852c77df
md"""
The projector will transform the latent space back into data space.
"""

# ╔═╡ f0486891-b8b3-4a39-91df-1389d6f799e1
projector = Lux.Dense(latent_dims => data_dims)

# ╔═╡ 001c318e-b7a6-48a5-bfd5-6dd0368873ac
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
	dt=(tspan[end]/datasize)
)

# ╔═╡ 0f6f4520-576f-42d3-9126-2076a51a6e22
rng = Xoshiro()

# ╔═╡ 05568880-f931-4394-b31e-922850203721
ps_, st = Lux.setup(rng, latent_sde)

# ╔═╡ e5c9f00f-849b-4523-85da-4dab567b2ca8
ps = ComponentArray(ps_)

# ╔═╡ 0667038f-4187-4044-9bec-941800fdba22
@bind i Slider(1:length(timeseries))

# ╔═╡ bdf4c32c-9a2b-4814-8513-c6e16ebee69c
plot(timeseries[i].u, legend=nothing)

# ╔═╡ f08369b8-6eb9-4bb2-a389-f054fbe72645
reshape(timeseries[i].u, 1, :, 1)

# ╔═╡ cb9b4f6d-1b0e-4ca7-97f1-db377a9afca3
plot(timeseries[i].u)

# ╔═╡ 0c339b92-664d-4fef-a57d-af6c69c65597
example_context, st_ = encoder(reshape(timeseries[i].u, 1, :, 1), ps.encoder, st.encoder)

# ╔═╡ e1fa0645-42e7-48a1-8066-8b564d4ad519
plot(reduce((x,y) -> cat(x, y; dims = 3), example_context)[1, :, :]')

# ╔═╡ 9ea12ddb-ff8a-4c16-b2a5-8b7603f262a3
md"""
Integrate the prior SDE in latent space to see what a run looks like:
"""

# ╔═╡ 9346f569-d5f9-43cd-9302-1ee64ef9a030
plot(NeuralSDEExploration.sample_prior(latent_sde, ps, st; b=5))

# ╔═╡ b98200a8-bf73-42a2-a357-af56812d01c3
md"""
Integrate the posterior SDE in latent space given context:
"""

# ╔═╡ 26885b24-df80-4fbf-9824-e175688f1322
@bind seed Slider(1:1000)

# ╔═╡ 5f56cc7c-861e-41a4-b783-848623b94bf9
@bind ti Slider(1:length(timeseries))

# ╔═╡ f5267a6b-ff1d-4dc1-9d5e-088328ecf617
timeseries[[1,501,901]]

# ╔═╡ 88fa1b08-f0d4-4fcf-89c2-8a9f33710d4c
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, timeseries[[1,501,901]], st; seed=seed, ensemblemode=EnsembleThreads())

# ╔═╡ 737a7c93-7da6-4ce8-8f41-1aa32df04568
posterior_latent[:, :, 2]

# ╔═╡ dabf2a1f-ec78-4942-973f-4dbf9037ee7b
plot(logterm_[1, :, :]', label="KL-Divergence")

# ╔═╡ 38324c42-e5c7-4b59-8129-0e4c17ab5bf1
plot(posterior_data[1, :, :]', label="posterior")

# ╔═╡ 08021ed6-ac31-4829-9f21-f046af73d5a3
plot(posterior_latent[:, 1, :]')

# ╔═╡ 3d889727-ae6d-4fa0-98ae-d3ae73fb6a3c
plot(NeuralSDEExploration.sample_prior(latent_sde, ps, st; seed=seed+1))

# ╔═╡ b5c6d43c-8252-4602-8232-b3d1b0bcee33
function plotmodel()
	posteriors = []
	priors = []
	posterior_latent = nothing
	datas = []
	n = 5
	rng = Xoshiro(0)
	nums = sample(rng,1:length(timeseries),n;replace=false)

	posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, timeseries[nums], st, seed=seed+i)
	
	priorsamples = 25
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

# ╔═╡ 025b33d9-7473-4a54-a3f1-787a8650f9e7
plotmodel()

# ╔═╡ 550d8974-cd19-4d0b-9492-adb4e14a04b1
begin
	recorded_loss = []
	recorded_likelihood = []
	recorded_kl = []
	recorded_eta = []
	recorded_lr = []
end

# ╔═╡ fa43f63d-8293-43cc-b099-3b69dbbf4b6a
function plotlearning()

	plots = [
		plot(recorded_loss, legend=false, title="loss", yaxis=:log)
		plot(recorded_likelihood, legend=false, title="likelihood", yaxis=:log)
		plot(recorded_kl, legend=false, title="kl-divergence", yaxis=:log)
		plot(recorded_eta, legend=false, title="eta", yaxis=:log)
	]	

	
	l = @layout [a b ; c d]
	plot(plots...; layout=l)
end

# ╔═╡ f0a34be1-6aa2-4563-abc2-ea163a778752
function loss(ps, minibatch, eta)
	_, _, _, kl_divergence, likelihood = NeuralSDEExploration.pass(latent_sde, ps, minibatch, st; ensemblemode=EnsembleThreads())
	return mean(-likelihood .+ (eta * kl_divergence)), mean(kl_divergence), mean(likelihood)
end

# ╔═╡ f4a16e34-669e-4c93-bd83-e3622a747a3a
function train(learning_rate, num_steps, ar)
	# sched = Loop(Sequence([Loop(x -> (25*x)/ar, ar), Loop(x -> 25.0, ar)], [ar, ar]), ar*2)
	sched = Loop(x -> 25.0, 1)

	opt_state = Optimisers.setup(Optimisers.Adam(), ps)
	for (step, eta) in zip(1:num_steps, sched)
		s = sample(rng, 1:size(timeseries)[1], 8, replace=false)
		minibatch = timeseries[s]

		l, kl_divergence, likelihood = loss(ps, minibatch, eta)
		push!(recorded_loss, l)
		push!(recorded_kl, kl_divergence)
		push!(recorded_likelihood, likelihood)
		push!(recorded_eta, eta)
		push!(recorded_lr, learning_rate)
		
		dps = Zygote.gradient(ps -> loss(ps, minibatch, eta)[1], ps)
		println("Loss: $l")
		Optimisers.update!(opt_state, ps, dps[1])
	end
end

# ╔═╡ 9789decf-c384-42df-b7aa-3c2137a69a41
function exportresults(epoch)
	show(IOContext(stdout, :limit=>false), MIME"text/plain"(), ps)
	
	name = "currtrain"
	if "SLURM_JOB_ID" in keys(ENV)
		name = ENV["SLURM_JOB_ID"]
	end
	modelfig = plotmodel()
	savefig(modelfig, "~/$(name)_$(epoch)_model.pdf")
	learningfig = plotlearning()
	savefig(learningfig, "~/$(name)_$(epoch)_learning.pdf")
end

# ╔═╡ 78aa72e2-8188-441f-9910-1bc5525fda7a
begin
	if !(@isdefined PlutoRunner) && enabletraining  # running as job
		for epoch in 1:100
			train(0.1 - (epoch/100)*0.1, 1000, 500)
			exportresults(epoch)
		end
	end
end

# ╔═╡ 08b216d2-e752-4f18-9c0a-e9c571218a53
dps = Zygote.gradient(ps -> loss(ps, timeseries[1:5], 1.0)[1], ps)[1]

# ╔═╡ 38716b5c-fe06-488c-b6ed-d2e28bd3d397
begin
	if enabletraining
		@gif for epoch in 1:10
			train(0.01, 1, 1)
			plotmodel()
		end
	end
end


# ╔═╡ 7ce64e25-e5a5-4ace-ba6b-844ef6e4ef82
show(IOContext(stdout, :limit=>false), MIME"text/plain"(), ps)

# ╔═╡ Cell order:
# ╠═67cb574d-7bd6-40d9-9dc3-d57f4226cc83
# ╠═db557c9a-24d6-4008-8225-4b8867ee93db
# ╠═b6abba94-db07-4095-98c9-443e31832e7d
# ╠═d1440209-78f7-4a9a-9101-a06ad2534e5d
# ╠═13ef3cd9-7f58-459e-a659-abc35b550326
# ╟─32be3e35-a529-4d16-8ba0-ec4e223ae401
# ╠═f74dd752-485b-4203-9d72-c56e55a3ef76
# ╠═c799a418-d85e-4f9b-af7a-ed667fab21b6
# ╟─cc2418c2-c355-4291-b5d7-d9019787834f
# ╠═dd03f851-2e26-4850-a7d4-a64f154d2872
# ╠═d052d6c0-2065-4ae1-acf7-fbe90ff1cb02
# ╠═5d020072-8a2e-438d-8e7a-330cca97964b
# ╠═1a5b7d12-d972-45cf-a874-27202eb97975
# ╠═e97cd020-d180-4db3-8e19-3efc2529aad2
# ╠═c00a97bf-5e10-4168-8d58-f4f9270258ac
# ╟─1502612c-1489-4abf-8a8b-5b2d03a68cb1
# ╠═455263ef-2f94-4f3e-8401-f0da7fb3e493
# ╟─f4651b27-135e-45f1-8647-64ab08c2e8e8
# ╠═aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
# ╠═9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
# ╟─da11fb69-a8a1-456d-9ce7-63180ef27a83
# ╠═22453ba0-81a7-43f6-bb80-91d3c991530d
# ╠═d81ccb5f-de1c-4a01-93ce-3e7302caedc0
# ╠═b5721107-7cf5-4da3-b22a-552e3d56bcfa
# ╠═9767a8ea-bdda-43fc-b636-8681d150d29f
# ╟─39154e76-01f7-4710-8f21-6f3a7d3fcfcd
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
# ╠═08892d82-da8e-4fbd-be18-0e0b3a3fb498
# ╟─bfabcd80-fb62-410f-8710-f577852c77df
# ╠═f0486891-b8b3-4a39-91df-1389d6f799e1
# ╠═001c318e-b7a6-48a5-bfd5-6dd0368873ac
# ╠═0f6f4520-576f-42d3-9126-2076a51a6e22
# ╠═05568880-f931-4394-b31e-922850203721
# ╠═e5c9f00f-849b-4523-85da-4dab567b2ca8
# ╠═0667038f-4187-4044-9bec-941800fdba22
# ╠═f08369b8-6eb9-4bb2-a389-f054fbe72645
# ╠═cb9b4f6d-1b0e-4ca7-97f1-db377a9afca3
# ╠═0c339b92-664d-4fef-a57d-af6c69c65597
# ╠═e1fa0645-42e7-48a1-8066-8b564d4ad519
# ╟─9ea12ddb-ff8a-4c16-b2a5-8b7603f262a3
# ╠═9346f569-d5f9-43cd-9302-1ee64ef9a030
# ╟─b98200a8-bf73-42a2-a357-af56812d01c3
# ╠═26885b24-df80-4fbf-9824-e175688f1322
# ╠═5f56cc7c-861e-41a4-b783-848623b94bf9
# ╠═f5267a6b-ff1d-4dc1-9d5e-088328ecf617
# ╠═88fa1b08-f0d4-4fcf-89c2-8a9f33710d4c
# ╠═737a7c93-7da6-4ce8-8f41-1aa32df04568
# ╠═dabf2a1f-ec78-4942-973f-4dbf9037ee7b
# ╠═38324c42-e5c7-4b59-8129-0e4c17ab5bf1
# ╠═08021ed6-ac31-4829-9f21-f046af73d5a3
# ╠═3d889727-ae6d-4fa0-98ae-d3ae73fb6a3c
# ╠═b5c6d43c-8252-4602-8232-b3d1b0bcee33
# ╠═025b33d9-7473-4a54-a3f1-787a8650f9e7
# ╠═550d8974-cd19-4d0b-9492-adb4e14a04b1
# ╠═fa43f63d-8293-43cc-b099-3b69dbbf4b6a
# ╠═f0a34be1-6aa2-4563-abc2-ea163a778752
# ╠═f4a16e34-669e-4c93-bd83-e3622a747a3a
# ╠═9789decf-c384-42df-b7aa-3c2137a69a41
# ╠═78aa72e2-8188-441f-9910-1bc5525fda7a
# ╠═08b216d2-e752-4f18-9c0a-e9c571218a53
# ╠═38716b5c-fe06-488c-b6ed-d2e28bd3d397
# ╠═7ce64e25-e5a5-4ace-ba6b-844ef6e4ef82
