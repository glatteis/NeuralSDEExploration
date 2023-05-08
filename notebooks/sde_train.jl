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

# ╔═╡ 67cb574d-7bd6-40d9-9dc3-d57f4226cc83
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end

# ╔═╡ b6abba94-db07-4095-98c9-443e31832e7d
using Optimisers, StatsBase, Zygote, Lux, DifferentialEquations, ComponentArrays, ParameterSchedulers, Random, Distributed, ForwardDiff, LuxCore, Dates

# ╔═╡ d1440209-78f7-4a9a-9101-a06ad2534e5d
using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs

# ╔═╡ db557c9a-24d6-4008-8225-4b8867ee93db
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
	end
end

# ╔═╡ d38b3460-4c01-4bba-b726-150d207c020b
TableOfContents(title="Latent SDE")

# ╔═╡ 13ef3cd9-7f58-459e-a659-abc35b550326
begin
	if @isdefined PlutoRunner  # running inside Pluto
		md"Enable training (if in notebook) $(@bind enabletraining CheckBox())"		
	else
		enabletraining = true
	end
end


# ╔═╡ ff15555b-b1b5-4b42-94a9-da77daa546d0
md"""
# Model Definition
"""

# ╔═╡ 32be3e35-a529-4d16-8ba0-ec4e223ae401
md"""
Let's train a Neural SDE from a modified form of the simple zero-dimensional energy balance model. First, let's just instantiate the predefined model from the package...
"""

# ╔═╡ c799a418-d85e-4f9b-af7a-ed667fab21b6
println("Running on $(Threads.nthreads()) threads")

# ╔═╡ cc2418c2-c355-4291-b5d7-d9019787834f
md"Let's generate the data and plot a quick example:"

# ╔═╡ 0eec0598-7520-47ec-b13a-a7b9da550014
md"""
### Constants
"""

# ╔═╡ cb3a270e-0f2a-4be3-9ab3-ea5e4c56d0e7
md"""
Used model: $(@bind model_name Arg("model", Select(["sun", "fhn"]), short_name="m")), CLI arg: `--model`, `-m` (required!)
"""

# ╔═╡ 4c3b8784-368d-49c3-a875-c54960ec9be5
md"""
Number of timeseries in data: $(@bind n Arg("num-data", NumberField(1:1000000, default=5000), required=false)), CLI arg: `--num-data`
"""

# ╔═╡ a65a7405-d1de-4de5-9391-dcb971af0413
md"""
Length of timeseries: $(@bind datasize Arg("length", NumberField(1:1000, default=20), required=false)), CLI arg: `--length`
"""

# ╔═╡ 71a38a66-dd66-4000-b664-fc3e04f6d4b8
md"""
Timespan of simulations: $(@bind tspan_end Arg("tspan", NumberField(0.5:100.0; default=0.5), required=false)), CLI arg: `--tspan`
"""

# ╔═╡ fe7e2889-88de-49b3-b20b-342357596bfc
tspan = (0f0, Float32(tspan_end))

# ╔═╡ de70d89a-275d-49d2-9da4-4470c869e56e
tspan_sim = (-tspan_end, tspan_end)

# ╔═╡ 9a89a97c-da03-4887-ac8c-ef1f5264436e
println((num_data=n, length=datasize, tspan=tspan))

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

# ╔═╡ 7e6256ef-6a0a-40cc-aa0a-c467b3a524c4
md"""
### Data Generation
"""

# ╔═╡ 2da6bbd4-8036-471c-b94e-10182cf8a834
(initial_condition, model) = if model_name == "sun"
	(
		range(210f0, 350f0, n),
		NeuralSDEExploration.ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, 0.135)
	)
elseif model_name == "fhn"
	(
		[[0f0, 0f0] for i in 1:n],
		NeuralSDEExploration.FitzHughNagumoModel()
	)
else
	@error "Invalid model name!"
	nothing
end

# ╔═╡ c00a97bf-5e10-4168-8d58-f4f9270258ac
solution_full = NeuralSDEExploration.series(model, initial_condition, tspan_sim, datasize*2; seed=1)

# ╔═╡ 15cef7cc-30b6-499d-b968-775b3251dedb
solution = shuffle([(t=map(Float32, x.t[datasize:datasize*2-1]), u=map(Float32 ∘ first, x.u[datasize:datasize*2-1])) for x in solution_full])

# ╔═╡ 1502612c-1489-4abf-8a8b-5b2d03a68cb1
md"""
Let's also plot some example trajectories:
"""

# ╔═╡ 455263ef-2f94-4f3e-8401-f0da7fb3e493
plot(reduce(hcat, [solution[i].u for i in 1:100]); legend=false)

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

# ╔═╡ 8280424c-b86f-49f5-a854-91e7abcf13ec
md"""
# Neural Networks
"""

# ╔═╡ fdceee23-b91e-4b2e-af78-776c01733de3
md"""
### Constants
"""

# ╔═╡ 97d724c2-24b0-415c-b90f-6a36e877e9d1
md"""
The size of the context given to the posterior SDE:
$(@bind context_size Arg("context-size", NumberField(1:100, default=8), required=false)).
CLI arg: `--context-size`
"""

# ╔═╡ d81ccb5f-de1c-4a01-93ce-3e7302caedc0
md"""
The hidden layer size for the ANNs:
$(@bind hidden_size Arg("hidden-size", NumberField(1:1000000, default=128), required=false)).
CLI arg: `--hidden-size`
"""

# ╔═╡ b5721107-7cf5-4da3-b22a-552e3d56bcfa
md"""
Dimensions of the latent space:
$(@bind latent_dims Arg("latent-dims", NumberField(1:100, default=2), required=false)).
CLI arg: `--latent-dims`
"""

# ╔═╡ 86620e12-9631-4156-8b1c-60545b8a8352
if gpu
	using CUDA, DiffEqGPU
end

# ╔═╡ 16c12354-5ab6-4c0e-833d-265642119ed2
md"""
Batch size
$(@bind batch_size Arg("batch-size", NumberField(1:200, default=32), required=false)).
CLI arg: `--batch-size`
"""

# ╔═╡ f12633b6-c770-439d-939f-c41b74a5c309
md"""
Eta
$(@bind eta Arg("eta", NumberField(0.1:1000.0, default=50.0), required=false)).
CLI arg: `--eta`
"""

# ╔═╡ 3c630a3a-7714-41c7-8cc3-601cd6efbceb
md"""
Learning rate
$(@bind learning_rate Arg("learning-rate", NumberField(0.001:1000.0, default=0.03), required=false)).
CLI arg: `--learning-rate`
"""

# ╔═╡ 9767a8ea-bdda-43fc-b636-8681d150d29f
data_dims = length(solution[1].u[1]) # Dimensions of our input data.

# ╔═╡ db88cae4-cb25-4628-9298-5a694c4b29ef
println((context_size=context_size, hidden_size=hidden_size, latent_dims=latent_dims, data_dims=data_dims, stick_landing=stick_landing, batch_size=batch_size))

# ╔═╡ 0c0e5a95-195e-4057-bcba-f1d92d75cbab
md"""
### Encoder
"""

# ╔═╡ bec46289-4d61-4b90-bc30-0a86f174a599
md"""
The encoder takes a timeseries and outputs context that can be passed to the posterior SDE, that is, the SDE that has information about the data and encodes $p_\theta(z \mid x)$.
"""

# ╔═╡ 36a0fae8-c384-42fd-a6a0-159ea3664aa1
encoder = Lux.Recurrence(Lux.LSTMCell(data_dims => context_size); return_sequence=true)

# ╔═╡ 25558746-2baf-4f46-b21f-178c49106ed1
md"""
### Initial Values
"""

# ╔═╡ 0d09a662-78d3-4202-b2be-843a0669fc9f
md"""
The `initial_posterior` net is the posterior for the initial state. It takes the context and outputs a mean and standard devation for the position zero of the posterior. The `initial_prior` is a fixed gaussian distribution.
"""

# ╔═╡ cdebb87f-9759-4e6b-a00a-d764b3c7fbf8
initial_posterior = Lux.Dense(context_size => latent_dims + latent_dims; init_weight=zeros)

# ╔═╡ 3ec28482-2d6c-4125-8d85-4fb46b130677
initial_prior = Lux.Dense(1 => latent_dims + latent_dims) 

# ╔═╡ 8d2616e0-190c-4e98-92d0-18d5312569d6
md"""
### Latent Drifts
"""

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

# ╔═╡ 27ce1274-6731-4587-a603-83beee226476
md"""
### Shared Diffusion
"""

# ╔═╡ 4a97576c-96de-458e-b881-3f5dd140fa6a
md"""
Diffusion. Prior and posterior share the same diffusion (they are not actually evaluated seperately while training, only their KL divergence). This is a diagonal diffusion, i.e. every term in the latent space has its own independent Wiener process.
"""

# ╔═╡ a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9
diffusion = Lux.Parallel(nothing, [Lux.Chain(Lux.Dense(1 => hidden_size, Lux.tanh), Lux.Dense(hidden_size => 1, Lux.sigmoid_fast), Lux.Scale(1, init_weight=ones, init_bias=ones)) for i in 1:latent_dims]...)

# ╔═╡ b0421c4a-f243-4d39-8cca-a29ea140486d
md"""
### Projector
"""

# ╔═╡ bfabcd80-fb62-410f-8710-f577852c77df
md"""
The projector will transform the latent space back into data space.
"""

# ╔═╡ f0486891-b8b3-4a39-91df-1389d6f799e1
projector = Lux.Dense(latent_dims => data_dims)

# ╔═╡ b8b2f4b5-e90c-4066-8dad-27e8dfa1d7c5
md"""
### Latent SDE Model
"""

# ╔═╡ 001c318e-b7a6-48a5-bfd5-6dd0368873ac
latent_sde = LatentSDE(
	initial_prior,
	initial_posterior,
	drift_prior,
	drift_posterior,
	diffusion,
	encoder,
	projector,
	#DRI1(),
	EM(),
	tspan,
	datasize;

	weak_timeseries_errors=false,
    weak_dense_errors=false,
)

# ╔═╡ 0f6f4520-576f-42d3-9126-2076a51a6e22
rng = Xoshiro()

# ╔═╡ 1938e122-2c05-46fc-b179-db38322530ff
md"""
# Parameters
"""

# ╔═╡ 05568880-f931-4394-b31e-922850203721
ps_, st = Lux.setup(rng, latent_sde)

# ╔═╡ fd10820a-eb9b-4ff0-b66b-2d74ba4f1af3
md"""
### Load from file
"""

# ╔═╡ 421ca47e-2d28-4340-97d5-1a31582d4bed
md"""
Load parameters from file (random if no file selected): $(@bind ps_file FilePicker())
"""

# ╔═╡ dfe0f6ef-ecd5-46a1-a808-77ef9af44b56
ps = if ps_file !== nothing
	ps_data = String(ps_file["data"]);
	ComponentArray(eval(Meta.parse(ps_data)))
else
	ComponentArray{Float32}(ps_)
end

# ╔═╡ cbc85049-9563-4b5d-8d14-a171f4d0d6aa
md"""
# Example Simulations
"""

# ╔═╡ f5734b1a-4258-44ae-ac00-9520170997bc
md"""
### Select Examples
"""

# ╔═╡ 5f56cc7c-861e-41a4-b783-848623b94bf9
md"""
Select timeseries to do some simulations on: $(@bind ti RangeSlider(1:20; default=1:4))
"""

# ╔═╡ ee3d4a2e-0960-430e-921a-17d340af497c
md"""
Select a seed: $(@bind seed Scrubbable(481283))
"""

# ╔═╡ 34a3f397-287e-4f43-b76a-3efde5625f90
viz_batch = shuffle(Xoshiro(seed), timeseries)[ti]

# ╔═╡ cde87244-b6dd-4a1e-8114-5a130fabbe0a
tsmatrix_batch = reduce(hcat, [reshape(ts.u, 1, 1, :) for ts in viz_batch]);

# ╔═╡ a40ba796-e959-4402-8851-7334e13fc90f
md"""
### Timeseries and Context
"""

# ╔═╡ 77b41b7d-deee-4eca-8ead-f7af03ece23d
plot(viz_batch, title="Timeseries", label=false)

# ╔═╡ cee607cc-79fb-4aed-9d91-83d3ff683cb5
example_context, st_ = encoder(permutedims(tsmatrix_batch, (1, 3, 2)), ps.encoder, st.encoder)

# ╔═╡ 3c163505-22f7-4738-af96-369b30436dd7
plot(reduce((x,y) -> cat(x, y; dims = 3), example_context)[1, :, :]', title="Context")

# ╔═╡ 149f32fd-62bc-444d-a771-5f7e27435c73
md"""
### Latent SDE Pass
"""

# ╔═╡ 88fa1b08-f0d4-4fcf-89c2-8a9f33710d4c
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, viz_batch, st; seed=seed, ensemblemode=EnsembleSerial())

# ╔═╡ dabf2a1f-ec78-4942-973f-4dbf9037ee7b
plot(logterm_[1, :, :]', title="KL-Divergence")

# ╔═╡ 38324c42-e5c7-4b59-8129-0e4c17ab5bf1
plot(posterior_data[1, :, :]', label="posterior")

# ╔═╡ 08021ed6-ac31-4829-9f21-f046af73d5a3
plot(posterior_latent[:, 1, :]')

# ╔═╡ 3d889727-ae6d-4fa0-98ae-d3ae73fb6a3c
plot(NeuralSDEExploration.sample_prior(latent_sde, ps, st; seed=seed+1))

# ╔═╡ 590a0541-e6bf-4bd5-9bf5-1ef9931e60fb
md"""
### Simulation of Prior
"""

# ╔═╡ 2827fe3a-3ecd-4662-a2d3-c980f1e1cd84
[x for x in NeuralSDEExploration.sample_prior(latent_sde, ps, st; b=5).u[3]]

# ╔═╡ b5c6d43c-8252-4602-8232-b3d1b0bcee33
function plotmodel()
	posteriors = []
	priors = []
	posterior_latent = nothing
	datas = []
	n = 5
	rng = Xoshiro(0)
	nums = sample(rng,1:length(timeseries),n;replace=false)

	posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, timeseries[nums], st, seed=seed)
	
	priorsamples = 25
	prior_latent = NeuralSDEExploration.sample_prior(latent_sde,ps,st;seed=abs(rand(rng, Int)),b=priorsamples)
	projected_prior = reduce(hcat, [reduce(vcat, [latent_sde.projector(u, ps.projector, st.projector)[1] for u in batch.u]) for batch in prior_latent.u])

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

# ╔═╡ 225791b1-0ffc-48e2-8131-7f54848d8d83
md"""
# Training
"""

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
		plot(recorded_loss, legend=false, title="loss")
		plot(recorded_likelihood, legend=false, title="likelihood")
		plot(recorded_kl, legend=false, title="kl-divergence")
		plot(recorded_eta, legend=false, title="eta")
	]	

	
	l = @layout [a b ; c d]
	plot(plots...; layout=l)
end

# ╔═╡ f943cd4b-ab94-40c0-bf59-3db777eee928
ensemblemode = if gpu
	EnsembleGPUKernel(0.0)
else
	EnsembleThreads()
end

# ╔═╡ f0a34be1-6aa2-4563-abc2-ea163a778752
function loss(ps, minibatch, eta)
	_, _, _, kl_divergence, likelihood = NeuralSDEExploration.pass(latent_sde, ps, minibatch, st; ensemblemode=EnsembleThreads(), stick_landing=stick_landing, seed=abs(rand(rng, Int)))
	return mean(-likelihood .+ (eta * kl_divergence)), mean(kl_divergence), mean(likelihood)
end

# ╔═╡ f4a16e34-669e-4c93-bd83-e3622a747a3a
function train(learning_rate, num_steps, opt_state, ar=1)
	# sched = Loop(Sequence([Loop(x -> (25*x)/ar, ar), Loop(x -> 25.0, ar)], [ar, ar]), ar*2)
	sched = Loop(x -> eta, 1)

	for (step, eta) in zip(1:num_steps, sched)
		s = sample(rng, 1:size(timeseries)[1], batch_size, replace=false)
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
	folder_name = if "SLURM_JOB_ID" in keys(ENV)
		ENV["SLURM_JOB_ID"]
	else
		Dates.format(now(), "YMMddHHmm")
	end

	folder = homedir() * "/artifacts/$(folder_name)/"

	mkpath(folder)

	write(folder * "$(epoch)_params.txt", "$ps")
	
	modelfig = plotmodel()
	savefig(modelfig, folder * "$(epoch)_model.pdf")

	learningfig = plotlearning()
	savefig(learningfig, folder * "$(epoch)_learning.pdf")
end

# ╔═╡ 7a7e8e9b-ca89-4826-8a5c-fe51d96152ad
if enabletraining
	dps = Zygote.gradient(ps -> loss(ps, timeseries[1:batch_size], 1.0)[1], ps)[1]
end

# ╔═╡ 78aa72e2-8188-441f-9910-1bc5525fda7a
begin
	if !(@isdefined PlutoRunner) && enabletraining  # running as job
		opt_state_job = Optimisers.setup(Optimisers.Adam(), ps)
		for epoch in 1:100
			train(learning_rate, 250, opt_state_job)
			exportresults(epoch)
		end
	end
end

# ╔═╡ 830f7e7a-71d0-43c8-8e74-d1709b8a6707
function gifplot()
	p1 = plotmodel()
	p2 = plotlearning()
	plot(p1, p2, layout=@layout[a;b],size=(600,700))
end

# ╔═╡ 763f07e6-dd46-42d6-b57a-8f1994386302
gifplot()

# ╔═╡ 38716b5c-fe06-488c-b6ed-d2e28bd3d397
begin
	if enabletraining
		opt_state = Optimisers.setup(Optimisers.Adam(), ps)

		@gif for epoch in 1:10
			train(learning_rate, 40, opt_state)
			gifplot()
		end
	end
end


# ╔═╡ 8880282e-1b5a-4c85-95ef-699ccf8d4203
md"""
# Statistical Analysis
"""

# ╔═╡ 47b2ec07-40f4-480d-b650-fbf1b44b7527
begin
	prior_latent = NeuralSDEExploration.sample_prior(latent_sde,ps,st;b=n)
	projected_prior = vcat(reduce(vcat, [latent_sde.projector(x[10:end], ps.projector, st.projector)[1] for x in prior_latent.u])...)
	plot(fit(Histogram, projected_prior, 0.0:0.01:1.0), xlims=(0.0,1.0))
end

# ╔═╡ 14f9a62d-9caa-40e9-8502-d2a27b9c950e
begin
	ts = vcat(reduce(vcat, [s.u[10:end] for s in timeseries])...)
	plot(fit(Histogram, ts, 0.0:0.01:1.0), xlims=(0.0,1.0))
end

# ╔═╡ fe749caf-393f-45b0-98e5-5d10c1821a9d
md"""
Stick landing: $(@bind stick_landing Arg("stick-landing", CheckBox()), required=false)
CLI arg: `--stick-landing`
"""

# ╔═╡ 60b5397d-7350-460b-9117-319dc127cc7e
md"""
Use GPU: $(@bind gpu Arg("gpu", CheckBox()), required=false)
CLI arg: `--gpu`
"""

# ╔═╡ Cell order:
# ╠═67cb574d-7bd6-40d9-9dc3-d57f4226cc83
# ╠═db557c9a-24d6-4008-8225-4b8867ee93db
# ╠═b6abba94-db07-4095-98c9-443e31832e7d
# ╠═d1440209-78f7-4a9a-9101-a06ad2534e5d
# ╟─d38b3460-4c01-4bba-b726-150d207c020b
# ╟─13ef3cd9-7f58-459e-a659-abc35b550326
# ╟─ff15555b-b1b5-4b42-94a9-da77daa546d0
# ╟─32be3e35-a529-4d16-8ba0-ec4e223ae401
# ╟─c799a418-d85e-4f9b-af7a-ed667fab21b6
# ╟─cc2418c2-c355-4291-b5d7-d9019787834f
# ╟─0eec0598-7520-47ec-b13a-a7b9da550014
# ╟─cb3a270e-0f2a-4be3-9ab3-ea5e4c56d0e7
# ╟─4c3b8784-368d-49c3-a875-c54960ec9be5
# ╟─a65a7405-d1de-4de5-9391-dcb971af0413
# ╟─71a38a66-dd66-4000-b664-fc3e04f6d4b8
# ╟─fe7e2889-88de-49b3-b20b-342357596bfc
# ╟─de70d89a-275d-49d2-9da4-4470c869e56e
# ╟─9a89a97c-da03-4887-ac8c-ef1f5264436e
# ╟─d052d6c0-2065-4ae1-acf7-fbe90ff1cb02
# ╟─5d020072-8a2e-438d-8e7a-330cca97964b
# ╟─7e6256ef-6a0a-40cc-aa0a-c467b3a524c4
# ╠═2da6bbd4-8036-471c-b94e-10182cf8a834
# ╠═c00a97bf-5e10-4168-8d58-f4f9270258ac
# ╠═15cef7cc-30b6-499d-b968-775b3251dedb
# ╟─1502612c-1489-4abf-8a8b-5b2d03a68cb1
# ╠═455263ef-2f94-4f3e-8401-f0da7fb3e493
# ╟─f4651b27-135e-45f1-8647-64ab08c2e8e8
# ╠═aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
# ╠═9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
# ╟─da11fb69-a8a1-456d-9ce7-63180ef27a83
# ╟─8280424c-b86f-49f5-a854-91e7abcf13ec
# ╟─fdceee23-b91e-4b2e-af78-776c01733de3
# ╟─97d724c2-24b0-415c-b90f-6a36e877e9d1
# ╟─d81ccb5f-de1c-4a01-93ce-3e7302caedc0
# ╟─b5721107-7cf5-4da3-b22a-552e3d56bcfa
# ╟─fe749caf-393f-45b0-98e5-5d10c1821a9d
# ╟─60b5397d-7350-460b-9117-319dc127cc7e
# ╟─16c12354-5ab6-4c0e-833d-265642119ed2
# ╟─f12633b6-c770-439d-939f-c41b74a5c309
# ╟─3c630a3a-7714-41c7-8cc3-601cd6efbceb
# ╟─9767a8ea-bdda-43fc-b636-8681d150d29f
# ╟─db88cae4-cb25-4628-9298-5a694c4b29ef
# ╟─86620e12-9631-4156-8b1c-60545b8a8352
# ╟─0c0e5a95-195e-4057-bcba-f1d92d75cbab
# ╟─bec46289-4d61-4b90-bc30-0a86f174a599
# ╠═36a0fae8-c384-42fd-a6a0-159ea3664aa1
# ╟─25558746-2baf-4f46-b21f-178c49106ed1
# ╟─0d09a662-78d3-4202-b2be-843a0669fc9f
# ╠═cdebb87f-9759-4e6b-a00a-d764b3c7fbf8
# ╠═3ec28482-2d6c-4125-8d85-4fb46b130677
# ╟─8d2616e0-190c-4e98-92d0-18d5312569d6
# ╟─8b283d5e-1ce7-4deb-b382-6fa8e5612ef1
# ╠═c14806bd-42cf-480b-b618-bfe72183feb3
# ╟─64dc2da0-48cc-4242-bb17-449a300688c7
# ╠═df2034fd-560d-4529-836a-13745f976c1f
# ╟─27ce1274-6731-4587-a603-83beee226476
# ╟─4a97576c-96de-458e-b881-3f5dd140fa6a
# ╠═a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9
# ╟─b0421c4a-f243-4d39-8cca-a29ea140486d
# ╟─bfabcd80-fb62-410f-8710-f577852c77df
# ╠═f0486891-b8b3-4a39-91df-1389d6f799e1
# ╟─b8b2f4b5-e90c-4066-8dad-27e8dfa1d7c5
# ╠═001c318e-b7a6-48a5-bfd5-6dd0368873ac
# ╠═0f6f4520-576f-42d3-9126-2076a51a6e22
# ╟─1938e122-2c05-46fc-b179-db38322530ff
# ╠═05568880-f931-4394-b31e-922850203721
# ╟─fd10820a-eb9b-4ff0-b66b-2d74ba4f1af3
# ╟─421ca47e-2d28-4340-97d5-1a31582d4bed
# ╟─dfe0f6ef-ecd5-46a1-a808-77ef9af44b56
# ╟─cbc85049-9563-4b5d-8d14-a171f4d0d6aa
# ╟─f5734b1a-4258-44ae-ac00-9520170997bc
# ╟─5f56cc7c-861e-41a4-b783-848623b94bf9
# ╟─ee3d4a2e-0960-430e-921a-17d340af497c
# ╟─34a3f397-287e-4f43-b76a-3efde5625f90
# ╟─cde87244-b6dd-4a1e-8114-5a130fabbe0a
# ╟─a40ba796-e959-4402-8851-7334e13fc90f
# ╟─77b41b7d-deee-4eca-8ead-f7af03ece23d
# ╠═cee607cc-79fb-4aed-9d91-83d3ff683cb5
# ╟─3c163505-22f7-4738-af96-369b30436dd7
# ╟─149f32fd-62bc-444d-a771-5f7e27435c73
# ╠═88fa1b08-f0d4-4fcf-89c2-8a9f33710d4c
# ╟─dabf2a1f-ec78-4942-973f-4dbf9037ee7b
# ╠═38324c42-e5c7-4b59-8129-0e4c17ab5bf1
# ╠═08021ed6-ac31-4829-9f21-f046af73d5a3
# ╠═3d889727-ae6d-4fa0-98ae-d3ae73fb6a3c
# ╟─590a0541-e6bf-4bd5-9bf5-1ef9931e60fb
# ╠═2827fe3a-3ecd-4662-a2d3-c980f1e1cd84
# ╟─b5c6d43c-8252-4602-8232-b3d1b0bcee33
# ╠═025b33d9-7473-4a54-a3f1-787a8650f9e7
# ╟─225791b1-0ffc-48e2-8131-7f54848d8d83
# ╠═550d8974-cd19-4d0b-9492-adb4e14a04b1
# ╠═fa43f63d-8293-43cc-b099-3b69dbbf4b6a
# ╠═f943cd4b-ab94-40c0-bf59-3db777eee928
# ╠═f0a34be1-6aa2-4563-abc2-ea163a778752
# ╠═f4a16e34-669e-4c93-bd83-e3622a747a3a
# ╠═9789decf-c384-42df-b7aa-3c2137a69a41
# ╠═7a7e8e9b-ca89-4826-8a5c-fe51d96152ad
# ╠═78aa72e2-8188-441f-9910-1bc5525fda7a
# ╠═830f7e7a-71d0-43c8-8e74-d1709b8a6707
# ╠═763f07e6-dd46-42d6-b57a-8f1994386302
# ╠═38716b5c-fe06-488c-b6ed-d2e28bd3d397
# ╟─8880282e-1b5a-4c85-95ef-699ccf8d4203
# ╟─47b2ec07-40f4-480d-b650-fbf1b44b7527
# ╟─14f9a62d-9caa-40e9-8502-d2a27b9c950e
