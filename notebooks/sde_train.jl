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
using Optimisers, StatsBase, Zygote, Lux, DifferentialEquations, ComponentArrays, ParameterSchedulers, Random, Distributed, ForwardDiff, LuxCore, Dates, JLD2, SciMLSensitivity, JLD2, Random123, Distributions, DiffEqBase

# ╔═╡ d1440209-78f7-4a9a-9101-a06ad2534e5d
using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs

# ╔═╡ db557c9a-24d6-4008-8225-4b8867ee93db
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
		println(join(ARGS, " "))
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
Used model: $(@bind model_name Arg("model", Select(["sun", "fhn", "ou", "ouli"]), short_name="m")), CLI arg: `--model`, `-m` (required!)
"""

# ╔═╡ 95bdd676-d8df-4fef-bdd7-cce85b717018
md"""
Noise term size: $(@bind noise_term Arg("noise", NumberField(0.0:1.0, 0.135), required=false)), CLI arg: `--noise`
"""

# ╔═╡ 4c3b8784-368d-49c3-a875-c54960ec9be5
md"""
Number of timeseries in data: $(@bind n Arg("num-data", NumberField(1:1000000, default=5000), required=false)), CLI arg: `--num-data`
"""

# ╔═╡ a65a7405-d1de-4de5-9391-dcb971ae0413
md"""
Timestep size: $(@bind dt Arg("dt", NumberField(0.0:1.0, 0.05), required=false)), CLI arg: `--dt`
"""

# ╔═╡ e6a71aae-9d81-45a9-af9a-c4188dda2787
md"""
Timespan start of data generation: $(@bind tspan_start_data Arg("tspan-start-data", NumberField(0e0:100.0e0, 0e0), required=false)), CLI arg: `--tspan-start-data`
"""

# ╔═╡ 71a38a66-dd66-4000-b664-fc3e04f6d4b8
md"""
Timespan end of data generation: $(@bind tspan_end_data Arg("tspan-end-data", NumberField(0.5e0:100e0, 1e0), required=false)), CLI arg: `--tspan-end-data`
"""

# ╔═╡ bae92e09-2e87-4a1e-aa2e-906f33985f6d
md"""
Timespan start of training data: $(@bind tspan_start_train Arg("tspan-start-train", NumberField(0.5e0:100e0, tspan_start_data), required=false)), CLI arg: `--tspan-start-train`
"""

# ╔═╡ bd7acf1a-c09a-4531-ad2c-b5e7e28af382
md"""
Timespan end of training data: $(@bind tspan_end_train Arg("tspan-end-train", NumberField(0.5e0:100e0, tspan_end_data), required=false)), CLI arg: `--tspan-end-train`
"""

# ╔═╡ 42ece6c1-9e8a-45e7-adf4-6f353da6a4e5
md"""
Timespan start of model: $(@bind tspan_start_model Arg("tspan-start-model", NumberField(0.5e0:100.0e0, tspan_start_train), required=false)), CLI arg: `--tspan-start-model`
"""

# ╔═╡ 3665efa6-6527-4771-82fd-285c3c0f8b41
md"""
Timespan end of model: $(@bind tspan_end_model Arg("tspan-end-model", NumberField(0.5e0:100e0, tspan_end_train), required=false)), CLI arg: `--tspan-end-model`
"""

# ╔═╡ 10c206ef-2321-4c64-bdf4-7f4e9934d911
md"""
Likelihood scale
$(@bind scale Arg("scale", NumberField(0.001:1.0, 0.01), required=false)).
CLI arg: `--scale`
"""

# ╔═╡ fe7e2889-88de-49b3-b20b-342357596bfc
tspan_train = (Float64(tspan_start_train), Float64(tspan_end_train))

# ╔═╡ de70d89a-275d-49d2-9da4-4470c869e56e
tspan_data = (Float64(tspan_start_data), Float64(tspan_end_data))

# ╔═╡ 986c442a-d02e-42d4-bda4-f66a1c92f799
tspan_model = (Float64(tspan_start_model), Float64(tspan_end_model))

# ╔═╡ 9a89a97c-da03-4887-ac8c-ef1f5264436e
println((num_data=n, dt=dt, tspan_train=tspan_train, tspan_data=tspan_data, tspan_model=tspan_model))

# ╔═╡ c441712f-e4b2-4f4a-83e1-aad558685288
function steps(tspan, dt)
	return Int(ceil((tspan[2] - tspan[1]) / dt))
end

# ╔═╡ 7e6256ef-6a0a-40cc-aa0a-c467b3a524c4
md"""
### Data Generation
"""

# ╔═╡ 2da6bbd4-8036-471c-b94e-10182cf8a834
(initial_condition, model) = if model_name == "sun"
	(
		[only(rand(Normal(260e0, 50e0), 1)) for i in 1:n],
		NeuralSDEExploration.ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, noise_term)
	)
elseif model_name == "fhn"
	(
		[[only(rand(Normal(0e0, 2e0), 1)), only(rand(Normal(0e0, 0.1e0), 1))] for i in 1:n],
		NeuralSDEExploration.FitzHughNagumoModelGamma()
	)
elseif model_name == "ou"
	(
		[only(rand(Normal(0e0, 1e0), 1)) for i in 1:n],
		NeuralSDEExploration.OrnsteinUhlenbeck()
	)
elseif model_name == "ouli"
	(
		[0e0 for i in 1:n],
		NeuralSDEExploration.OrnsteinUhlenbeck(0.0, 1.0, 0.5)
	)
else
	@error "Invalid model name!"
	nothing
end

# ╔═╡ c00a97bf-5e10-4168-8d58-f4f9270258ac
solution_full = NeuralSDEExploration.series(model, initial_condition, tspan_data, steps(tspan_data, dt); seed=1)

# ╔═╡ 5691fcc5-29b3-4236-9154-59c6fede49ce
tspan_train_steps = (searchsortedlast(solution_full.t, tspan_start_train)): (searchsortedlast(solution_full.t, tspan_end_train))

# ╔═╡ 15cef7cc-30b6-499d-b968-775b3251dedb
solution = Timeseries(shuffle([(t=map(Float64, solution_full.t[tspan_train_steps]), u=map(Float64 ∘ first, x[tspan_train_steps])) for x in solution_full.u]))

# ╔═╡ 1502612c-1489-4abf-8a8b-5b2d03a68cb1
md"""
Let's also plot some example trajectories:
"""

# ╔═╡ 455263ef-2f94-4f3e-8401-e0da7fb3e493
plot(select_ts(1:40, solution_full))

# ╔═╡ f4651b27-135e-45f1-8647-64ab08c2e8e8
md"""
Let's normalize our data for training:
"""

# ╔═╡ aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
begin
    datamax = max([max(x...) for x in solution.u]...) |> only
    datamin = min([min(x...) for x in solution.u]...) |> only

    function normalize(x)
        return ((x - datamin) / (datamax - datamin))
    end
end

# ╔═╡ 5d78e254-4134-4c2a-8092-03f6df7d5092
println((datamin=datamin, datamax=datamax))

# ╔═╡ c79a3a3a-5599-4585-83a4-c7b6bc017436
function corrupt(value)
	value + only(rand(Normal{Float64}(0e0, Float64(scale)), 1))
end

# ╔═╡ 9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
timeseries = map_dims(x -> map(normalize, x), solution)

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
$(@bind context_size Arg("context-size", NumberField(1:100, 2), required=false)).
CLI arg: `--context-size`
"""

# ╔═╡ d81ccb5f-de1c-4a01-93ce-3e7302caedc0
md"""
The hidden layer size for the ANNs:
$(@bind hidden_size Arg("hidden-size", NumberField(1:1000000, 64), required=false)).
CLI arg: `--hidden-size`
"""

# ╔═╡ 6489b190-e08f-466c-93c4-92a723f8e594
md"""
The hidden layer size for the prior:
$(@bind prior_size Arg("prior-size", NumberField(1:1000000, hidden_size), required=false)).
CLI arg: `--prior-size`
"""

# ╔═╡ b5721107-7cf5-4da3-b22a-552e3d56bcfa
md"""
Dimensions of the latent space:
$(@bind latent_dims Arg("latent-dims", NumberField(1:100, 2), required=false)).
CLI arg: `--latent-dims`
"""

# ╔═╡ efd438bc-13cc-457f-82c1-c6e0711079b3
md"""
Neural network depth:
$(@bind depth Arg("depth", NumberField(1:100, 2), required=false)).
CLI arg: `--depth`
"""

# ╔═╡ 9382314d-c076-4b95-8171-71d903bb9271
md"""
Time dependence: $(@bind time_dependence Arg("time-dep", CheckBox(), required=false))
CLI arg: `--time-dep`
"""

# ╔═╡ 03a21651-9c95-49e8-bb07-b03640f7e5b7
md"""
Fix projector to just use first dimension: $(@bind fixed_projector Arg("fixed-projector", CheckBox(), required=false))
CLI arg: `--fixed-projector`
"""

# ╔═╡ ad6247f6-6cb9-4a57-92d3-6328cbd84ecd
in_dims = latent_dims + (time_dependence ? 1 : 0)

# ╔═╡ 60b5397d-7350-460b-9117-319dc127cc7e
md"""
Use GPU: $(@bind gpu Arg("gpu", CheckBox(), required=false))
CLI arg: `--gpu`
"""

# ╔═╡ 16c12354-5ab6-4c0e-833d-265642119ed2
md"""
Batch size
$(@bind batch_size Arg("batch-size", NumberField(1:200, 128), required=false)).
CLI arg: `--batch-size`
"""

# ╔═╡ f12633b6-c770-439d-939f-c41b74a5c309
md"""
Eta
$(@bind eta Arg("eta", NumberField(0.1:1000.0, 1.0), required=false)).
CLI arg: `--eta`
"""

# ╔═╡ 3c630a3a-7714-41c7-8cc3-601cd6efbceb
md"""
Learning rate
$(@bind learning_rate Arg("learning-rate", NumberField(0.0001:1000.0, 0.03), required=false)).
CLI arg: `--learning-rate`
"""

# ╔═╡ 2961879e-cb52-4980-931b-6f8de1f26fa4
md"""
Max learning rate
$(@bind max_learning_rate Arg("max-learning-rate", NumberField(0.0001:1000.0, 2*learning_rate), required=false)).
CLI arg: `--max-learning-rate`
"""

# ╔═╡ 7c23c32f-97bc-4c8d-ac54-42753be61345
md"""
Learning rate decay
$(@bind decay Arg("decay", NumberField(0.0001:2.0, 0.999), required=false)).
CLI arg: `--decay`
"""

# ╔═╡ 64e7bba4-fb17-4ed8-851f-de9204e0f42d
md"""
LR Cycling Enabled: $(@bind lr_cycle Arg("lr-cycle", CheckBox(), required=false))
CLI arg: `--lr-cycle`
"""

# ╔═╡ 33d53264-3c8f-4f63-9dd2-46ebd00f4e28
md"""
LR oscillation time
$(@bind lr_rate Arg("lr-rate", NumberField(1:100000, 500), required=false)).
CLI arg: `--lr-rate`
"""

# ╔═╡ 8bb3084f-5fde-413e-b0fe-8b2e19673fae
md"""
KL Annealing Enabled: $(@bind kl_anneal Arg("kl-anneal", CheckBox(), required=false))
CLI arg: `--kl-anneal`
"""

# ╔═╡ 2c64b173-d4ad-477d-afde-5f3916e922ef
md"""
KL Annealing oscillation time
$(@bind kl_rate Arg("kl-rate", NumberField(1:100000, 1000), required=false)).
CLI arg: `--kl-rate`
"""

# ╔═╡ 9767a8ea-bdda-43fc-b636-8681d150d29f
data_dims = length(solution.u[1][1]) # Dimensions of our input data.

# ╔═╡ 3db229e0-0e13-4d80-8680-58b89161db35
md"""
Use backsolve: $(@bind backsolve Arg("backsolve", CheckBox(true), required=false))
CLI arg: `--backsolve`
"""

# ╔═╡ cb1c2b2e-a2a2-45ed-9fc1-655d28f267d1
md"""
Brownian tree cache depth: $(@bind tree_depth Arg("tree-depth", NumberField(1:100, 2), required=false))
CLI arg: `--tree-depth`
"""

# ╔═╡ 7f219c33-b37b-480a-9d21-9ea8d898d5d5
md"""
Use Kidger's initial state trick: $(@bind kidger_trick Arg("kidger", CheckBox(false), required=false))
CLI arg: `--kidger`
"""

# ╔═╡ 2bb433bb-17df-4a34-9ccf-58c0cf8b4dd3
(sense, noise) = if backsolve
	(
		BacksolveAdjoint(autojacvec=ZygoteVJP(), checkpointing=true),
		function(seed)
			rng_tree = Xoshiro(seed)
			VirtualBrownianTree(-3e0, fill(0e0, latent_dims + 1), tend=tspan_model[2]*2e0; tree_depth=tree_depth, rng=Threefry4x((rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32))))
		end,
	)
else
	(
		InterpolatingAdjoint(autojacvec=ZygoteVJP()),
		(seed) -> nothing,
	)
end

# ╔═╡ db88cae4-cb25-4628-9298-5a694c4b29ef
println((context_size=context_size, hidden_size=hidden_size, latent_dims=latent_dims, data_dims=data_dims, stick_landing=stick_landing, batch_size=batch_size))

# ╔═╡ b8b2f4b5-e90c-4066-8dad-27e8dfa1d7c5
md"""
### Latent SDE Model
"""

# ╔═╡ 08759cda-2a2a-41ff-af94-5b1000c9e53f
solver = if gpu
	GPUEulerHeun()
else
	EulerHeun()
end

# ╔═╡ ec41b765-2f73-43a5-a575-c97a5a107c4e
println("Steps that will be derived: $(steps(tspan_model, dt))")

# ╔═╡ 63960546-2157-4a23-8578-ec27f27d5185
projector = if fixed_projector
	Chain(FlattenLayer(), Chain((x, ps, st) -> (x[1, :, :], st)))
else
	Lux.Dense(latent_dims => data_dims)
end

# ╔═╡ 001c318e-b7a6-48a5-bfd5-6dd0368873ac
latent_sde = StandardLatentSDE(
	solver,
	tspan_model,
	steps(tspan_model, dt);
	data_dims=data_dims,
	latent_dims=latent_dims,
	prior_size=prior_size,
	posterior_size=hidden_size,
	diffusion_size=Int(floor(hidden_size/latent_dims)),
	depth=depth,
	rnn_size=context_size,
	context_size=context_size,
	hidden_activation=tanh,
    final_activation=tanh,
	timedependent=time_dependence,
	adaptive=false,
	# we only have this custom layer - the others are default
	projector=projector
)

# ╔═╡ 0f6f4520-576f-42d3-9126-2076a51a6e22
begin
	rng = Xoshiro()
end

# ╔═╡ 1938e122-2c05-46fc-b179-db38322530ff
md"""
# Parameters
"""

# ╔═╡ 05568880-f931-4394-b31e-922850203721
ps_, st = if gpu
	Lux.setup(rng, latent_sde) |> gpu
else
	Lux.setup(rng, latent_sde)
end

# ╔═╡ b0692162-bdd2-4cb8-b99c-1ebd2177a3fd
begin
	ps = ComponentArray{Float64}(ps_)
end

# ╔═╡ ee3d4a2e-0960-430e-921a-17d340af497c
md"""
Select a seed: $(@bind seed Scrubbable(481283))
"""

# ╔═╡ 1af41258-0c18-464d-af91-036f5a4c074c
ensemblemode = if gpu
	EnsembleGPUKernel(CUDA.CUDABackend())
else
	EnsembleThreads()
end

# ╔═╡ 3ab9a483-08f2-4767-8bd5-ae1375a62dbe
function plot_prior(priorsamples; rng=rng, tspan=latent_sde.tspan, datasize=latent_sde.datasize)
	prior = NeuralSDEExploration.sample_prior_dataspace(latent_sde,ps,st;seed=abs(rand(rng, Int)),b=priorsamples, noise=(seed) -> noise(seed), tspan=tspan, datasize=datasize)
	return plot(prior, linewidth=.5,color=:black,legend=false,title="projected prior")
end

# ╔═╡ b5c6d43c-8252-4602-8232-b3d1b0bcee33
function plotmodel()
	n = 5
	rng_plot = Xoshiro(0)
	nums = sample(rng_plot, 1:length(timeseries.u), n; replace=false)
	
	posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(select_ts(nums, timeseries), ps, st, seed=seed, noise=noise)
	
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
		plot(map(x -> max(1e-8, x+100.0), recorded_loss), legend=false, title="loss", yscale=:log10)
		plot(map(x -> max(1e-8, -x+100.0), recorded_likelihood), legend=false, title="-loglike", yscale=:log10)
		plot(recorded_kl, legend=false, title="kl-divergence")
		plot(recorded_eta, legend=false, title="beta")
		plot(recorded_lr, legend=false, title="learning rate")
	]	

	
	l = @layout [a ; b c; d e]
	plot(plots...; layout=l)
end

# ╔═╡ e0a34be1-6aa2-4563-abc2-ea163a778752
function loss(ps, minibatch, eta, seed)
	_, _, _, kl_divergence, likelihood = latent_sde(minibatch, ps, st; sense=sense, noise=noise, ensemblemode=ensemblemode, seed=seed, likelihood_scale=scale)
	return mean(-likelihood .+ (eta * kl_divergence)), mean(kl_divergence), mean(likelihood)
end

# ╔═╡ f4a16e34-669e-4c93-bd83-e3622a747a3a
function train(lr_sched, num_steps, opt_state; kl_sched=Loop(x -> eta, 1))
	for step in 1:num_steps
		s = sample(rng, 1:length(timeseries.u), batch_size)
		
		minibatch = select_ts(s, timeseries)
		
		seed = rand(rng, UInt32)

		eta = popfirst!(kl_sched)
		lr = popfirst!(lr_sched)

		l, kl_divergence, likelihood = loss(ps, minibatch, eta, seed)

		push!(recorded_loss, l)
		push!(recorded_kl, kl_divergence)
		push!(recorded_likelihood, likelihood)
		push!(recorded_eta, eta)
		push!(recorded_lr, lr)

		println("Loss: $l, KL: $kl_divergence")
		dps = Zygote.gradient(ps -> loss(ps, minibatch, eta, seed)[1], ps)
		
		if step % 10 == 1
			GC.gc(true)
		end

		if kidger_trick
			dps[1].initial_prior *= length(timeseries.t)
		end
		
		Optimisers.update!(opt_state, ps, dps[1])
		Optimisers.adjust!(opt_state, lr)
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

	data = Dict(
		"latent_sde" => latent_sde,
		"timeseries" => timeseries,
		"ps" => ps,
		"st" => st,
		"model" => model,
		"initial_condition" => initial_condition,
		"args" => ARGS,
		"datamin" => datamin,
		"datamax" => datamax,
		"tspan_train" => tspan_train,
		"tspan_data" => tspan_data,
		"tspan_model" => tspan_model,
		"recorded_loss" => recorded_loss,
		"recorded_kl" => recorded_kl,
		"recorded_likelihood" => recorded_likelihood,
		"recorded_eta" => recorded_eta,
		"recorded_lr" => recorded_lr,
	)

	mkpath(folder)

	jldsave(folder * "$(epoch).jld"; data)
	
	modelfig = plotmodel()
	savefig(modelfig, folder * "$(epoch)_model.pdf")
	# savefig(modelfig, folder * "$(epoch)_model.tex")
	
	learningfig = plotlearning()
	savefig(learningfig, folder * "$(epoch)_learning.pdf")
	# savefig(learningfig, folder * "$(epoch)_learning.tex")
end

# ╔═╡ 7a7e8e9b-ca89-4826-8a5c-fe51d96152ad
if enabletraining
	@time dps = Zygote.gradient(ps -> loss(ps, select_ts(1:batch_size, timeseries), 1.0, 10)[1], ps)[1]
end

# ╔═╡ 67e5ae14-3062-4a93-9492-fc6e9861577f
kl_sched = if kl_anneal
	Iterators.Stateful(Loop(Sequence([Loop(x -> (eta*x)/kl_rate, kl_rate), Loop(x -> eta, kl_rate*3)], [kl_rate, kl_rate*3]), kl_rate*4))
else
	Iterators.Stateful(Loop(x -> eta, 1))
end

# ╔═╡ da2de05a-5d40-4293-98e0-abd20d6dcd2a
lr_sched = if lr_cycle
	Iterators.Stateful(CosAnneal(learning_rate, max_learning_rate, lr_rate))
else
	Iterators.Stateful(Exp(λ = learning_rate, γ = decay))
end

# ╔═╡ 78aa72e2-8188-441f-9910-1bc5525fda7a
begin
	if !(@isdefined PlutoRunner) && enabletraining  # running as job
		opt_state_job = Optimisers.setup(Optimisers.Adam(), ps)
		# precompile exportresults because there are some memory problems
		exportresults(0)
		for epoch in 1:1000
			train(lr_sched, 100, opt_state_job; kl_sched=kl_sched)
			exportresults(epoch)
			GC.gc(true)
		end
	end
end

# ╔═╡ 830f7e7a-71d0-43c8-8e74-d1709b8a6707
function gifplot()
	p1 = plotmodel()
	p2 = plotlearning()
	plot(p1, p2, layout=@layout[a;b],size=(600,700))
end

# ╔═╡ 763e07e6-dd46-42d6-b57a-8f1994386302
gifplot()

# ╔═╡ 655877c5-d636-4c1c-85c6-82129c1a4999
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	if enabletraining
		opt_state = Optimisers.setup(Optimisers.Adam(), ps)

		@gif for epoch in 1:10
			train(lr_sched, 10, opt_state; kl_sched=kl_sched)
			gifplot()
		end
	end
end
  ╠═╡ =#

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
# ╟─95bdd676-d8df-4fef-bdd7-cce85b717018
# ╟─4c3b8784-368d-49c3-a875-c54960ec9be5
# ╟─a65a7405-d1de-4de5-9391-dcb971ae0413
# ╟─e6a71aae-9d81-45a9-af9a-c4188dda2787
# ╟─71a38a66-dd66-4000-b664-fc3e04f6d4b8
# ╟─bae92e09-2e87-4a1e-aa2e-906f33985f6d
# ╟─bd7acf1a-c09a-4531-ad2c-b5e7e28af382
# ╟─42ece6c1-9e8a-45e7-adf4-6f353da6a4e5
# ╟─3665efa6-6527-4771-82fd-285c3c0f8b41
# ╟─10c206ef-2321-4c64-bdf4-7f4e9934d911
# ╟─fe7e2889-88de-49b3-b20b-342357596bfc
# ╟─de70d89a-275d-49d2-9da4-4470c869e56e
# ╟─986c442a-d02e-42d4-bda4-f66a1c92f799
# ╟─9a89a97c-da03-4887-ac8c-ef1f5264436e
# ╠═c441712f-e4b2-4f4a-83e1-aad558685288
# ╟─7e6256ef-6a0a-40cc-aa0a-c467b3a524c4
# ╠═2da6bbd4-8036-471c-b94e-10182cf8a834
# ╠═c00a97bf-5e10-4168-8d58-f4f9270258ac
# ╟─5691fcc5-29b3-4236-9154-59c6fede49ce
# ╠═15cef7cc-30b6-499d-b968-775b3251dedb
# ╟─1502612c-1489-4abf-8a8b-5b2d03a68cb1
# ╠═455263ef-2f94-4f3e-8401-e0da7fb3e493
# ╟─f4651b27-135e-45f1-8647-64ab08c2e8e8
# ╠═aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
# ╠═5d78e254-4134-4c2a-8092-03f6df7d5092
# ╠═c79a3a3a-5599-4585-83a4-c7b6bc017436
# ╠═9a5c942f-9e2d-4c6c-9cb1-b0dffd8050a0
# ╟─da11fb69-a8a1-456d-9ce7-63180ef27a83
# ╟─8280424c-b86f-49f5-a854-91e7abcf13ec
# ╟─fdceee23-b91e-4b2e-af78-776c01733de3
# ╟─97d724c2-24b0-415c-b90f-6a36e877e9d1
# ╟─d81ccb5f-de1c-4a01-93ce-3e7302caedc0
# ╟─6489b190-e08f-466c-93c4-92a723f8e594
# ╟─b5721107-7cf5-4da3-b22a-552e3d56bcfa
# ╟─efd438bc-13cc-457f-82c1-c6e0711079b3
# ╟─9382314d-c076-4b95-8171-71d903bb9271
# ╟─03a21651-9c95-49e8-bb07-b03640f7e5b7
# ╟─ad6247f6-6cb9-4a57-92d3-6328cbd84ecd
# ╟─60b5397d-7350-460b-9117-319dc127cc7e
# ╟─16c12354-5ab6-4c0e-833d-265642119ed2
# ╟─f12633b6-c770-439d-939f-c41b74a5c309
# ╟─3c630a3a-7714-41c7-8cc3-601cd6efbceb
# ╟─2961879e-cb52-4980-931b-6f8de1f26fa4
# ╟─7c23c32f-97bc-4c8d-ac54-42753be61345
# ╟─64e7bba4-fb17-4ed8-851f-de9204e0f42d
# ╟─33d53264-3c8f-4f63-9dd2-46ebd00f4e28
# ╟─8bb3084f-5fde-413e-b0fe-8b2e19673fae
# ╟─2c64b173-d4ad-477d-afde-5f3916e922ef
# ╟─9767a8ea-bdda-43fc-b636-8681d150d29f
# ╟─3db229e0-0e13-4d80-8680-58b89161db35
# ╟─cb1c2b2e-a2a2-45ed-9fc1-655d28f267d1
# ╟─7f219c33-b37b-480a-9d21-9ea8d898d5d5
# ╠═2bb433bb-17df-4a34-9ccf-58c0cf8b4dd3
# ╟─db88cae4-cb25-4628-9298-5a694c4b29ef
# ╟─b8b2f4b5-e90c-4066-8dad-27e8dfa1d7c5
# ╠═08759cda-2a2a-41ff-af94-5b1000c9e53f
# ╟─ec41b765-2f73-43a5-a575-c97a5a107c4e
# ╠═63960546-2157-4a23-8578-ec27f27d5185
# ╠═001c318e-b7a6-48a5-bfd5-6dd0368873ac
# ╠═0f6f4520-576f-42d3-9126-2076a51a6e22
# ╟─1938e122-2c05-46fc-b179-db38322530ff
# ╠═05568880-f931-4394-b31e-922850203721
# ╠═b0692162-bdd2-4cb8-b99c-1ebd2177a3fd
# ╟─ee3d4a2e-0960-430e-921a-17d340af497c
# ╠═1af41258-0c18-464d-af91-036f5a4c074c
# ╠═3ab9a483-08f2-4767-8bd5-ae1375a62dbe
# ╠═b5c6d43c-8252-4602-8232-b3d1b0bcee33
# ╟─225791b1-0ffc-48e2-8131-7f54848d8d83
# ╠═550d8974-cd19-4d0b-9492-adb4e14a04b1
# ╠═fa43f63d-8293-43cc-b099-3b69dbbf4b6a
# ╠═e0a34be1-6aa2-4563-abc2-ea163a778752
# ╠═f4a16e34-669e-4c93-bd83-e3622a747a3a
# ╠═9789decf-c384-42df-b7aa-3c2137a69a41
# ╠═7a7e8e9b-ca89-4826-8a5c-fe51d96152ad
# ╠═67e5ae14-3062-4a93-9492-fc6e9861577f
# ╠═da2de05a-5d40-4293-98e0-abd20d6dcd2a
# ╠═78aa72e2-8188-441f-9910-1bc5525fda7a
# ╠═830f7e7a-71d0-43c8-8e74-d1709b8a6707
# ╠═763e07e6-dd46-42d6-b57a-8f1994386302
# ╠═655877c5-d636-4c1c-85c6-82129c1a4999
