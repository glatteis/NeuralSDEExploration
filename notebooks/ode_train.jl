### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 67cb574d-7bd6-40d9-9dc3-d57f4226cc83
begin
	import Pkg
	Pkg.activate("..")
end

# ╔═╡ d1440209-78f7-4a9a-9101-a06ad2534e5d
using NeuralSDEExploration, Plots, PlutoUI

# ╔═╡ b1b4b2c4-3d72-4a86-a302-b48bb5e7eef8
using Lux

# ╔═╡ bcb44277-1151-4cd3-8dfe-ea3b0b254f7c
using Random, ComponentArrays

# ╔═╡ 4cf2f415-0e74-4a2d-9611-ae58e4be783f
using Zygote, Optimisers, NODEData, StatsBase

# ╔═╡ 32be3e35-a529-4d16-8ba0-ec4e223ae401
md"""
Let's train a Neural ODE from a simple zero-dimensional energy balance model. First, let's just instantiate the predefined model from the package...
"""

# ╔═╡ f74dd752-485b-4203-9d72-c56e55a3ef76
ebm = NeuralSDEExploration.ZeroDEnergyBalanceModelNonStochastic()

# ╔═╡ cc2418c2-c355-4291-b5d7-d9019787834f
md"Let's generate the data and plot a quick example:"

# ╔═╡ dd03f851-2e26-4850-a7d4-a64f154d2872
begin
	n = 100
    datasize = 30
    tspan = (0.0e0, 1e0)
end

# ╔═╡ c00a97bf-5e10-4168-8d58-f4f9270258ac
timeseries = map(x -> NODEDataloader(x, 20), series(ebm, range(210.0e0, 320.0e0, n), tspan, datasize))

# ╔═╡ c58ebda6-4c51-4103-b340-ecac7339e551
md"""
We've packaged the data into a `NODEDataloader` container which splits it up into batches. This lets us use multiple "starting points" and smaller trajectories in the data instead of fitting an entire trajectory at once.
"""

# ╔═╡ f4651b27-135e-45f1-8647-64ab08c2e8e8
md"""
Let's normalize our data for training:
"""

# ╔═╡ aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
begin
	flatdata = vcat([[max(u...) for (t,u) in ts] for ts in timeseries]...)
    datamax = max(flatdata...)
    datamin = min(flatdata...)

    function normalize(x)
        return (x - datamin) / (datamax - datamin)
    end

    function rescale(x)
        return (x * (datamax - datamin)) + datamin
    end
end

# ╔═╡ c0b55609-4cf8-4c99-8ba8-3e737e2b2807
md"""
Let's define our neural network - we're using a helper in the NeuralSDEExploration package that would allow us to add known terms, but currently we just want to have an ANN in the right-hand side.
"""

# ╔═╡ 001c318e-b7a6-48a5-bfd5-6dd0368873ac
neural_ode = AugmentedNeuralODE(
    Lux.Chain(Lux.Dense(1 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 1)),
    function (u, p, t, model, st)
        net, st = model(u, p, st)
        return net
    end,
    tspan,
)

# ╔═╡ 02a3b498-2752-47f6-81f2-1114cb5614d9
md"""
Setup our Lux environment:
"""

# ╔═╡ c3a8d643-3c1f-4f8e-9802-77085fe43d4f
rng = Random.default_rng()

# ╔═╡ 2c9eb70f-2716-4ea9-a123-9b0d1da23051
pslux, st = Lux.setup(rng, neural_ode)

# ╔═╡ b37ca7a1-eaf8-4c21-b751-ab8fb279fb17
ps = ComponentArray(pslux)

# ╔═╡ 47f595df-cf72-4981-8445-2153787ea034
md"""
We need a function that runs the Neural ODE. The data is encoded into `(t, u)` tuples where `t` is the time and `u` is the current value. We'll unwrap this, integrate the Neural ODE on the normalized initial value of `u`, and pass the correct `tspan` (the timespan of the particular batch) and `saveat` (the time span + step num). We also need to pass the Lux state `ps` and `st`. This results in this monster expression:
"""

# ╔═╡ c236fa26-26d2-4cb6-8d31-1f78df10131e
function apply(ps, data)
    [Lux.apply(remake(neural_ode, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st)[1] for (t, u) in data]
end


# ╔═╡ 6d66e81b-66c2-4797-b4ed-553edae8dc9e
md"""
The loss function uses `apply`, rescales it and computes the squared sum between points generated by the Neural ODE and the data. It then sums over all batches:
"""

# ╔═╡ b56e2106-01cc-42fb-918d-0e1737547e09
function loss_neuralsde(ps, data)
    trajectories = apply(ps, data)

    sum(abs, map(v -> sum(abs2, v), map(x -> map(y -> rescale(first(y)), x.u), trajectories) .- map(tu -> tu[2], data)))
end

# ╔═╡ a532d098-d41a-4248-97e0-d26f54712f44
md"""
I'm also defining a callback function for plotting:
"""

# ╔═╡ 8c4ebf80-bc10-4964-904d-1781a848f368
function cb(data)
	trajectories = apply(ps, data)

	trajectories2 = []
	t = 0e0:0.01e0:3e0

	st´ = st
	for u in 200e0:5e0:350e0
		trajectory, st´´ = Lux.apply(remake(neural_ode, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st´)
		push!(trajectories2, trajectory)
		st´ = merge(st´, st´´)
	end

	pl = plot()
	for t in trajectories
		plot!(pl, (map(first, t.t), map(y -> rescale(first(y)), t.u)), linewidth=1, label="series (prediction)", color="orange", legend=false)
	end
	for t in trajectories2
		plot!(pl, (map(first, t.t), map(y -> rescale(first(y)), t.u)), linewidth=1, label="series (prediction)", color="green", legend=false)
	end

	plot!(pl, data, linewidth=1, label="series (data)", legend=false, color="blue", dpi=100)
end


# ╔═╡ a7b7c87c-862a-4896-86c0-092c2f3b9040
function train(learning_rate, batch_size, num_steps)
	opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)
	batch = map(x -> NODEDataloader(x, batch_size), timeseries)
	println("Batch Size $batch_size")
	anim = @animate for step in 1:num_steps
		minibatch = batch[sample(1:size(batch)[1], 10, replace=false)]
		data = reduce(vcat, [[x for x in minibatch[i]] for i in 1:size(minibatch)[1]])
		if step % 20 == 1
			cb(data)
		end
		loss = loss_neuralsde(ps, data)
		grads = Zygote.gradient((x -> loss_neuralsde(x, data)), ps)[1]
		#println("Loss: $loss, Grads: $grads")
		Optimisers.update!(opt_state, ps, grads)
	end
	return gif(anim)
end

# ╔═╡ e40c4b61-ac8c-43db-ba7e-4f17fb0f4915
plot()

# ╔═╡ 326a166c-64e6-4dae-bbcc-4471c637f7ed
train(0.05, 5, 200)

# ╔═╡ 18724b64-1add-4889-93ec-3abbcc91ad78
train(0.02, 10, 200)

# ╔═╡ Cell order:
# ╠═67cb574d-7bd6-40d9-9dc3-d57f4226cc83
# ╠═d1440209-78f7-4a9a-9101-a06ad2534e5d
# ╟─32be3e35-a529-4d16-8ba0-ec4e223ae401
# ╠═f74dd752-485b-4203-9d72-c56e55a3ef76
# ╟─cc2418c2-c355-4291-b5d7-d9019787834f
# ╠═dd03f851-2e26-4850-a7d4-a64f154d2872
# ╠═c00a97bf-5e10-4168-8d58-f4f9270258ac
# ╟─c58ebda6-4c51-4103-b340-ecac7339e551
# ╟─f4651b27-135e-45f1-8647-64ab08c2e8e8
# ╠═aff1c9d9-b29b-4b2c-b3f1-1e06a9370f64
# ╟─c0b55609-4cf8-4c99-8ba8-3e737e2b2807
# ╠═b1b4b2c4-3d72-4a86-a302-b48bb5e7eef8
# ╠═001c318e-b7a6-48a5-bfd5-6dd0368873ac
# ╟─02a3b498-2752-47f6-81f2-1114cb5614d9
# ╠═bcb44277-1151-4cd3-8dfe-ea3b0b254f7c
# ╠═c3a8d643-3c1f-4f8e-9802-77085fe43d4f
# ╠═2c9eb70f-2716-4ea9-a123-9b0d1da23051
# ╠═b37ca7a1-eaf8-4c21-b751-ab8fb279fb17
# ╟─47f595df-cf72-4981-8445-2153787ea034
# ╠═c236fa26-26d2-4cb6-8d31-1f78df10131e
# ╟─6d66e81b-66c2-4797-b4ed-553edae8dc9e
# ╠═b56e2106-01cc-42fb-918d-0e1737547e09
# ╟─a532d098-d41a-4248-97e0-d26f54712f44
# ╠═8c4ebf80-bc10-4964-904d-1781a848f368
# ╠═4cf2f415-0e74-4a2d-9611-ae58e4be783f
# ╠═a7b7c87c-862a-4896-86c0-092c2f3b9040
# ╠═e40c4b61-ac8c-43db-ba7e-4f17fb0f4915
# ╠═326a166c-64e6-4dae-bbcc-4471c637f7ed
# ╠═18724b64-1add-4889-93ec-3abbcc91ad78
