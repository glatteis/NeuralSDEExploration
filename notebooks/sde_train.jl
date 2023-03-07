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

# ╔═╡ d1440209-78f7-4a9a-9101-a06ad2534e5d
using NeuralSDEExploration, Plots, PlutoUI

# ╔═╡ b1b4b2c4-3d72-4a86-a302-b48bb5e7eef8
using Lux

# ╔═╡ bcb44277-1151-4cd3-8dfe-ea3b0b254f7c
using Random, ComponentArrays

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
encoder = Lux.Chain(Lux.StatefulRecurrentCell(Lux.GRUCell(data_dims => hidden_size)), Lux.Dense(hidden_size => context_size, tanh))

# ╔═╡ 95f42c13-4a06-487c-bd35-a8d6bac9e02e
@bind i Slider(1:length(timeseries))

# ╔═╡ bdf4c32c-9a2b-4814-8513-c6e16ebee69c
plot(timeseries[i].u, legend=nothing)

# ╔═╡ 0d09a662-78d3-4202-b2be-843a0669fc9f
md"""
The `initial_posterior` net is the posterior for the initial state. It takes the context and outputs a mean and standard devation for the position zero of the posterior. The `initial_prior` is a fixed gaussian distribution.
"""

# ╔═╡ cdebb87f-9759-4e6b-a00a-d764b3c7fbf8
initial_posterior = Lux.Dense(context_size => latent_dims + latent_dims)

# ╔═╡ 3ec28482-2d6c-4125-8d85-4fb46b130677
initial_prior = Lux.Dense(0 => latent_dims + latent_dims, init_bias=Lux.glorot_uniform)

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
Diffusion. Prior and posterior share the same diffusion (they are not actually evaluated seperately while training, only their KL divergence). This is a diagonal diffusion, i.e. every term in the latent space has its own independent Weiner process.
"""

# ╔═╡ a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9
#diffusion = Lux.Chain(Lux.Dense(latent_dims => hidden_size, tanh; bias=false), Lux.Dense(hidden_size => latent_dims, tanh; bias=false), Lux.Scale(latent_dims; bias=false))
diffusion = Lux.Chain(Lux.Scale(latent_dims; bias=false))

# ╔═╡ bfabcd80-fb62-410f-8710-f577852c77df
md"""
The projector will transform the latent space back into data space.
"""

# ╔═╡ f0486891-b8b3-4a39-91df-1389d6f799e1
projector = Lux.Dense(latent_dims => data_dims)

# ╔═╡ c0b55609-4cf8-4c99-8ba8-3e737e2b2807
md"""
Let's define our neural network - we're using a helper in the NeuralSDEExploration package that would allow us to add known terms, but currently we just want to have two ANNs in the right-hand side. The Neural SDE has the form
$d u = f_{\theta}(u) dt + g_{\pi}(u) d W_t$
where $f$ and $g$ are the ANNs parametrized by $\theta$ and $\pi$.
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
    tspan;
	saveat=range(tspan[1], tspan[end], datasize),
	seed=300
)

# ╔═╡ 02a3b498-2752-47f6-81f2-1114cb5614d9
md"""
Setup our Lux environment:
"""

# ╔═╡ c3a8d643-3c1f-4f8e-9802-77085fe43d4f
rng = Random.default_rng()

# ╔═╡ 2c9eb70f-2716-4ea9-a123-9b0d1da23051
pslux, st = Lux.setup(rng, latent_sde)

# ╔═╡ b37ca7a1-eaf8-4c21-b751-ab8fb279fb17
ps = ComponentArray(pslux)

# ╔═╡ 63632673-4374-450c-a48d-493ae7a6b1ce
begin
	example_context = NeuralSDEExploration.encode(latent_sde, timeseries[i], ps, st)
	plot(hcat(example_context...)', legend=nothing)
end

# ╔═╡ 9ea12ddb-ff8a-4c16-b2a5-8b7603f262a3
md"""
Integrate the prior SDE in latent space to see what a run looks like:
"""

# ╔═╡ 9346f569-d5f9-43cd-9302-1ee64ef9a030
plot(NeuralSDEExploration.sample_prior(latent_sde, ps, st)[1])

# ╔═╡ b98200a8-bf73-42a2-a357-af56812d01c3
md"""
Integrate the posterior SDE in latent space given context:
"""

# ╔═╡ ce6a7357-0dd7-4581-b786-92d1eddd114d
plot(NeuralSDEExploration.sample_posterior(latent_sde, timeseries[1], ps, st, eps=0.4)[1])

# ╔═╡ 7a3e92cb-51b6-455f-b6dd-ff90896a9ffb
CairoMakie.streamplot((x, y) -> CairoMakie.Point2(Lux.apply(drift_prior, [x, y], ps.drift_prior, st.drift_prior)[1]), (-4, 4), (-4, 4))

# ╔═╡ 557b968f-b409-4bd9-8460-edef0ee8f2e6
posterior_, logterm_, kl_divergence_ = NeuralSDEExploration.pass(latent_sde, timeseries[1], ps, st)

# ╔═╡ dabf2a1f-ec78-4942-973f-4dbf9037ee7b
plot(hcat(logterm_...)')

# ╔═╡ 38324c42-e5c7-4b59-8129-0e4c17ab5bf1
plot(hcat([only(projector(x, ps.projector, st.projector)[1]) for x in posterior_], timeseries[1].u))

# ╔═╡ Cell order:
# ╠═67cb574d-7bd6-40d9-9dc3-d57f4226cc83
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
# ╟─847587ec-9297-471e-b788-7b776c05851e
# ╠═95f42c13-4a06-487c-bd35-a8d6bac9e02e
# ╠═63632673-4374-450c-a48d-493ae7a6b1ce
# ╠═bdf4c32c-9a2b-4814-8513-c6e16ebee69c
# ╟─0d09a662-78d3-4202-b2be-843a0669fc9f
# ╟─cdebb87f-9759-4e6b-a00a-d764b3c7fbf8
# ╠═3ec28482-2d6c-4125-8d85-4fb46b130677
# ╟─8b283d5e-1ce7-4deb-b382-6fa8e5612ef1
# ╟─c14806bd-42cf-480b-b618-bfe72183feb3
# ╟─64dc2da0-48cc-4242-bb17-449a300688c7
# ╟─df2034fd-560d-4529-836a-13745f976c1f
# ╟─4a97576c-96de-458e-b881-3f5dd140fa6a
# ╟─a1cb11fb-ec69-4ba2-9ed1-2d1a6d24ccd9
# ╟─bfabcd80-fb62-410f-8710-f577852c77df
# ╟─f0486891-b8b3-4a39-91df-1389d6f799e1
# ╟─c0b55609-4cf8-4c99-8ba8-3e737e2b2807
# ╠═b1b4b2c4-3d72-4a86-a302-b48bb5e7eef8
# ╠═001c318e-b7a6-48a5-bfd5-6dd0368873ac
# ╟─02a3b498-2752-47f6-81f2-1114cb5614d9
# ╠═bcb44277-1151-4cd3-8dfe-ea3b0b254f7c
# ╠═c3a8d643-3c1f-4f8e-9802-77085fe43d4f
# ╠═2c9eb70f-2716-4ea9-a123-9b0d1da23051
# ╠═b37ca7a1-eaf8-4c21-b751-ab8fb279fb17
# ╟─9ea12ddb-ff8a-4c16-b2a5-8b7603f262a3
# ╠═9346f569-d5f9-43cd-9302-1ee64ef9a030
# ╟─b98200a8-bf73-42a2-a357-af56812d01c3
# ╠═ce6a7357-0dd7-4581-b786-92d1eddd114d
# ╠═7a3e92cb-51b6-455f-b6dd-ff90896a9ffb
# ╠═557b968f-b409-4bd9-8460-edef0ee8f2e6
# ╠═dabf2a1f-ec78-4942-973f-4dbf9037ee7b
# ╠═38324c42-e5c7-4b59-8129-0e4c17ab5bf1
