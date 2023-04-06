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

# ╔═╡ 65947e8c-d1f3-11ed-2067-93e06230d83c
begin
	import Pkg
	Pkg.activate("..")
	using Revise
end

# ╔═╡ 9f89ad4a-4ffc-4cc8-bd7d-916ff6d6fa10
using Optimisers, StatsBase, Zygote, ForwardDiff, Enzyme, Flux, DifferentialEquations, Functors, ComponentArrays, Distributions, ParameterSchedulers, Random

# ╔═╡ 682a8844-d1a8-4919-8a57-41b942b7da25
using NeuralSDEExploration, Plots, PlutoUI, ProfileSVG

# ╔═╡ d86a3b4e-6735-4de2-85d3-b6106ae48444
Revise.retry()

# ╔═╡ c56ceab4-6ca7-4ea8-a905-24b5e9d8f0e1
begin
	initial_prior = Flux.Dense(1 => 4; bias=false, init=zeros) |> f64
    initial_posterior = Flux.Dense(reshape([1.0, 0.01, 1.0, 0.01], :, 1), false) |> f64
    drift_prior = Flux.Scale([1.0, 1.0], false) |> f64
    drift_posterior = Flux.Dense(ones(2, 3), false) |> f64
    diffusion = [Flux.Dense(1 => 1, bias=[0.000001], init=zeros) |> f64 for i in 1:2]
    encoder = Flux.RNN(1 => 1, (x) -> x; init=ones) |> f64
    projector = Flux.Dense(2 => 1; bias=false, init=ones) |> f64

    tspan = (0.0, 5.0)
    datasize = 10

    latent_sde = LatentSDE(
        initial_prior,
        initial_posterior,
        drift_prior,
        drift_posterior,
        diffusion,
        encoder,
        projector,
        tspan,
        EulerHeun(),
		EnsembleSerial();
        saveat=range(tspan[1], tspan[end], datasize),
        dt=(tspan[end]/datasize),
    )
    ps_, re = Functors.functor(latent_sde)
    ps = ComponentArray(ps_)
end

# ╔═╡ af673c70-c7bf-4fe6-92c0-b5e09fd99195
inputs = [(t=range(tspan[1],tspan[end],datasize),u=[f(x) for x in range(tspan[1],tspan[end],datasize)]) for f in [(x)->-100-x, (x)->100+x, (x)->100+x, (x)->100+x]]

# ╔═╡ 7b81e46b-c55a-42e4-af34-9d1101de4b9c
@bind seed Slider(1:100)

# ╔═╡ e8ef1773-8087-4f47-abfe-11e73f28a269
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, inputs, seed=seed)

# ╔═╡ bfa3098d-1a8a-496b-89b4-5970d426d8c6
posterior_latent[1, :, 1]

# ╔═╡ f715c673-d017-4154-92be-cf8222cc22e6
latent_sde.drift_posterior_p

# ╔═╡ a8a635b8-41a6-4c96-8351-0e06595da0a5
latent_sde.drift_posterior_re(latent_sde.drift_posterior_p)([-100.0428108378343, -100.0428108378343, -100.0])

# ╔═╡ cec50edd-d73d-443f-8805-885f4e8c23c4
plot(posterior_latent[1, :, :]')

# ╔═╡ 2b1b8c73-b8fc-4c8e-860d-72374f29e7b5
plot(posterior_data[1, :, :]')

# ╔═╡ Cell order:
# ╠═65947e8c-d1f3-11ed-2067-93e06230d83c
# ╠═d86a3b4e-6735-4de2-85d3-b6106ae48444
# ╠═9f89ad4a-4ffc-4cc8-bd7d-916ff6d6fa10
# ╠═682a8844-d1a8-4919-8a57-41b942b7da25
# ╠═c56ceab4-6ca7-4ea8-a905-24b5e9d8f0e1
# ╠═af673c70-c7bf-4fe6-92c0-b5e09fd99195
# ╠═7b81e46b-c55a-42e4-af34-9d1101de4b9c
# ╠═e8ef1773-8087-4f47-abfe-11e73f28a269
# ╠═bfa3098d-1a8a-496b-89b4-5970d426d8c6
# ╠═f715c673-d017-4154-92be-cf8222cc22e6
# ╠═a8a635b8-41a6-4c96-8351-0e06595da0a5
# ╠═cec50edd-d73d-443f-8805-885f4e8c23c4
# ╠═2b1b8c73-b8fc-4c8e-860d-72374f29e7b5
