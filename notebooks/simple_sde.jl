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
using Optimisers, StatsBase, Zygote, ForwardDiff, Enzyme, Lux, DifferentialEquations, Functors, ComponentArrays, Distributions, ParameterSchedulers, Random

# ╔═╡ 682a8844-d1a8-4919-8a57-41b942b7da25
using NeuralSDEExploration, Plots, PlutoUI, ProfileSVG

# ╔═╡ b94d5f36-b7fe-493d-a774-f03062ee5afa
using Profile, PProf

# ╔═╡ d86a3b4e-6735-4de2-85d3-b6106ae48444
Revise.retry()

# ╔═╡ 9e4de245-815a-4d38-bb14-7b7b29da24cf
rng = Xoshiro()

# ╔═╡ c56ceab4-6ca7-4ea8-a905-24b5e9d8f0e1
begin
	initial_prior = Lux.Dense(1 => 4; bias=false, init_weight=zeros)
    initial_posterior = Lux.Dense(1 => 4; bias=false, init_weight=ones)
    drift_prior = Lux.Scale(2; bias=false)
    drift_posterior = Lux.Dense(3 => 2; init_weight=ones, bias=false)
	diffusion = Lux.Parallel(nothing, [Lux.Dense(1 => 1; init_weight=zeros, init_bias=ones) for i in 1:2]...)
    encoder = Lux.Recurrence(Lux.RNNCell(1 => 1; init_weight=ones); return_sequence=true)
    projector = Lux.Dense(2 => 1; bias=false, init_weight=ones)

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
		EulerHeun(),
        tspan;
        saveat=range(tspan[1], tspan[end], datasize),
        dt=(tspan[end]/datasize),
    )
	ps_, st = Lux.setup(rng, latent_sde)
end

# ╔═╡ 23601b08-2d78-426c-9702-53232c0db1bc
display(ps_)

# ╔═╡ d5214491-e569-4a0c-b2f9-dbf0b91e0566
ps__ = (initial_prior = (weight = [0.0; 0.0; 0.0; 0.0;;],), initial_posterior = (weight = [100.0; 0.01; 100.0; 0.01;;],), drift_prior = (weight = Float32[1.0, 1.0],), drift_posterior = (weight = [0.1 0.1 0.1; 1.0 1.0 1.0],), diffusion = (layer_1 = (weight = [0.0;;], bias = [0.01;;]), layer_2 = (weight = [0.0;;], bias = [1.0;;])), encoder = (weight_ih = [1.0;;], weight_hh = [1.0;;], bias = Float32[0.0]), projector = (weight = [1.0 1.0],))

# ╔═╡ 9057f759-322b-4fe7-8a3a-3a3c16c03de2
ps = ComponentArray(ps__)

# ╔═╡ af673c70-c7bf-4fe6-92c0-b5e09fd99195
inputs = [(t=range(tspan[1],tspan[end],datasize),u=[f(x) for x in range(tspan[1],tspan[end],datasize)]) for f in [(x)->-100-x, (x)->100+x, (x)->100+x, (x)->100+x]]

# ╔═╡ 7b81e46b-c55a-42e4-af34-9d1101de4b9c
@bind seed Slider(1:100)

# ╔═╡ 02bc2a6e-1497-4bf3-ac67-26c770425c22
m1 = reshape([.1, .2, .3, .4], 1, :, 1)

# ╔═╡ ae0c97bf-a307-4e61-83c1-8bf84d2d7f0a
m2 = reshape([.5, .6, .7, .8], 1, :, 1)

# ╔═╡ 0903e805-2147-42fd-919f-5bf2103df859
m3 = cat(m1, m2; dims=3)

# ╔═╡ 78ad30c4-38b9-4f1a-8cf8-9174708286e1
encoder(m1, ps.encoder, st.encoder)

# ╔═╡ 7dfb3923-c0fc-4f88-9104-7d619ff415cd
encoder(m2, ps.encoder, st.encoder)

# ╔═╡ 1bcb6fba-8286-4008-98ff-d581760ffff6
encoder(m3, ps.encoder, st.encoder)

# ╔═╡ e8ef1773-8087-4f47-abfe-11e73f28a269
posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, inputs, st)

# ╔═╡ 7fabae7c-abef-4d2c-a3b5-b9d2f683dc26
plot(logterm_[1, :, :]', label="KL-Divergence")

# ╔═╡ 937d5963-eddc-4296-9b6e-9532eb57bdf2
plot(posterior_data[1, :, 1:4]', label="posterior")

# ╔═╡ 3eb43be1-1ae9-499d-a350-d34d8daa30ca
Profile.clear()

# ╔═╡ fd6d9e72-edc6-4ed1-9956-0b2164bc8b97
@profile posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, inputs, st, seed=seed)

# ╔═╡ 4b555ed0-0f4e-4619-9358-9bdc5e09fa76
pprof()

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
# ╠═9e4de245-815a-4d38-bb14-7b7b29da24cf
# ╠═c56ceab4-6ca7-4ea8-a905-24b5e9d8f0e1
# ╠═23601b08-2d78-426c-9702-53232c0db1bc
# ╠═d5214491-e569-4a0c-b2f9-dbf0b91e0566
# ╠═9057f759-322b-4fe7-8a3a-3a3c16c03de2
# ╠═af673c70-c7bf-4fe6-92c0-b5e09fd99195
# ╠═7b81e46b-c55a-42e4-af34-9d1101de4b9c
# ╠═02bc2a6e-1497-4bf3-ac67-26c770425c22
# ╠═ae0c97bf-a307-4e61-83c1-8bf84d2d7f0a
# ╠═0903e805-2147-42fd-919f-5bf2103df859
# ╠═78ad30c4-38b9-4f1a-8cf8-9174708286e1
# ╠═7dfb3923-c0fc-4f88-9104-7d619ff415cd
# ╠═1bcb6fba-8286-4008-98ff-d581760ffff6
# ╠═e8ef1773-8087-4f47-abfe-11e73f28a269
# ╠═7fabae7c-abef-4d2c-a3b5-b9d2f683dc26
# ╠═937d5963-eddc-4296-9b6e-9532eb57bdf2
# ╠═b94d5f36-b7fe-493d-a774-f03062ee5afa
# ╠═3eb43be1-1ae9-499d-a350-d34d8daa30ca
# ╠═fd6d9e72-edc6-4ed1-9956-0b2164bc8b97
# ╠═4b555ed0-0f4e-4619-9358-9bdc5e09fa76
# ╠═bfa3098d-1a8a-496b-89b4-5970d426d8c6
# ╠═f715c673-d017-4154-92be-cf8222cc22e6
# ╠═a8a635b8-41a6-4c96-8351-0e06595da0a5
# ╠═cec50edd-d73d-443f-8805-885f4e8c23c4
# ╠═2b1b8c73-b8fc-4c8e-860d-72374f29e7b5
