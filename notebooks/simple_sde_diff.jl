### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

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

# ╔═╡ b1bd691b-6635-4cdb-9841-bd7aca73e50a
drift = Lux.Dense(2 => 2)

# ╔═╡ 82a649b4-0f72-490f-8b5b-d69255ebfcde
diffusion = Lux.Dense(2 => 2)

# ╔═╡ 02d27388-4f12-44f3-b123-df10da452348
ps_drift, st_drift = Lux.setup(rng, drift)

# ╔═╡ 8ca6821b-2a06-4c50-a3af-af00636e8a7d
ps_diffusion, st_diffusion = Lux.setup(rng, diffusion)

# ╔═╡ 9b85af6a-6931-49b7-b3f0-75febe5925f3
ps = ComponentArray((drift=ps_drift, diffusion=ps_diffusion))

# ╔═╡ 9eda52c2-0eeb-453b-9fd4-8eb1fa4c9d5e
solver = EulerHeun()

# ╔═╡ e2efc1b1-de35-48e6-85f2-9df852ba888d
drift_f(u, p, t) = drift(u, p.drift, st_drift)[1]

# ╔═╡ 1657a1fc-f98b-4e9e-b347-75086575aeaf
diffusion_f(u, p, t) = diffusion(u, p.diffusion, st_diffusion)[1]

# ╔═╡ 38732a20-009b-48e5-ad23-07b449def12a
problem = SDEProblem{false}(drift_f, diffusion_f, [1f0, 1f0], (0f0, 1f0), ps)

# ╔═╡ e63e7869-2de6-478e-bc04-998b8858c1ce
plot(solve(problem, solver, dt=0.05))

# ╔═╡ ee76f88c-069e-45b1-bce0-6019983b5260
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 04d8ee06-9872-4481-8df5-26d47862261b


# ╔═╡ 6bf6a59c-549c-495b-a574-caa12c87e055
#=╠═╡
dps = Zygote.gradient(ps -> mean(NeuralSDEExploration.loss(latent_sde, ps, inputs, st, 1.0; seed=1, ensemblemode=EnsembleSerial())), ps)[1]
  ╠═╡ =#

# ╔═╡ ada03e49-be82-418a-a915-efbc78a81368
#=╠═╡
println(dps)
  ╠═╡ =#

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
# ╠═b1bd691b-6635-4cdb-9841-bd7aca73e50a
# ╠═82a649b4-0f72-490f-8b5b-d69255ebfcde
# ╠═02d27388-4f12-44f3-b123-df10da452348
# ╠═8ca6821b-2a06-4c50-a3af-af00636e8a7d
# ╠═9b85af6a-6931-49b7-b3f0-75febe5925f3
# ╠═9eda52c2-0eeb-453b-9fd4-8eb1fa4c9d5e
# ╠═e2efc1b1-de35-48e6-85f2-9df852ba888d
# ╠═1657a1fc-f98b-4e9e-b347-75086575aeaf
# ╠═38732a20-009b-48e5-ad23-07b449def12a
# ╠═e63e7869-2de6-478e-bc04-998b8858c1ce
# ╠═ee76f88c-069e-45b1-bce0-6019983b5260
# ╠═04d8ee06-9872-4481-8df5-26d47862261b
# ╠═6bf6a59c-549c-495b-a574-caa12c87e055
# ╠═ada03e49-be82-418a-a915-efbc78a81368
# ╠═e99f0f6a-950b-4ccc-8798-a3a10730b4f5
# ╠═83e1b7df-9cc4-4504-936e-69028ed7ee02
# ╠═07e32ac3-f419-46a5-91cb-4f88eb57c7e3
