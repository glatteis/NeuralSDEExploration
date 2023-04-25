### A Pluto.jl notebook ###
# v0.19.24

using Markdown
using InteractiveUtils

# ╔═╡ 3c16a0ef-3e0b-4538-b5df-919b7f1c1eba
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ 827671ee-e27a-11ed-3826-09545f50bcbb
using Flux, Profile, PProf, Random, ComponentArrays

# ╔═╡ d0181b4a-40d8-485e-98ca-fc77d762b27e
layer_size = 128

# ╔═╡ d9379dc9-f607-47e0-8399-6c7a62f45553
chain = Flux.Chain(Flux.Dense(1 => layer_size), Flux.Dense(layer_size => layer_size), Flux.Dense(layer_size => 1))

# ╔═╡ 54e332bd-66cc-461c-b947-fdf44e8be256
params_, re = Flux.destructure(chain)

# ╔═╡ b3fef36e-bf34-4000-b0da-4fea3bcca186
params = ComponentArray(params_)

# ╔═╡ c7126838-2ee3-4e35-8c87-100aa1fb30cf
function eval(steps, input)
	vec = input
	for i in 1:steps
		vec = re(params)(vec)
	end
end

# ╔═╡ 803c27f2-a12c-4e15-8072-7903efd7c591
begin
	Profile.clear()
	@profile eval(100000, rand(Float32, 1))
	pprof(; web=true)
end

# ╔═╡ Cell order:
# ╠═3c16a0ef-3e0b-4538-b5df-919b7f1c1eba
# ╠═827671ee-e27a-11ed-3826-09545f50bcbb
# ╠═d0181b4a-40d8-485e-98ca-fc77d762b27e
# ╠═d9379dc9-f607-47e0-8399-6c7a62f45553
# ╠═54e332bd-66cc-461c-b947-fdf44e8be256
# ╠═b3fef36e-bf34-4000-b0da-4fea3bcca186
# ╠═c7126838-2ee3-4e35-8c87-100aa1fb30cf
# ╠═803c27f2-a12c-4e15-8072-7903efd7c591
