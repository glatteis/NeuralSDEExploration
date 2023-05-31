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

# ╔═╡ 7a6bbfd6-ffb7-11ed-39d7-5b673fe4cdae
begin
	if @isdefined PlutoRunner  # running inside Pluto
		import Pkg
		Pkg.activate("..")
		using Revise
	end
end

# ╔═╡ ae7d6e11-55da-44a2-a5b6-60d11caa9dbf
using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs, JLD2

# ╔═╡ 6995cd16-0c69-49c7-9523-4c842c0db339
begin
	if @isdefined PlutoRunner  # running inside Pluto
		Revise.retry()
	else
		ENV["GKSwstype"]="nul" # no GTK for plots
		println(join(ARGS, " "))
	end
end

# ╔═╡ 43e6a892-8b2d-4fbb-b49c-681d78f0202c
md"""
Model filename: $(@bind f TextField())
"""

# ╔═╡ 09a13088-a784-4d53-8012-cef948eb796c
dict = f === "" ? (nothing, nothing) : load(f)["data"]

# ╔═╡ 9f89a8d9-05e5-4af0-9dd8-1a528ea7e9de
latent_sde = dict["latent_sde"]

# ╔═╡ 7da4b3b6-3cee-4b93-9359-a2f7e2341da9
ps = dict["ps"]

# ╔═╡ 23bc5697-f1ca-4598-92fb-2d43e94ce310
st = dict["st"]

# ╔═╡ 5e6398df-a1f5-4cce-a8ba-36c497cd6cda
plot(sample_prior(latent_sde, ps, st))

# ╔═╡ Cell order:
# ╠═7a6bbfd6-ffb7-11ed-39d7-5b673fe4cdae
# ╠═6995cd16-0c69-49c7-9523-4c842c0db339
# ╠═ae7d6e11-55da-44a2-a5b6-60d11caa9dbf
# ╟─43e6a892-8b2d-4fbb-b49c-681d78f0202c
# ╠═09a13088-a784-4d53-8012-cef948eb796c
# ╠═9f89a8d9-05e5-4af0-9dd8-1a528ea7e9de
# ╠═7da4b3b6-3cee-4b93-9359-a2f7e2341da9
# ╠═23bc5697-f1ca-4598-92fb-2d43e94ce310
# ╠═5e6398df-a1f5-4cce-a8ba-36c497cd6cda
