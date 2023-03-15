### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ c5916df0-c180-11ed-27a8-038cdb9d5940
begin
	import Pkg
	Pkg.activate("..")
	using Revise
end

# ╔═╡ 734441ce-fdae-49c9-8ec2-20a54630abfc
using Zygote, Enzyme, Lux, Random, ForwardDiff

# ╔═╡ f04a5047-1ee0-4518-b559-7560bfa4908b
grucell = Lux.Recurrence(Lux.GRUCell(1 => 3))

# ╔═╡ 4578fdd4-e70c-45a3-a5ca-5cf047c593d7
rng = Random.default_rng()

# ╔═╡ eed80a27-7b4a-444c-ab8c-0421db77a531
ps, st = Lux.setup(rng, grucell)

# ╔═╡ 4b7f2095-b2d6-4cc2-b12b-12b851e5af26
function gruit(input, ps)
	grucell(input, ps, st)
end

# ╔═╡ 841a59cc-44f3-4734-a2c9-67bcae369ce6
arg = reshape(1:2, 1, 1, :)

# ╔═╡ 5088b64d-91c6-4568-bab5-88d06cb6489c
gruit(arg, ps)

# ╔═╡ 6b1b17f8-2658-4932-ae7b-baad005281b6
gruit(reshape(1:3, 1, 1, :), ps)

# ╔═╡ 8be2dc0a-aa0b-4912-b544-1706a9bc23e8
gruit(reshape(3:3, 1, 1, :), ps)

# ╔═╡ d9f77ac2-7345-471f-a92e-20d90a8eaa7c
gruit(arg, ps)[1] |> size

# ╔═╡ 765307bd-1f5d-42da-97e1-333b695234c2
Zygote.gradient(gruit, arg, ps)

# ╔═╡ 9fbdc8d8-8f3f-46e5-acd1-94a205b83d0b
ForwardDiff.gradient(gruit, arg, ps)

# ╔═╡ Cell order:
# ╠═c5916df0-c180-11ed-27a8-038cdb9d5940
# ╠═734441ce-fdae-49c9-8ec2-20a54630abfc
# ╠═f04a5047-1ee0-4518-b559-7560bfa4908b
# ╠═4578fdd4-e70c-45a3-a5ca-5cf047c593d7
# ╠═eed80a27-7b4a-444c-ab8c-0421db77a531
# ╠═4b7f2095-b2d6-4cc2-b12b-12b851e5af26
# ╠═841a59cc-44f3-4734-a2c9-67bcae369ce6
# ╠═5088b64d-91c6-4568-bab5-88d06cb6489c
# ╠═6b1b17f8-2658-4932-ae7b-baad005281b6
# ╠═8be2dc0a-aa0b-4912-b544-1706a9bc23e8
# ╠═d9f77ac2-7345-471f-a92e-20d90a8eaa7c
# ╠═765307bd-1f5d-42da-97e1-333b695234c2
# ╠═9fbdc8d8-8f3f-46e5-acd1-94a205b83d0b
