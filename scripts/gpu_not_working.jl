using Lux, Zygote, DifferentialEquations, ComponentArrays, Random, CUDA, SciMLSensitivity

rng = Xoshiro()

drift_net = Dense(1 => 1)
diffusion_net = Dense(1 => 1)

ps_drift_, st_drift = Lux.setup(rng, drift_net)

ps_diffusion_, st_diffusion = Lux.setup(rng, diffusion_net)

ps_ = ComponentArray((ps_drift=ps_drift_,ps_diffusion=ps_diffusion_)) |> Lux.gpu

function drift(u, ps, t)
    drift_net(u, ps.ps_drift, st_drift)[1]
end
function diffusion(u, ps, t)
    diffusion_net(u, ps.ps_diffusion, st_diffusion)[1]
end

u0 = [1f0] |> Lux.gpu

tspan = (0f0, 1f0)
datasize = 10

solver = EulerHeun()
sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())

function loss(ps)
    problem = SDEProblem(drift, diffusion, u0, tspan, ps)
    solution = solve(problem, solver; sensealg=sensealg, saveat=collect(range(tspan[1], tspan[end], datasize)), dt=(tspan[end] / datasize))
    return sum(vec(solution |> Lux.gpu))
end

println(loss(ps_))
println(Zygote.gradient(ps -> loss(ps), ps_))
