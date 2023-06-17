export FitzHughNagumoModel

struct FitzHughNagumoModel <: TimeseriesModel
    b
    ɑ₁
    ɑ₃
    c
    σx
    σy
    β
end

FitzHughNagumoModel() = FitzHughNagumoModel(0.04, 2.23, 0.82, -6.98, 4.46, 0.0, -1.51)
FitzHughNagumoModelGamma() = FitzHughNagumoModel(2.55, 0.63, 2.71, 0.22, 4.8, 11.08, -0.67)

drift(u, fhn::FitzHughNagumoModel, t) = [driftx(u, fhn, t), drifty(u, fhn, t)]
driftx(u, fhn::FitzHughNagumoModel, t) = fhn.ɑ₁ * u[1] - fhn.ɑ₃ * u[1]^3 + fhn.b * u[2]
drifty(u, fhn::FitzHughNagumoModel, t) = tan(fhn.β) * u[2] - u[1] + fhn.c
diffusion(u, fhn::FitzHughNagumoModel, t) = [fhn.σx, fhn.σy]

function series(fhn::FitzHughNagumoModel, u0s, tspan, datasize; seed=rand(UInt32), noise=(seed) -> nothing, kwargs...)
    t = range(tspan[1], tspan[2], length=datasize)
    Timeseries([solve(SDEProblem(drift, diffusion, u0, tspan, fhn, reltol=0.01, seed=seed+i, maxiters=1_000_000, noise=noise(seed+i), kwargs...), saveat=t) for (i, u0) in enumerate(u0s)])
end

ylabel(fhn::FitzHughNagumoModel) = "ẟ¹⁸O anomaly"
