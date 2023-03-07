export FitzHughNagumoModel

struct FitzHughNagumoModel <: Timeseries
    b
    ɑ₁
    ɑ₃
    c
    σ
    β
end

FitzHughNagumoModel() = FitzHughNagumoModel(0.04, 2.23, 0.82, -6.98, 4.46, -1.51)

drift(u, fhn::FitzHughNagumoModel, t) = [driftx(u, fhn, t), drifty(u, fhn, t)]
driftx(u, fhn::FitzHughNagumoModel, t) = fhn.ɑ₁ * u[1] - fhn.ɑ₃ * u[1]^3 + fhn.b * u[2]
drifty(u, fhn::FitzHughNagumoModel, t) = tan(fhn.β) * u[2] - u[1] + fhn.c
diffusion(u, fhn::FitzHughNagumoModel, t) = [fhn.σ 0; 0 0] * u

function series(fhn::FitzHughNagumoModel, u0s, tspan, datasize)
    t = range(tspan[1], tspan[2], length=datasize)
    [solve(SDEProblem(drift, diffusion, u0, tspan, fhn, reltol=0.01), saveat=t) for u0 in u0s]
end

ylabel(fhn::FitzHughNagumoModel) = "ẟ¹⁸O anomaly"
