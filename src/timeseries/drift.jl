export OrnsteinUhlenbeck

struct Drift <: TimeseriesModel
    constant
    exp
end

drift(u, d::Drift, t) = d.constant + u * d.exp
diffusion(u, d::Drift, t) = 0.0

function series(d::Drift, u0s, tspan, datasize; seed=rand(UInt32), noise=(seed) -> nothing, kwargs...)
    t = range(tspan[1], tspan[2], length=datasize)
    Timeseries([solve(SDEProblem(drift, diffusion, u0, tspan, d; seed=seed+i, noise=noise(seed+i), kwargs...), EulerHeun(); dt=tspan[2]/(datasize*2), saveat=t) for (i, u0) in enumerate(u0s)])
end
