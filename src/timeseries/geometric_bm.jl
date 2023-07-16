export GeometricBM

struct GeometricBM <: TimeseriesModel
    mu
    sigma
end

GeometricBM() = GeometricBM(1.0, 0.5)

drift(u, ou::GeometricBM, t) = ou.mu * u
diffusion(u, ou::GeometricBM, t) = ou.sigma * u

function series(ebm::GeometricBM, u0s, tspan, datasize; seed=rand(UInt32), noise=(seed) -> nothing, kwargs...)
    t = range(tspan[1], tspan[2], length=datasize)
    Timeseries([solve(SDEProblem(drift, diffusion, u0, tspan, ebm; seed=seed+i, noise=noise(seed+i), kwargs...), EulerHeun(); dt=tspan[2]/(datasize*2), saveat=t) for (i, u0) in enumerate(u0s)])
end
