export OrnsteinUhlenbeck

struct OrnsteinUhlenbeck <: TimeseriesModel
    mu
    sigma
    psi
end

OrnsteinUhlenbeck() = OrnsteinUhlenbeck(0.02, 0.1, 0.3)

drift(u, ou::OrnsteinUhlenbeck, t) = (ou.mu * t - ou.sigma * u)
diffusion(u, ou::OrnsteinUhlenbeck, t) = ou.psi

function series(ebm::OrnsteinUhlenbeck, u0s, tspan, datasize; seed=rand(UInt32), noise=(seed) -> nothing, kwargs...)
    t = range(tspan[1], tspan[2], length=datasize)
    Timeseries([solve(SDEProblem(drift, diffusion, u0, tspan, ebm; seed=seed+i, noise=noise(seed+i), kwargs...), EulerHeun(); dt=tspan[2]/(datasize*2), saveat=t) for (i, u0) in enumerate(u0s)])
end