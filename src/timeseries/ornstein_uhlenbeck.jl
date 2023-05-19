export OrnsteinUhlenbeck

struct OrnsteinUhlenbeck <: Timeseries
    mu
    sigma
    psi
end

OrnsteinUhlenbeck() = OrnsteinUhlenbeck(0.02, 0.1, 0.4)

drift(u, ou::OrnsteinUhlenbeck, t) = (ou.mu * t - ou.sigma * u)
diffusion(u, ou::OrnsteinUhlenbeck, t) = ou.psi

function series(ebm::OrnsteinUhlenbeck, u0s, tspan, datasize; seed=nothing)
    t = range(tspan[1], tspan[2], length=datasize)
    [solve(SDEProblem(drift, diffusion, u0, tspan, ebm; seed=seed+i), EulerHeun(); dt=tspan[2]/(datasize*2), saveat=t) for (i, u0) in enumerate(u0s)]
end