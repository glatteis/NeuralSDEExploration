using Flux
using DifferentialEquations
using Functors
using Plots
using ComponentArrays

@testset "Latent SDE" begin
    initial_prior = Flux.Dense(1 => 2; bias=false, init=zeros)
    initial_posterior = Flux.Dense(1 => 2; bias=false, init=zeros)
    drift_prior = Flux.Scale([1.0], false)
    drift_posterior = Flux.Dense(ones(1, 2), false)
    diffusion = [Flux.Dense(1 => 1, bias=[0.0], init=zeros)]
    encoder = Flux.RNN(1 => 1, (x) -> 1.0; init=ones)
    projector = Flux.Scale([1.0], false)

    tspan = (0.0, 5.0)
    datasize = 50

    latent_sde = LatentSDE(
        initial_prior,
        initial_posterior,
        drift_prior,
        drift_posterior,
        diffusion,
        encoder,
        projector,
        tspan,
        EulerHeun();
        saveat=range(tspan[1], tspan[end], datasize),
        dt=(tspan[end]/datasize),
    )
    ps_, re = Functors.functor(latent_sde)
    ps = ComponentArray(ps_)

    input = (t=range(tspan[1],tspan[end],datasize),u=repeat([1.0], datasize))
    
    seed = 0

    posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, input, seed=seed)
    
    p = plot(posterior_latent)
    savefig(p, "posterior_latent.pdf")
    
    println(kl_divergence_)
end
