using Flux
using DifferentialEquations
using Functors
using Plots
using ComponentArrays
using Zygote

@testset "Latent SDE" begin
    initial_prior = Flux.Dense(1 => 4; bias=false, init=zeros) |> f64
    initial_posterior = Flux.Dense(1 => 4; bias=false, init=zeros) |> f64
    drift_prior = Flux.Scale([1.0, 1.0], false) |> f64
    drift_posterior = Flux.Dense(ones(2, 3), false) |> f64
    diffusion = [Flux.Dense(1 => 1, bias=[0.01], init=zeros) |> f64 for i in 1:2]
    encoder = Flux.RNN(1 => 1, (x) -> 1.0; init=ones) |> f64
    projector = Flux.Dense(2 => 1; bias=false, init=ones) |> f64

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

    posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = NeuralSDEExploration.pass(latent_sde, ps, [input, input], seed=seed)
    
    # p = plot(posterior_latent)
    # savefig(p, "posterior_latent.pdf")
    
    println(kl_divergence_)

    function loss(ps)
        sum(NeuralSDEExploration.loss(latent_sde, ps, [input, input, input, input, input], 30.0, seed=seed))
    end
    
    grads = Zygote.gradient(loss, ps)[1]
    
    println(grads)
end
