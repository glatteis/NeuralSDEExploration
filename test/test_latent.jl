using Lux
using DifferentialEquations
using Functors
using Plots
using ComponentArrays
using Zygote
using Random
using FiniteDiff
using SciMLSensitivity
using DiffEqNoiseProcess
using Random123
using JLD2

@testset "Latent SDE" begin
    solver = EulerHeun()
    tspan = (0f0, 5f0)
    datasize = 50
    
    rng = Xoshiro()

    latent_sde = StandardLatentSDE(solver, tspan, datasize, prior_size=2, posterior_size=2, diffusion_size=2)
    ps_, st = Lux.setup(rng, latent_sde)
    ps = ComponentArray{Float32}(ps_)

    input1 = (t=collect(range(tspan[1],tspan[end],datasize)),u=collect(repeat([1f0], datasize)))
    input2 = (t=collect(range(tspan[1],tspan[end],datasize)),u=collect(repeat([0.5f0], datasize)))
    
    seed = 1

    posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde([input1, input2], ps, st, seed=seed)
    
    # p = plot(posterior_latent)
    # savefig(p, "posterior_latent.pdf")
    # 
    sense = BacksolveAdjoint(autojacvec=ZygoteVJP(), checkpointing=true)
    noise = function(seed)
        rng_tree = Xoshiro(seed)
        VirtualBrownianTree(-5f0, 0f0, tend=tspan[end]+5f0; rng=Threefry4x((rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32))))
    end
    
    function loss(ps)
        sum(NeuralSDEExploration.loss(latent_sde, [input1, input1, input2, input2], ps, st, 1f0; seed=seed, sense=sense))
    end
    
    @test loss(ps) == loss(ps)
    
    grads_zygote_1 = Zygote.gradient(loss, ps)[1]
    @time grads_zygote_2 = Zygote.gradient(loss, ps)[1]
    grads_finitediff_1 = FiniteDiff.finite_difference_gradient(loss, ps) 
    @time grads_finitediff_2 = FiniteDiff.finite_difference_gradient(loss, ps) 

    @test grads_zygote_1 == grads_zygote_2
    @test grads_finitediff_1 == grads_finitediff_2
    @test isapprox(grads_zygote_1, grads_finitediff_1, rtol=0.05)
    println("Please check the grads manually:")
    println(grads_zygote_1)
    println(grads_finitediff_1)
end

# @testset "Exported Latent SDE" begin
#     dict = load("14.jld")["data"]

#     latent_sde = dict["latent_sde"]

#     ps = dict["ps"]

#     st = dict["st"]

#     timeseries = dict["timeseries"]
    
#     minibatch = timeseries[1:1]
    
#     seed = 1

#     sense = BacksolveAdjoint(autojacvec=ZygoteVJP(), checkpointing=true)
#     noise = function(seed)
#         rng_tree = Xoshiro(seed)
#         VirtualBrownianTree(-5f0, 0f0, tend=tspan[end]+5f0; rng=Threefry4x((rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32))))
#     end
    
#     function loss(ps)
#         sum(NeuralSDEExploration.loss(latent_sde, minibatch, ps, st, 1f0; seed=seed, sense=sense))
#     end
    
#     @test loss(ps) == loss(ps)
#     grads_zygote = Zygote.gradient(loss, ps)[1]
#     grads_finitediff = FiniteDiff.finite_difference_gradient(loss, ps) 
#     @test isapprox(grads_zygote, grads_finitediff, rtol=0.05)
#     println("Please check the grads manually:")
#     println(grads_zygote)
#     println(grads_finitediff)
# end