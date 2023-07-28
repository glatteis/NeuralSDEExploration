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

    input1 = (t=collect(range(tspan[1],tspan[end],datasize)),u=collect(range(0f0, 1f0, datasize)))
    input2 = (t=collect(range(tspan[1],tspan[end],datasize)),u=collect(range(2f0, 1f0, datasize)))
    
    seed = rand(rng, UInt32)

    posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(Timeseries([input1, input2, input2, input2]), ps, st, seed=seed)
    
    # p = plot(posterior_latent)
    # savefig(p, "posterior_latent.pdf")
    # 
    sense = BacksolveAdjoint(autojacvec=ZygoteVJP(), checkpointing=false)
    noise = function(seed, noise_size)
        rng_tree = Xoshiro(seed)
        VirtualBrownianTree(-5f0, fill(0f0, noise_size), tend=tspan[end]+5f0; rng=Threefry4x((rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32))))
    end
    
    input = Timeseries([input1, input1, input2, input2])
    function loss(ps)
        sum(NeuralSDEExploration.loss(latent_sde, input, ps, st, 1f0; seed=seed, sense=sense, noise=noise))
    end
    
    @test loss(ps) == loss(ps)
    
    grads_zygote_1 = Zygote.gradient(loss, ps)[1]
    @time grads_zygote_2 = Zygote.gradient(loss, ps)[1]
    grads_finitediff_1 = FiniteDiff.finite_difference_gradient(loss, ps) 
    @time grads_finitediff_2 = FiniteDiff.finite_difference_gradient(loss, ps) 

    @test grads_zygote_1 == grads_zygote_2
    @test grads_finitediff_1 == grads_finitediff_2
    outliers = findall(x -> !isapprox(x[1], x[2], rtol=1.0), collect(zip(vec(grads_zygote_1), vec(grads_finitediff_1))))
    
    println(outliers)
    println(grads_zygote_1[outliers])
    println(grads_finitediff_1[outliers])
    println(labels(grads_zygote_1)[outliers])
    println(labels(grads_finitediff_1)[outliers])
    println("Please check the grads manually:")
    println(grads_zygote_1)
    println(grads_finitediff_1)
    @test length(outliers) < length(ps) / 10
end

@testset "Exported Latent SDE" begin
    dict = load("11.jld")["data"]

    latent_sde = dict["latent_sde"]

    ps = dict["ps"]

    st = dict["st"]

    timeseries = Timeseries(dict["timeseries"])
    
    tspan = dict["tspan_model"]
    
    rng = Xoshiro()
    
    seed = rand(rng, UInt32)
    
    new_tspan = (0f0, range(tspan[1], tspan[2], latent_sde.datasize)[2])
    
    minibatch = select_tspan(new_tspan, select_ts(1:10, timeseries))

    latent_sde.datasize = length(minibatch.t)
    latent_sde.tspan = new_tspan

    hidden_dims = latent_sde.initial_prior.out_dims ÷ 2
    backsolve = BacksolveAdjoint(autojacvec=ZygoteVJP())
    interpolating = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    tree = function(seed, noise_size)
        rng_tree = Xoshiro(seed)
        VirtualBrownianTree(-3f0, fill(0f0, noise_size), tend=tspan[2]*2f0; tree_depth=0, rng=Threefry4x((rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32))))
    end
    
    function loss(ps, sense, noise)
        sum(NeuralSDEExploration.loss(latent_sde, minibatch, ps, st, 1f0; seed=seed, sense=sense, noise=noise))
    end
    
    @test loss(ps, nothing, (seed, size) -> nothing) == loss(ps, nothing, (seed, size) -> nothing)
    grads_backsolve = Zygote.gradient((x) -> loss(x, backsolve, tree), ps)[1]
    grads_finitediff_1 = FiniteDiff.finite_difference_gradient((x) -> loss(x, nothing, tree), ps)
    grads_interpolating = Zygote.gradient((x) -> loss(x, interpolating, (seed, size) -> nothing), ps)[1]
    grads_finitediff_2 = FiniteDiff.finite_difference_gradient((x) -> loss(x, nothing, (seed, size) -> nothing), ps) 

    outliers1 = findall(x -> !isapprox(x[1], x[2], atol=0.01, rtol=1.0), collect(zip(vec(grads_finitediff_1), vec(grads_backsolve))))
    outliers2 = findall(x -> !isapprox(x[1], x[2], atol=0.01, rtol=1.0), collect(zip(vec(grads_finitediff_2), vec(grads_interpolating))))

    println(outliers1)
    println(outliers2)
    println(labels(grads_finitediff_1)[outliers1])
    println(labels(grads_finitediff_2)[outliers2])
    println(grads_finitediff_1[outliers1])
    println(grads_finitediff_2[outliers2])
    @test length(outliers1) < length(ps) / 10
    @test length(outliers2) < length(ps) / 10
end
