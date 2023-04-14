using Flux, Distributions, InformationGeometry, Functors, ChainRulesCore, DifferentialEquations
export LatentSDE, sample_prior

struct LatentSDE{N1,N2,N3,N4,N5,N6,N7,R1,R2,R3,R4,R5,R6,R7,P1,P2,P3,P4,P5,P6,P7,T,A,K}
    initial_prior::N1
    initial_posterior::N2
    drift_prior::N3
    drift_posterior::N4
    diffusion::N5
    encoder::N6
    projector::N7
    initial_prior_re::R1
    initial_posterior_re::R2
    drift_prior_re::R3
    drift_posterior_re::R4
    diffusion_re::R5
    encoder_re::R6
    projector_re::R7
    initial_prior_p::P1
    initial_posterior_p::P2
    drift_prior_p::P3
    drift_posterior_p::P4
    diffusion_p::P5
    encoder_p::P6
    projector_p::P7
    tspan::T
    args::A
    kwargs::K
end

function LatentSDE(initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder, projector, tspan, args...; kwargs...)
    models = [initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder, projector]
    p = []
    res = []
    for model in models
        p_, re_ = Flux.destructure(model)
        push!(p, ComponentArray(p_))
        push!(res, re_)
    end
    LatentSDE{
        [typeof(x) for x in models]...,
        [typeof(x) for x in res]...,
        [typeof(x) for x in p]...,
        typeof(tspan),typeof(args),typeof(kwargs)
    }(
        models...,
        res...,
        p...,
        tspan, args, kwargs
    )
end

@functor LatentSDE (initial_prior_p,initial_posterior_p,drift_prior_p,drift_posterior_p,diffusion_p,encoder_p,projector_p,)

function get_distributions(model_re, model_p, context)
    normsandvars = model_re(model_p)(context)
    # batches are on the second dimension
    batch_indices = eachindex(context[1, :])
    # output ordered like [norm, var, norm, var, ...]
    halfindices = 1:Int(length(normsandvars[:, 1])/2)
    
    return hcat([reshape([Normal{Float64}(normsandvars[2*i-1, j], exp(0.5e0 * normsandvars[2*i, j])) for i in halfindices], :, 1) for j in batch_indices]...)
end

function sample_prior(n::LatentSDE, ps; b=1, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    eps = reshape([only(rand(Normal{Float64}(0e0, 1e0), 1)) for i in 1:b], 1, :)

    dudt_prior(u, p, t) = n.drift_prior_re(p.drift_prior_p)(u)
    dudw_diffusion(u, p, t) = mapslices(row -> vcat([n.diffusion_re(p.diffusion_p)[i]([row[i]]) for i in eachindex(row)]...), u; dims=[1])
    
    initialdists = get_distributions(n.initial_prior_re, ps.initial_prior_p, [1e0])
    z0 = hcat([reshape([x.μ + ep * x.σ for x in initialdists], :, 1) for ep in eps[1, :]]...)

    if seed !== nothing
        prob = SDEProblem{false}(dudt_prior,dudw_diffusion,z0,n.tspan,ps,seed=seed)
    else
        prob = SDEProblem{false}(dudt_prior,dudw_diffusion,z0,n.tspan,ps)
    end
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    return solve(prob,n.args...;sensealg=sense,n.kwargs...)
end

# from https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
function stable_divide(a, b, eps=1e-7)
    b = map(x -> abs(x) <= eps ? eps * sign(x) : x, b)
    a ./ b
end

function sample_posterior(n::LatentSDE, ps, timeseries; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    eps = reshape([only(rand(Normal{Float64}(0e0, 1e0), 1)) for i in eachindex(timeseries)], 1, :)

    Flux.reset!(n.encoder)
    context = n.encoder_re(n.encoder_p)(hcat([reshape(ts.u, 1, 1, :) for ts in timeseries]...))
    initialdists = get_distributions(n.initial_posterior_re, n.initial_posterior_p, context[:, :, 1])
    z0 = hcat([reshape([x.μ + eps[1, batch] * x.σ for x in initialdists[:, batch]], :, 1) for batch in eachindex(timeseries)]...)

    dudt_posterior = function(u, p, t)
        # assumption: each timeseries in batch has the same t (fix later please)
        timedctx = context[:, :, min(searchsortedfirst(timeseries[1].t, t), length(context[1, 1, :]))]
        net_input = hcat([vcat(u[:, batch], vcat(timedctx[:, batch]...)) for batch in eachindex(timeseries)]...)
        n.drift_posterior_re(p.drift_posterior_p)(net_input)
    end
    
    dudw_diffusion(u, p, t) = mapslices(row -> vcat([n.diffusion_re(p.diffusion_p)[i]([row[i]]) for i in eachindex(row)]...), u; dims=[1])

    if seed !== nothing
        prob = SDEProblem{false}(dudt_posterior,dudw_diffusion,z0,n.tspan,ps,seed=seed)
    else
        prob = SDEProblem{false}(dudt_posterior,dudw_diffusion,z0,n.tspan,ps)
    end
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    # sense = BacksolveAdjoint(autojacvec=ZygoteVJP())
    return solve(prob,n.args...;sensealg=sense,n.kwargs...)
end

function pass(n::LatentSDE, ps, timeseries; seed=nothing)
    # We have a lot of hcats and vcats and reshapes in here. This is because we
    # are using matrices with the following dimensions:
    # 1 = latent space dimension
    # 2 = batch number
    # 3 = time step
    eps = ChainRulesCore.ignore_derivatives() do
        if seed !== nothing
            Random.seed!(seed)
        end
        reshape([only(rand(Normal{Float64}(0e0, 1e0), 1)) for i in eachindex(timeseries)], 1, :)
    end
    
    tsmatrix = reduce(hcat, [reshape(ts.u, 1, 1, :) for ts in timeseries])
    Flux.reset!(n.encoder)
    context = n.encoder_re(n.encoder_p)(tsmatrix)
    # println("context: $context")

    initialdists_prior = get_distributions(n.initial_prior_re, ps.initial_prior_p, [1e0])
    initialdists_posterior = get_distributions(n.initial_posterior_re, n.initial_posterior_p, context[:, :, 1])
    
    initialdists_kl = reduce(hcat, [reshape([KullbackLeibler(a, b) for (a, b) in zip(initialdists_posterior[:, batch], initialdists_prior)], :, 1) for batch in eachindex(timeseries)])

    z0 = reduce(hcat, [reshape([x.μ + eps[1, batch] * x.σ for x in initialdists_posterior[:, batch]], :, 1) for batch in eachindex(timeseries)])

    augmented_z0 = vcat(z0, zeros(1, length(z0[1, :])))
    
    augmented_drift = function(batch)
        return function(u, p, t)
            # Remove augmented term from input
            u = u[1:end-1]
            
            # Get the context for the posterior at the current time
            time_index = min(searchsortedfirst(timeseries[1].t, t), length(context[1, 1, :]))
            timedctx = context[:, batch, time_index]
            
            posterior_net_input = vcat(u, timedctx)
            
            prior = n.drift_prior_re(p.drift_prior_p)(u)
            posterior = n.drift_posterior_re(p.drift_posterior_p)(posterior_net_input)
            # println(p.drift_posterior_p)
            # println("$posterior_net_input => $posterior")
            diffusion = reduce(vcat, n.diffusion_re(p.diffusion_p)[i](u[i:i]) for i in eachindex(u))

            u_term = stable_divide(posterior .- prior, diffusion)
            augmented_term = 0.5e0 * sum(abs2, u_term; dims=[1])

            return_val = vcat(posterior, augmented_term)
            # println("batch $batch, time $t, index $time_index, context $timedctx, u $u, return $return_val")
            return return_val
        end
    end
    augmented_diffusion = function(u, p, t)
        u = u[1:end-1]
        diffusion = reduce(vcat, n.diffusion_re(p.diffusion_p)[i](u[i:i]) for i in eachindex(u))
        return_val = vcat(diffusion, zeros(1))
        return return_val
    end

    # println("augmented_z0: $augmented_z0")

    function prob_func(prob, batch, repeat)
        if seed !== nothing
            return SDEProblem{false}(augmented_drift(Int(batch)),augmented_diffusion,augmented_z0[:, Int(batch)],n.tspan,ps,seed=seed+Int(batch))
        else
            return SDEProblem{false}(augmented_drift(Int(batch)),augmented_diffusion,augmented_z0[:, Int(batch)],n.tspan,ps)
        end
        # DifferentialEquations.remake(prob; f = SDEFunction{false}(augmented_drift(Int(batch)),augmented_diffusion), u0 = augmented_z0[:, batch])
    end

    ensemble = EnsembleProblem(nothing, output_func=(sol, i) -> (sol, false), prob_func=prob_func)

    # sense = ForwardDiffSensitivity()
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())

    # sense = BacksolveAdjoint(autojacvec=ZygoteVJP())
    solution = solve(ensemble,n.args...,EnsembleSerial();trajectories=length(timeseries),sensealg=sense,n.kwargs...)
   
    batchcat(x, y) = cat(x, y; dims = 3)
    
    # println([u[2] for u in solution.u])

    posterior = reduce(hcat, [reduce(batchcat, [reshape(u[1:end-1], :, 1, 1) for u in batch.u]) for batch in solution.u])
    logterm = reduce(hcat, [reduce(batchcat, [reshape(u[end:end], :, 1, 1) for u in batch.u]) for batch in solution.u])
    kl_divergence = sum(initialdists_kl, dims=1) .+ logterm[:, :, end]

    projected_z0 = n.projector_re(ps.projector_p)(z0)
    projected_ts = reduce(batchcat, [n.projector_re(ps.projector_p)(x) for x in eachslice(posterior, dims=3)])
        
    logp(x, y) = loglikelihood(Normal(y, 0.05), x)
    likelihoods_initial = [logp(x, y) for (x,y) in zip(tsmatrix[:, :, 1], projected_z0)]
    likelihoods_time = sum([logp(x, y) for (x,y) in zip(tsmatrix, projected_ts)], dims=3)[:, :, 1]
    likelihoods = likelihoods_initial .+ likelihoods_time
    
    return posterior, projected_ts, logterm, kl_divergence, likelihoods
end

function loss(n::LatentSDE, ps, timeseries, beta; seed=nothing)
    posterior, projected_ts, logterm, kl_divergence, distance = pass(n, ps, timeseries, seed=seed)
    return -distance .+ (beta * kl_divergence)
    # return -distance
end
