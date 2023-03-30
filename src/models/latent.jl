using Flux, Distributions, InformationGeometry, Functors, ChainRulesCore
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

function sample_posterior(n::LatentSDE, timeseries; seed=nothing)
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
        prob = SDEProblem{false}(dudt_posterior,dudw_diffusion,z0,n.tspan,ComponentArray(Functors.functor(n)[1]),seed=seed)
    else
        prob = SDEProblem{false}(dudt_posterior,dudw_diffusion,z0,n.tspan,ComponentArray(Functors.functor(n)[1]))
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
    
    tsmatrix = hcat([reshape(ts.u, 1, 1, :) for ts in timeseries]...)
    
    Flux.reset!(n.encoder)
    context = n.encoder_re(n.encoder_p)(tsmatrix)

    initialdists_prior = get_distributions(n.initial_prior_re, ps.initial_prior_p, [1e0])
    initialdists_posterior = get_distributions(n.initial_posterior_re, n.initial_posterior_p, context[:, :, 1])
    
    initialdists_kl = hcat([reshape([KullbackLeibler(a, b) for (a, b) in zip(initialdists_prior, initialdists_posterior[:, batch])], :, 1) for batch in eachindex(timeseries)]...)

    z0 = hcat([reshape([x.μ + eps[1, batch] * x.σ for x in initialdists_posterior[:, batch]], :, 1) for batch in eachindex(timeseries)]...)

    augmented_z0 = vcat(z0, zeros(1, length(z0[1, :])))

    augmented_drift = function(u, p, t)
        timedctx = context[:, :, min(searchsortedfirst(timeseries[1].t, t), length(context[1, 1, :]))]
        # Remove augmented term from input
        u = u[1:end-1, :]

        posterior_net_input = hcat([vcat(u[:, batch], vcat(timedctx[:, batch]...)) for batch in eachindex(timeseries)]...)

        prior = n.drift_prior_re(p.drift_prior_p)(u)
        posterior = n.drift_posterior_re(p.drift_posterior_p)(posterior_net_input)
        diffusion = hcat([vcat([n.diffusion_re(p.diffusion_p)[i]([u[i, batch]]) for i in eachindex(u[:, batch])]...) for batch in eachindex(timeseries)]...)

        # from https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
        function stable_divide(a, b, eps=1e-7)
            b = map(x -> abs(x) <= eps ? eps * sign(x) : x, b)
            a ./ b
        end

        u_term = stable_divide(posterior .- prior, diffusion)
        augmented_term = 0.5e0 * sum(abs2, u_term; dims=[1])

        return vcat(posterior, augmented_term)
    end
    augmented_diffusion = function(u, p, t)
        u = u[1:end-1, :]
        diffusion = hcat([vcat([n.diffusion_re(p.diffusion_p)[i]([u[i, batch]]) for i in eachindex(u[:, batch])]...) for batch in eachindex(timeseries)]...)
        return vcat(diffusion, zeros(1, length(u[1, :])))
    end

    if seed !== nothing
        prob = SDEProblem{false}(augmented_drift,augmented_diffusion,augmented_z0,n.tspan,ps,seed=seed)
    else
        prob = SDEProblem{false}(augmented_drift,augmented_diffusion,augmented_z0,n.tspan,ps)
    end
    # sense = ForwardDiffSensitivity()
    sense = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())

    # sense = BacksolveAdjoint(autojacvec=ZygoteVJP())
    solution = solve(prob,n.args...;sensealg=sense,n.kwargs...)

    posterior = cat([u[1 : end - 1, :] for u in solution.u]..., dims=3)
    logterm = cat([u[end : end, :] for u in solution.u]..., dims=3)
    kl_divergence = 10.0 * sum(initialdists_kl, dims=1) .+ logterm[:, :, end]

    projected_ts = cat([n.projector_re(ps.projector_p)(x) for x in eachslice(posterior, dims=3)]..., dims=3)
    
    likelihoods = sum([loglikelihood(Laplace(y, 0.05), x) for (x,y) in zip(tsmatrix, projected_ts)], dims=3)[:, :, 1]
        
    return posterior, projected_ts, logterm, kl_divergence, likelihoods
end

function loss(n::LatentSDE, ps, timeseries, beta; seed=nothing)
    posterior, projected_ts, logterm, kl_divergence, distance = pass(n, ps, timeseries, seed=seed)
    return -distance .+ (beta * kl_divergence)
end
