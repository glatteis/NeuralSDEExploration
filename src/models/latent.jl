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
    return [Normal{Float64}(normsandvars[2*i-1], exp(0.5e0 * normsandvars[2*i])) for i in 1:Int(length(normsandvars)/2)]
end

function sample_prior(n::LatentSDE; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    eps = only(rand(Normal{Float64}(0e0, 1e0), 1))

    dudt_prior(u, p, t) = n.drift_prior_re(p.drift_prior_p)(u)
    dudw_diffusion(u, p, t) = vcat([n.diffusion_re(p.diffusion_p)[i]([u[i]]) for i in eachindex(u)]...)
    
    initialdists = get_distributions(n.initial_prior_re, n.initial_prior_p, [1e0])
    z0 = [x.μ + eps * x.σ for x in initialdists]
    
    if seed !== nothing
        prob = SDEProblem{false}(dudt_prior,dudw_diffusion,z0,n.tspan,ComponentArray(Functors.functor(n)[1]),seed=seed)
    else
        prob = SDEProblem{false}(dudt_prior,dudw_diffusion,z0,n.tspan,ComponentArray(Functors.functor(n)[1]))
    end
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    return solve(prob,n.args...;sensealg=sense,n.kwargs...)
end

function sample_posterior(n::LatentSDE, timeseries; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    eps = only(rand(Normal{Float64}(0e0, 1e0), 1))

    Flux.reset!(n.encoder)
    context = n.encoder_re(n.encoder_p)(reshape(timeseries.u, 1, 1, :))
    initialdists = get_distributions(n.initial_posterior_re, n.initial_posterior_p, context[:, 1, 1])
    z0 = [x.μ + eps * x.σ for x in initialdists]

    dudt_posterior = function(u, p, t)
        timedctx = context[:, 1, min(searchsortedfirst(timeseries.t, t), length(context[1, 1, :]))]
        net_input = vcat(u, vcat(timedctx...))
        n.drift_posterior_re(p.drift_posterior_p)(net_input)
    end
    
    dudw_diffusion(u, p, t) = vcat([n.diffusion_re(p.diffusion_p)[i]([u[i]]) for i in eachindex(u)]...)

    if seed !== nothing
        prob = SDEProblem{false}(dudt_posterior,dudw_diffusion,z0,n.tspan,ComponentArray(Functors.functor(n)[1]),seed=seed)
    else
        prob = SDEProblem{false}(dudt_posterior,dudw_diffusion,z0,n.tspan,ComponentArray(Functors.functor(n)[1]))
    end
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    return solve(prob,n.args...;sensealg=sense,n.kwargs...)
end


function pass(n::LatentSDE, ps, timeseries; seed=nothing)
    eps = ChainRulesCore.ignore_derivatives() do 
        if seed !== nothing
            Random.seed!(seed)
        end
        only(rand(Normal{Float64}(0e0, 1e0), 1))
    end
    
    Flux.reset!(n.encoder)
    context = n.encoder_re(ps.encoder_p)(reshape(timeseries.u, 1, 1, :))

    initialdists_prior = get_distributions(n.initial_prior_re, ps.initial_prior_p, [1e0])
    initialdists_posterior = get_distributions(n.initial_posterior_re, ps.initial_posterior_p, context[:, 1, 1])
    
    initialdists_kl = [KullbackLeibler(a, b) for (a, b) in zip(initialdists_prior, initialdists_posterior)]


    z0 = [x.μ + eps * x.σ for x in initialdists_posterior]

    augmented_z0 = vcat(z0, zeros32(1))

    augmented_drift = function(u, p, t)
        timedctx = context[:, 1, searchsortedlast(timeseries.t, t)]
        #timedctx = context[:, 1, 1]
        # Remove augmented term from input
        u = u[1 : end - 1]

        posterior_net_input = vcat(u, vcat(timedctx...))

        prior = n.drift_prior_re(p.drift_prior_p)(u)
        posterior = n.drift_posterior_re(p.drift_posterior_p)(posterior_net_input)
        diffusion = vcat([n.diffusion_re(p.diffusion_p)[i]([u[i]]) for i in eachindex(u)]...)

        # from https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
        function stable_divide(a, b, eps=1e-7)
            b = map(x -> abs(x) <= eps ? eps * sign(x) : x, b)
            a ./ b
        end

        u_term = stable_divide(posterior .- prior, diffusion)
        augmented_term = 0.5e0 * sum(abs2, u_term)

        return vcat(posterior, augmented_term)
    end
    augmented_diffusion = function(u, p, t)
        u = u[1 : end - 1]
        diffusion = vcat([n.diffusion_re(p.diffusion_p)[i]([u[i]]) for i in eachindex(u)]...)
        return vcat(diffusion, zeros32(1))
    end

    if seed !== nothing
        prob = SDEProblem{false}(augmented_drift,augmented_diffusion,augmented_z0,n.tspan,ps,seed=seed)
    else
        # ps = Functors.functor(n)[1]
        # p = ChainRulesCore.ignore_derivatives() do
        #     # getaxes(ComponentArray(ps))
        #     ComponentArray(ps)
        # end
        # ax = getaxes(ps)
        # @unpack p1, p2, p3, p4, p5, p6, p7 = ps
        # p = ComponentArray(ps, ax)
        prob = SDEProblem{false}(augmented_drift,augmented_diffusion,augmented_z0,n.tspan,ps)
    end
    # sense = ForwardDiffSensitivity()
    sense = InterpolatingAdjoint(autojacvec=ReverseDiffVJP())
    solution = solve(prob,n.args...;sensealg=sense,n.kwargs...)
    # solution = solve(prob,n.args...;n.kwargs...)

    posterior = [u[1 : end - 1] for u in solution.u]
    logterm = [u[end : end] for u in solution.u]
    kl_divergence = sum(initialdists_kl) + only(logterm[end])

    projected_ts = [n.projector_re(ps.projector_p)(x) for x in posterior]
    
    likelihoods = sum([loglikelihood(Laplace(y, 0.05), x) for (x,y) in zip(timeseries.u, map(only, projected_ts))])

    return posterior, projected_ts, logterm, kl_divergence, likelihoods
end

function loss(n::LatentSDE, ps, timeseries, beta; seed=nothing)
    posterior, projected_ts, logterm, kl_divergence, distance = pass(n, ps, timeseries, seed=seed)
    return -distance + beta * kl_divergence
end
