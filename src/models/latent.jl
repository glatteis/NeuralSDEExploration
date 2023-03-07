using LuxCore, Distributions, InformationGeometry
export LatentSDE, sample_prior

struct LatentSDE{N1,N2,N3,N4,N5,N6,N7,P,T,A,K} <: LuxCore.AbstractExplicitContainerLayer{(:initial_prior, :initial_posterior, :drift_prior, :drift_posterior, :diffusion, :encoder, :projector,)}
    initial_prior::N1
    initial_posterior::N2
    drift_prior::N3
    drift_posterior::N4
    diffusion::N5
    encoder::N6
    projector::N7
    p::P
    tspan::T
    args::A
    kwargs::K
end

function LatentSDE(initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder, projector, tspan, args...; p=nothing, kwargs...)
    LatentSDE{typeof(initial_prior), typeof(initial_posterior), typeof(drift_prior),typeof(drift_posterior),typeof(diffusion),typeof(encoder),typeof(projector),typeof(p),typeof(tspan),typeof(args),typeof(kwargs)}(
        initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder, projector, p, tspan, args, kwargs
    )
end

function encode(n::LatentSDE, timeseries, ps, st)
    st_ = Lux.update_state(st.encoder, :carry, nothing)
    result = []
    for x in timeseries.u
        y, st_ = Lux.apply(n.encoder, reshape([x], 1, 1), ps.encoder, st_)
        push!(result, y)
    end
    return result
	# hcat([n.encoder(reshape([u], 1, 1), ps.encoder, st.encoder)[1] for u in timeseries.u]...)
end

function get_distributions(model, context, ps, st)
    normsandvars, st_ = model(context, ps, st)
    return [Normal(norm, exp(0.5 * var)) for (norm, var) in collect(Iterators.partition(normsandvars, 2))]
end

function sample_prior(n::LatentSDE, ps, st; eps=nothing)
    if eps === nothing
        eps = only(rand(Normal(0, 1), 1))
    end

    dudt_prior = function(u, p, t)
        net, st_ = n.drift_prior(u, ps.drift_prior, st.drift_prior)
        return net
    end

    dudw_diffusion = function(u, p, t)
        net, st_ = n.diffusion(u, ps.diffusion, st.diffusion)
        return net
    end

    initialdists = get_distributions(n.initial_prior, [], ps.initial_prior, st.initial_prior)
    z0 = [x.μ + eps * x.σ for x in initialdists]
    
    prob = SDEProblem{false}(dudt_prior,dudw_diffusion,z0,n.tspan,n.p,noise=WienerProcess(0.0, 0.0))
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    return solve(prob,n.args...;sensealg=sense,n.kwargs...), st
end

function sample_posterior(n::LatentSDE, timeseries, ps, st; eps=nothing)
    if eps === nothing
        eps = only(rand(Normal(0, 1), 1))
    end

    context = encode(n, timeseries, ps, st)
    initialdists = get_distributions(n.initial_posterior, context[1], ps.initial_posterior, st.initial_posterior)
    z0 = [x.μ + eps * x.σ for x in initialdists]

    dudt_posterior = function(u, p, t)
        timedctx = context[min(searchsortedfirst(timeseries.t, t), length(context))]
        net_input = vcat(u, vcat(timedctx...))
        net, st_ = n.drift_posterior(net_input, ps.drift_posterior, st.drift_posterior)
        return net
    end

    dudw_diffusion = function(u, p, t)
        net, st_ = n.diffusion(u, ps.diffusion, st.diffusion)
        return net
    end

    prob = SDEProblem{false}(dudt_posterior,dudw_diffusion,z0,n.tspan,n.p,noise=WienerProcess(0.0, 0.0))
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    return solve(prob,n.args...;sensealg=sense,n.kwargs...), st
end


function pass(n::LatentSDE, timeseries, ps, st; eps=nothing)
    if eps === nothing
        eps = only(rand(Normal(0, 1), 1))
    end
    
    context = encode(n, timeseries, ps, st)

    initialdists_prior = get_distributions(n.initial_prior, [], ps.initial_prior, st.initial_prior)
    initialdists_posterior = get_distributions(n.initial_posterior, context[1], ps.initial_posterior, st.initial_posterior)
    initialdists_kl = [KullbackLeibler(a, b) for (a, b) in zip(initialdists_prior, initialdists_posterior)]

    z0 = [x.μ + eps * x.σ for x in initialdists_posterior]

    augmented_z0 = hcat(z0, zeros32(length(z0)))

    augmented_drift = function(u, p, t)
        timedctx = context[min(searchsortedfirst(timeseries.t, t), length(context))]
        # We are only outputting the augmented term, remove it for input
        u = u[:, 1]

        posterior_net_input = vcat(u, vcat(timedctx...))
        
        prior, st_ = n.drift_prior(u, ps.drift_prior, st.drift_prior)
        posterior, st_ = n.drift_posterior(posterior_net_input, ps.drift_posterior, st.drift_posterior)
        diffusion, st_ = n.diffusion(u, ps.diffusion, st.diffusion)
        
        u_term = (prior .- posterior) ./ diffusion
        augmented_term = 0.5 .* (u_term .^2)
        return hcat(posterior, augmented_term)
    end
    augmented_diffusion = function(u, p, t)
        u = u[:, 1]
        diffusion, st_ = n.diffusion(u, ps.diffusion, st.diffusion)
        return hcat(diffusion, zeros32(length(diffusion)))
    end

    prob = SDEProblem{false}(augmented_drift,augmented_diffusion,augmented_z0,n.tspan,nothing,noise=WienerProcess(0.0, 0.0))
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solution = solve(prob,n.args...;sensealg=sense,n.kwargs...)

    posterior = [u[:, 1] for u in solution.u]
    logterm = [u[:, 2] for u in solution.u]
    kl_divergence = initialdists_kl .+ reduce(.+, logterm)
    
    return posterior, logterm, kl_divergence
end

function loss(n::LatentSDE, timeseries, ps, st; eps=nothing)
end
