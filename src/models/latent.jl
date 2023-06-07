using Lux, LuxCore, Distributions, InformationGeometry, Functors, ChainRulesCore, DifferentialEquations
export LatentSDE, StandardLatentSDE, sample_prior

struct LatentSDE{N1,N2,N3,N4,N5,N6,N7,N8,S,T,D,TD,K} <: LuxCore.AbstractExplicitContainerLayer{(:initial_prior, :initial_posterior, :drift_prior, :drift_posterior, :diffusion, :encoder_recurrent, :encoder_net, :projector,)}
    initial_prior::N1
    initial_posterior::N2
    drift_prior::N3
    drift_posterior::N4
    diffusion::N5
    encoder_recurrent::N6
    encoder_net::N7
    projector::N8
    solver::S
    tspan::T
    datasize::D
    timedependent::TD
    kwargs::K
end

function LatentSDE(initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder_recurrent, encoder_net, projector, solver, tspan, datasize; timedependent=false, kwargs...)
    models = [initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder_recurrent, encoder_net, projector]
    LatentSDE{
        [typeof(x) for x in models]...,
        typeof(solver),typeof(tspan),typeof(datasize),typeof(timedependent),typeof(kwargs)
    }(
        models...,
        solver, tspan, datasize, timedependent, kwargs
    )
end

"""
    StandardLatentSDE(solver, tspan, datasize)
    
Constructs a "standard" latent sde - so you don't need to construct all of the neural nets.

## Arguments

- `solver`: SDE solver.
- `tspan`: SDE timespan.
- `datasize`: SDE timesteps.

## Keyword Arguments

- `data_dims`: Dimension of the data space.
- `latent_dims`: Dimension of the latent space.
- `prior_size`: Size of the prior net hidden layers.
- `posterior_size`: Size of the posterior net hidden layers.
- `diffusion_size`: Size of the diffusion hidden layers. The diffusion is not
    super configurable. Please just swap it out if you need a different one. Prior
- `depth`: Depth of the prior and posterior nets.
- `hidden_activation`: Activation of the hidden layers of the neural nets.
- `final_activation`: Activation of the final layers of the neural nets. There's a scale layer after them, so it's fine to take a bounded activation.
- `rnn_size`: Size of the RNN's output. There's a neural net after the RNN that goes to `context_size`.
- `context_size`: Size of the context vector.
- `timedependent`: Whether time is an input to prior drift, posterior drift and diffusion.
"""
# TODO Make Networks replacable here
function StandardLatentSDE(solver, tspan, datasize;
        data_dims=1,
        latent_dims=2,
        prior_size=128,
        posterior_size=128,
        diffusion_size=16,
        depth=1,
        rnn_size=2,
        context_size=2,
        hidden_activation=tanh,
        final_activation=tanh,
        timedependent=false,
        kwargs...
    )
    encoder_recurrent = Lux.Recurrence(Lux.GRUCell(data_dims => rnn_size); return_sequence=true)

    encoder_net = Lux.Dense(rnn_size => context_size)

    # The initial_posterior net is the posterior for the initial state. It
    # takes the context and outputs a mean and standard devation for the
    # position zero of the posterior. The initial_prior is a fixed gaussian
    # distribution.
    initial_posterior = Lux.Dense(context_size => latent_dims + latent_dims; init_weight=zeros)
    initial_prior = Lux.Dense(1 => latent_dims + latent_dims) 
    
    in_dims = latent_dims + (timedependent ? 1 : 0)
    # Drift of prior. This is just an SDE drift in the latent space
    drift_prior = Lux.Chain(
        Lux.Dense(in_dims => prior_size, hidden_activation),
        repeat([Lux.Dense(prior_size => prior_size, hidden_activation)], depth)...,
        Lux.Dense(prior_size => latent_dims, final_activation),
        Lux.Scale(latent_dims)
    )
    # Drift of posterior. This is the term of an SDE when fed with the context.
    drift_posterior = Lux.Chain(
        Lux.Dense(in_dims + context_size => posterior_size, hidden_activation),
        repeat([Lux.Dense(posterior_size => posterior_size, hidden_activation)], depth)...,
        Lux.Dense(posterior_size => latent_dims, final_activation),
        Lux.Scale(latent_dims)
    )
    # Prior and posterior share the same diffusion (they are not actually evaluated
    # seperately while training, only their KL divergence). This is a diagonal
    # diffusion, i.e. every term in the latent space has its own independent
    # Wiener process.
    diffusion = Lux.Parallel(nothing, [
            Lux.Chain(Lux.Dense((timedependent ? 2 : 1) => diffusion_size, tanh),
            Lux.Dense(diffusion_size => 1, Lux.sigmoid_fast),
            Lux.Scale(1, init_weight=ones, init_bias=ones))
        for i in 1:latent_dims]...
    )
    # The projector will transform the latent space back into data space.
    projector = Lux.Dense(latent_dims => data_dims)
    
    return LatentSDE(initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder_recurrent, encoder_net, projector, solver, tspan, datasize; kwargs...)
end

function get_distributions(model, model_p, st, context)
    normsandvars, _ = model(context, model_p, st)
    # batches are on the second dimension
    batch_indices = eachindex(context[1, :])
    # output ordered like [norm, var, norm, var, ...]
    halfindices = 1:Int(length(normsandvars[:, 1]) / 2)

    return hcat([reshape([Normal{Float32}(normsandvars[2*i-1, j], exp(0.5f0 * normsandvars[2*i, j])) for i in halfindices], :, 1) for j in batch_indices]...)
end

function sample_prior(n::LatentSDE, ps, st; b=1, seed=nothing, noise=(seed) -> nothing, tspan=n.tspan, datasize=n.datasize)
    (eps, seed) = ChainRulesCore.ignore_derivatives() do
        if seed !== nothing
            Random.seed!(seed)
        end
        (reshape([only(rand(Normal{Float32}(0.0f0, 1.0f0), 1)) for i in 1:b], 1, :), rand(UInt32))
    end

    function dudt_prior(u, p, t) 
        time_or_empty = n.timedependent ? [t] : []
        n.drift_prior(vcat(u, time_or_empty), p.drift_prior, st.drift_prior)[1]
    end
    function dudw_diffusion(u, p, t) 
        time_or_empty = n.timedependent ? [t] : []
        reduce(vcat, n.diffusion(Tuple([vcat(x, time_or_empty) for x in u]), p.diffusion, st.diffusion)[1])
    end

    initialdists_prior = get_distributions(n.initial_prior, ps.initial_prior, st.initial_prior, [1.0f0])
    z0 = hcat([reshape([x.μ + ep * x.σ for x in initialdists_prior], :, 1) for ep in eps[1, :]]...)

    function prob_func(prob, batch, repeat)
        noise_instance = ChainRulesCore.ignore_derivatives() do
            noise(Int(floor(seed + batch)))
        end
        return SDEProblem{false}(dudt_prior, dudw_diffusion, z0[:, batch], tspan, ps, seed=seed + batch, noise=noise_instance)
    end

    ensemble = EnsembleProblem(nothing, output_func=(sol, i) -> (sol, false), prob_func=prob_func)

    return solve(ensemble, n.solver, trajectories=b; saveat=range(tspan[1], tspan[end], datasize), dt=(tspan[end] / datasize), n.kwargs...)
end

# from https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
function stable_divide(a, b, eps=1e-7)
    if any([x <= eps for x in b])
        @warn "diffusion to small"
    end
    b = map(x -> abs(x) <= eps ? eps * sign(x) : x, b)
    a ./ b
end

"""
    (n::LatentSDE)(timeseries, ps::ComponentVector, st)
    
Sample from the Latent SDE's posterior, compute the KL-divergence and the terms for the loss.

If the model's `tspan` starts earlier than the `timeseries`, the initial states are not directly scored.
Autodiff this function to train the Latent SDE.

## Arguments

- `sense`: Sensitivity Algorithm. Consult https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/
- `ensemblemode`: We use ensembles, this is the evaluation mode (serial / parallel / GPU). Consult https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/
- `seed`: Seed for simulations, we use `seed`, `seed + 1`, `seed + 2`, and so on. If no seed is provided, it's generated by the global RNG.
- `noise`: Function from `seed` to the noise used.
- `stick_landing`: Enable a broken implementation of "sticking-the-landing".
- `likelihood_dist`: Distribution for computing log-likelihoods (as a way of computing distance).
- `likelihood_scale`: Variance of `likelihood_dist`.
"""
function (n::LatentSDE)(timeseries::Vector{NamedTuple{(:t, :u), Tuple{Vector{Float32}, Vector{Float32}}}}, ps::ComponentVector, st;
    sense=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=true),
    ensemblemode=EnsembleThreads(),
    seed=nothing,
    noise=(seed) -> nothing,
    stick_landing=false,
    likelihood_dist=Normal,
    likelihood_scale=0.01f0,
    buffer_context=true, # TODO
)
    # We are using matrices with the following dimensions:
    # 1 = latent space dimension
    # 2 = batch number
    # 3 = time step
    (eps, seed) = ChainRulesCore.ignore_derivatives() do
        if seed !== nothing
            Random.seed!(seed)
        end
        (reshape([only(rand(Normal{Float32}(0.0f0, 1.0f0), 1)) for i in eachindex(timeseries)], 1, :), rand(UInt32))
    end

    tsmatrix = reduce(hcat, [reshape(ts.u, 1, 1, :) for ts in timeseries])

    timecat(x, y) = cat(x, y; dims=3)

    # Lux recurrent uses batches / time the other way around...
    # time: dimension 3 => dimension 2
    # batches: dimension 2 => dimension 3
    tsmatrix_flipped = reverse(permutedims(tsmatrix, (1, 3, 2)), dims=2)

    # ## !! LUX.JL BUG WORKAROUND !! (fixed in Lux.jl v0.4.51)
    # tsmatrix_flipped_wrong = hcat(tsmatrix_flipped[:, 1:1, :], reverse(tsmatrix_flipped[:, 2:end, :], dims=2))

    precontext_flipped = reverse(n.encoder_recurrent(tsmatrix_flipped, ps.encoder_recurrent, st.encoder_recurrent)[1])
    context_flipped = [n.encoder_net(x, ps.encoder_net, st.encoder_net)[1] for x in precontext_flipped]

    # context_flipped is now a vector of 2-dim matrices
    # latent space: dimension 1
    # batch: dimension 2
    context = reduce(timecat, context_flipped)

    initialdists_prior = get_distributions(n.initial_prior, ps.initial_prior, st.initial_prior, [1.0f0])

    initialdists_posterior = get_distributions(n.initial_posterior, ps.initial_posterior, st.initial_posterior, context[:, :, 1])

    z0 = reduce(hcat, [reshape([x.μ + eps[1, batch] * x.σ for x in initialdists_posterior[:, batch]], :, 1) for batch in eachindex(timeseries)])

    augmented_z0 = vcat(z0, zeros32(1, length(z0[1, :])))

    function augmented_drift(batch)
        function (u_in::Vector{Float32}, p::ComponentVector, t::Float32)
            # Remove augmented term from input
            u = if n.timedependent
                vcat(u_in[1:end-1], t)
            else
                u_in[1:end-1]
            end

            # Get the context for the posterior at the current time
            # initial state evolve => get the posterior at future start time
            time_index = max(1, searchsortedlast(timeseries[1].t, t))
            timedctx_check = context[:, batch, time_index] # check if the context is correct, but we need to re-compute it here...
            # tsmatrix_flipped has the batch / time indices the other way around
            # and it has the time reversed
            precontext_flipped_local = n.encoder_recurrent(tsmatrix_flipped[:, 1:end - time_index + 1, batch:batch], p.encoder_recurrent, st.encoder_recurrent)[1]
            
            timedctx = n.encoder_net(precontext_flipped_local[end], p.encoder_net, st.encoder_net)[1]
            
            @assert vec(timedctx) == vec(timedctx_check)
            
            # The posterior gets u and the context as information
            posterior_net_input = vcat(u, vec(timedctx))

            prior = n.drift_prior(u, p.drift_prior, st.drift_prior)[1]
            posterior = n.drift_posterior(posterior_net_input, p.drift_posterior, st.drift_posterior)[1]
            # The diffusion is diagonal, so a single network is invoked on each dimension
            diffusion = reduce(vcat, n.diffusion(([n.timedependent ? vcat(x, t) : [x] for x in u]...,), p.diffusion, st.diffusion)[1])

            # The augmented term for computing the KL divergence
            u_term = stable_divide(posterior .- prior, diffusion)
            augmented_term = 0.5f0 * sum(abs2, u_term; dims=[1])

            return vcat(posterior, augmented_term)
        end
    end
    function augmented_diffusion(batch)
        function (u_in::Vector{Float32}, p::ComponentVector, t::Float32)
            time_or_empty = n.timedependent ? [t] : []
            u = vcat(u_in[1:end-1], time_or_empty)
            diffusion = reduce(vcat, n.diffusion(([n.timedependent ? vcat(x, t) : [x] for x in u]...,), p.diffusion, st.diffusion)[1])
            additional_term = if stick_landing
                time_index = max(1, searchsortedlast(timeseries[1].t, t))
                timedctx = context[:, batch, time_index]
                posterior_net_input = vcat(u, timedctx)
                prior = n.drift_prior(u, p.drift_prior, st.drift_prior)[1]
                posterior = ChainRulesCore.ignore_derivatives(n.drift_posterior(posterior_net_input, p.drift_posterior, st.drift_posterior)[1])
                u_term = stable_divide(posterior .- prior, diffusion)
                sum(u_term; dims=[1])
            else
                0.0f0
            end
            return vcat(diffusion, additional_term)
        end
    end

    function prob_func(prob, batch, repeat)
        noise_instance = ChainRulesCore.ignore_derivatives() do
            noise(Int(floor(seed + batch)))
        end
        return SDEProblem{false}(augmented_drift(batch), augmented_diffusion(batch), augmented_z0[:, batch], n.tspan, ps, seed=seed + Int(batch), noise=noise_instance)
    end

    ensemble = EnsembleProblem(nothing, output_func=(sol, i) -> (sol, false), prob_func=prob_func)

    solution = solve(ensemble, n.solver, ensemblemode; trajectories=length(timeseries), sensealg=sense, saveat=range(n.tspan[1], n.tspan[end], n.datasize), dt=(n.tspan[end] / n.datasize), n.kwargs...)

    # If ts_start > 0, the timeseries starts after the latent sde, thus only score after ts_start
    # The timeseries could be irregularily sampled or have a different rate than the model, so search for appropiate points here
    # TODO: Interpolate this?
    ts_indices = [searchsortedfirst(solution[1].t, t) for t in timeseries[1].t]
    ts_start = ts_indices[1]

    posterior_latent = reduce(hcat, [reduce(timecat, [reshape(u[1:end-1], :, 1, 1) for u in batch.u]) for batch in solution.u])
    logterm = reduce(hcat, [reduce(timecat, [reshape(u[end:end], :, 1, 1) for u in batch.u]) for batch in solution.u])
    initialdists_kl = reduce(hcat, [reshape([KullbackLeibler(a, b) for (a, b) in zip(initialdists_posterior[:, batch], initialdists_prior)], :, 1) for batch in eachindex(timeseries)])
    kl_divergence = sum(initialdists_kl, dims=1) .+ logterm[:, :, end]

    projected_z0 = n.projector(z0, ps.projector, st.projector)[1]
    projected_ts = reduce(timecat, [n.projector(x, ps.projector, st.projector)[1] for x in eachslice(posterior_latent, dims=3)])

    logp(x, y) = loglikelihood(likelihood_dist(y, likelihood_scale), x)
    likelihoods_initial = if ts_start == 1
        [logp(x, y) for (x, y) in zip(tsmatrix[:, :, 1], projected_z0)]
    else
        fill(0.0f0, size(projected_z0))
    end
    likelihoods_time = sum([logp(x, y) for (x, y) in zip(tsmatrix, projected_ts[:, :, ts_indices])], dims=3)[:, :, 1]
    likelihoods = likelihoods_initial .+ likelihoods_time

    return posterior_latent, projected_ts, logterm, kl_divergence, likelihoods
end

function loss(n::LatentSDE, timeseries, ps, st, beta; kwargs...)
    posterior, projected_ts, logterm, kl_divergence, distance = n(timeseries, ps, st; kwargs...)
    return -distance .+ (beta * kl_divergence)
end
