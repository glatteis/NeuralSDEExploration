using Lux, LuxCore, Distributions, InformationGeometry, Functors, ChainRulesCore, DifferentialEquations
export LatentSDE, sample_prior

struct LatentSDE{N1,N2,N3,N4,N5,N6,N7,S,T,D,TD,K} <: LuxCore.AbstractExplicitContainerLayer{(:initial_prior, :initial_posterior, :drift_prior, :drift_posterior, :diffusion, :encoder, :projector,)}
    initial_prior::N1
    initial_posterior::N2
    drift_prior::N3
    drift_posterior::N4
    diffusion::N5
    encoder::N6
    projector::N7
    solver::S
    tspan::T
    datasize::D
    timedependent::TD
    kwargs::K
end

function LatentSDE(initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder, projector, solver, tspan, datasize; timedependent=false, kwargs...)
    models = [initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder, projector]
    LatentSDE{
        [typeof(x) for x in models]...,
        typeof(solver),typeof(tspan),typeof(datasize),typeof(timedependent),typeof(kwargs)
    }(
        models...,
        solver, tspan, datasize, timedependent, kwargs
    )
end

function get_distributions(model, model_p, st, context)
    normsandvars, _ = model(context, model_p, st)
    # batches are on the second dimension
    batch_indices = eachindex(context[1, :])
    # output ordered like [norm, var, norm, var, ...]
    halfindices = 1:Int(length(normsandvars[:, 1]) / 2)

    return hcat([reshape([Normal{Float32}(normsandvars[2*i-1, j], exp(0.5f0 * normsandvars[2*i, j])) for i in halfindices], :, 1) for j in batch_indices]...)
end

function sample_prior(n::LatentSDE, ps, st; b=1, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    eps = reshape([only(rand(Normal{Float32}(0.0f0, 1.0f0), 1)) for i in 1:b], 1, :)

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
        if seed !== nothing
            return SDEProblem{false}(dudt_prior, dudw_diffusion, z0[:, batch], n.tspan, ps, seed=seed + batch)
        else
            return SDEProblem{false}(dudt_prior, dudw_diffusion, z0[:, batch], n.tspan, ps)
        end
    end

    ensemble = EnsembleProblem(nothing, output_func=(sol, i) -> (sol, false), prob_func=prob_func)

    return solve(ensemble, n.solver, trajectories=b; saveat=range(n.tspan[1], n.tspan[end], n.datasize), dt=(n.tspan[end] / n.datasize), n.kwargs...)
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
    likelihood_scale=0.05f0
)
    # We are using matrices with the following dimensions:
    # 1 = latent space dimension
    # 2 = batch number
    # 3 = time step
    eps = ChainRulesCore.ignore_derivatives() do
        if seed !== nothing
            Random.seed!(seed)
        end
        reshape([only(rand(Normal{Float32}(0.0f0, 1.0f0), 1)) for i in eachindex(timeseries)], 1, :)
    end

    tsmatrix = reduce(hcat, [reshape(ts.u, 1, 1, :) for ts in timeseries])

    timecat(x, y) = cat(x, y; dims=3)

    # Lux recurrent uses batches / time the other way around...
    # time: dimension 3 => dimension 2
    # batches: dimension 2 => dimension 3
    tsmatrix_flipped = permutedims(tsmatrix, (1, 3, 2))

    # ## !! LUX.JL BUG WORKAROUND !! (fixed in Lux.jl v0.4.51)
    # tsmatrix_flipped_wrong = hcat(tsmatrix_flipped[:, 1:1, :], reverse(tsmatrix_flipped[:, 2:end, :], dims=2))

    context_flipped = n.encoder(tsmatrix_flipped, ps.encoder, st.encoder)[1]

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
            timedctx = context[:, batch, time_index]
            
            # The posterior gets u and the context as information
            posterior_net_input = vcat(u, timedctx)

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
        if seed !== nothing
            return SDEProblem{false}(augmented_drift(batch), augmented_diffusion(batch), augmented_z0[:, batch], n.tspan, ps, seed=seed + Int(batch), noise=noise(seed))
        else
            return SDEProblem{false}(augmented_drift(batch), augmented_diffusion(batch), augmented_z0[:, batch], n.tspan, ps, noise=noise(nothing))
        end
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

function loss(n::LatentSDE, ps, timeseries, st, beta; kwargs...)
    posterior, projected_ts, logterm, kl_divergence, distance = pass(n, ps, timeseries, st; kwargs...)
    return -distance .+ (beta * kl_divergence)
end
