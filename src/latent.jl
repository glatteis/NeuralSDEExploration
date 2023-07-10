export LatentSDE, StandardLatentSDE, sample_prior, sample_prior_dataspace

mutable struct LatentSDE{N1,N2,N3,N4,N5,N6,N7,N8,S,T,D,TD,K} <: LuxCore.AbstractExplicitContainerLayer{(:initial_prior, :initial_posterior, :drift_prior, :drift_posterior, :diffusion, :encoder_recurrent, :encoder_net, :projector,)}
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
- `rnn_size`: Size of the RNN's output. There's a neural net after the RNN that goes to `context_size`.
- `context_size`: Size of the context vector.
- `timedependent`: Whether time is an input to prior drift, posterior drift and diffusion.
"""
function StandardLatentSDE(solver, tspan, datasize;
        data_dims=1,
        latent_dims=2,
        prior_size=64,
        posterior_size=64,
        diffusion_size=16,
        depth=1,
        rnn_size=16,
        context_size=16,
        hidden_activation=tanh,
        timedependent=false,
        kwargs...
    )
    
    solver_kwargs = Dict(kwargs)
    networks = []

    # this function serves the following purpose:
    # - create default network if there isn't one given in the kwargs
    # - remove that kwarg from the kwargs given to the solver
    # needs to be called in the same order as args to the LatentSDE
    function create_network(name, network)
        if haskey(solver_kwargs, name)
            # network is provided by user
            user_network = pop!(solver_kwargs, name)
            push!(networks, user_network)
        else
            push!(networks, network)
        end
    end

    in_dims = latent_dims + (timedependent ? 1 : 0)

    # The initial_posterior net is the posterior for the initial state. It
    # takes the context and outputs a mean and standard devation for the
    # position zero of the posterior. The initial_prior is a fixed gaussian
    # distribution.
    create_network(:initial_prior, Lux.Dense(1 => latent_dims + latent_dims) )
    create_network(:initial_posterior, Lux.Dense(context_size => latent_dims + latent_dims; init_weight=zeros))
    
    # Drift of prior. This is just an SDE drift in the latent space
    create_network(:drift_prior, Lux.Chain(
        Lux.Dense(in_dims => prior_size, hidden_activation),
        repeat([Lux.Dense(prior_size => prior_size, hidden_activation)], depth)...,
        Lux.Dense(prior_size => latent_dims),
    ))
    # Drift of posterior. This is the term of an SDE when fed with the context.
    create_network(:drift_posterior, Lux.Chain(
        Lux.Dense(in_dims + context_size => posterior_size, hidden_activation),
        repeat([Lux.Dense(posterior_size => posterior_size, hidden_activation)], depth)...,
        Lux.Dense(posterior_size => latent_dims),
    ))
    # Prior and posterior share the same diffusion (they are not actually evaluated
    # seperately while training, only their KL divergence). This is a diagonal
    # diffusion, i.e. every term in the latent space has its own independent
    # Wiener process.
    # create_network(:diffusion, Lux.Parallel(nothing, [
    #         Lux.Chain(
    #             Lux.Dense((timedependent ? 2 : 1) => diffusion_size, tanh),
    #             Lux.Dense(diffusion_size => 1, Lux.sigmoid_fast)
    #         )
    #         # Lux.Scale(1, init_weight=ones, init_bias=ones)
    #     for i in 1:latent_dims]...
    # ))
    
    create_network(:diffusion,
        Lux.Chain(
            Lux.Scale(latent_dims, init_weight=Lux.glorot_uniform, init_bias=Lux.glorot_uniform, Lux.sigmoid),
            Lux.WrappedFunction(Base.Fix1(broadcast, (x) -> x + 0.0001))
        )
    )

    # The encoder is a recurrent neural network.
    create_network(:encoder_recurrent, Lux.Recurrence(Lux.GRUCell(data_dims => rnn_size); return_sequence=true))
    
    # The encoder_net is appended to the results of the encoder. Couldn't make
    # this work directly in Lux.
    create_network(:encoder_net, Lux.Dense(rnn_size => context_size))

    # The projector will transform the latent space back into data space.
    create_network(:projector, Lux.Dense(latent_dims => data_dims))

    return LatentSDE(
        networks...,
        solver,
        tspan,
        datasize;
        solver_kwargs...
    )
end

function get_distributions(model, model_p, st, context)
    normsandvars, _ = model(context, model_p, st)
    # batches are on the second dimension
    batch_indices = eachindex(context[1, :])
    # output ordered like [norm, var, norm, var, ...]
    halfindices = 1:Int(length(normsandvars[:, 1]) / 2)

    return hcat([reshape([Normal{Float64}(normsandvars[2*i-1, j], exp(0.5e0 * normsandvars[2*i, j])) for i in halfindices], :, 1) for j in batch_indices]...)
end

function sample_prior(n::LatentSDE, ps, st; b=1, seed=nothing, noise=(seed) -> nothing, tspan=n.tspan, datasize=n.datasize)
    (eps, seed) = ChainRulesCore.ignore_derivatives() do
        if seed !== nothing
            Random.seed!(seed)
        end
        latent_dimensions = n.initial_prior.out_dims ÷ 2
        (rand(Normal{Float64}(0.0e0, 1.0e0), (latent_dimensions, b)), rand(UInt32))
    end

    # We vcat 0e0 to these so that the prior has the same dimensions as the posterior (for noise reasons)
    function dudt_prior(u, p, t) 
        time_or_empty = n.timedependent ? [t] : []
        vcat(n.drift_prior(vcat(u[1:end-1], time_or_empty), p.drift_prior, st.drift_prior)[1], 0e0)
    end
    function dudw_diffusion(u, p, t) 
        time_or_empty = n.timedependent ? [t] : []
        # vcat(reduce(vcat, n.diffusion(Tuple([vcat(x, time_or_empty) for x in u[1:end-1]]), p.diffusion, st.diffusion)[1]), 0e0)
        vcat(n.diffusion(u[1:end-1], p.diffusion, st.diffusion)[1], 0e0)
    end

    initialdists_prior = get_distributions(n.initial_prior, ps.initial_prior, st.initial_prior, [1.0e0])
    distributions_with_eps = [zip(initialdists_prior, eps_batch) for eps_batch in eachslice(eps, dims=2)]
    z0 = hcat([reshape([x.μ + ep * x.σ for (x, ep) in d], :, 1) for d in distributions_with_eps]...)
    function prob_func(prob, batch, repeat)
        noise_instance = ChainRulesCore.ignore_derivatives() do
            noise(Int(floor(seed + batch)))
        end
        SDEProblem{false}(dudt_prior, dudw_diffusion, vcat(z0[:, batch], 0e0), tspan, ps, seed=seed + batch, noise=noise_instance)
    end

    ensemble = EnsembleProblem(nothing, output_func=(sol, i) -> (sol, false), prob_func=prob_func)

    Timeseries(solve(ensemble, n.solver, trajectories=b; saveat=range(tspan[1], tspan[end], datasize), dt=(tspan[end] / datasize), n.kwargs...))
end

function sample_prior_dataspace(n::LatentSDE, ps, st; kwargs...)
    prior_latent = sample_prior(n, ps, st; kwargs...)
    map_dims(x -> n.projector(x[1:end-1], ps.projector, st.projector)[1], prior_latent)
end

# from https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
function stable_divide(a::Array{Float64}, b::Array{Float64}, eps=1e-4)
    ChainRulesCore.ignore_derivatives() do
        if any([abs(x) <= eps for x in b])
            @warn "diffusion too small"
        end
    end
    b = map(x -> abs(x) <= eps ? eps * sign(x) : x, b)
    a ./ b
end


function augmented_drift(n::LatentSDE, times::Vector{Float64}, batch::Int, st::NamedTuple, u_in::Vector{Float64}, info::ComponentVector{Float64}, t::Float64)
    # Remove augmented term from input
    u::Vector{Float64} = u_in[1:end-1]
    
    p = info.ps
    context = info.context

    # Get the context for the posterior at the current time
    # initial state evolve => get the posterior at future start time
    posterior_net_input::Vector{Float64} = ChainRulesCore.ignore_derivatives() do
        time_index = max(1, searchsortedlast(times, t))
        vcat(u, @view context[:, batch, time_index])
    end

    prior = n.drift_prior(u, p.drift_prior, st.drift_prior)[1]
    posterior = n.drift_posterior(posterior_net_input, p.drift_posterior, st.drift_posterior)[1]

    # # The diffusion is diagonal, so a single network is invoked on each dimension
    # diffusion = reduce(vcat, n.diffusion(([n.timedependent ? vcat(x, t) : [x] for x in u]...,), p.diffusion, st.diffusion)[1])
    diffusion = n.diffusion(u, p.diffusion, st.diffusion)[1]

    # The augmented term for computing the KL divergence
    u_term = stable_divide(posterior .- prior, diffusion)
    augmented_term = 0.5e0 * sum(abs2, u_term; dims=[1])

    vcat(posterior, augmented_term)
end

function augmented_drift_batch(n::LatentSDE, times::Array{Float64}, latent_dims::Int, batch_size::Int, st::NamedTuple, u_in_vec::Array{Float64}, info::ComponentVector{Float64}, t::Float64)
    u_in = reshape(u_in_vec, latent_dims + 1, batch_size)
    # Remove augmented term from input
    u::Array{Float64} = u_in[1:end-1, :]

    p = info.ps
    context = info.context

    # Get the context for the posterior at the current time
    # initial state evolve => get the posterior at future start time
    time_index = max(1, searchsortedlast(times, t))
    posterior_net_input::Array{Float64} = vcat(u, context[:, :, time_index])

    prior = n.drift_prior(u, p.drift_prior, st.drift_prior)[1]
    posterior = n.drift_posterior(posterior_net_input, p.drift_posterior, st.drift_posterior)[1]

    # # The diffusion is diagonal, so a single network is invoked on each dimension
    # diffusion = reduce(vcat, n.diffusion(([n.timedependent ? vcat(x, t) : [x] for x in u]...,), p.diffusion, st.diffusion)[1])
    diffusion = n.diffusion(u, p.diffusion, st.diffusion)[1]

    # The augmented term for computing the KL divergence
    u_term = stable_divide(posterior .- prior, diffusion)
    augmented_term = 0.5e0 * sum(abs2, u_term; dims=1)

    reshape(vcat(posterior, augmented_term), (latent_dims + 1) * batch_size)
end

function augmented_diffusion(n::LatentSDE, st::NamedTuple, u_in::Vector{Float64}, info::ComponentVector{Float64}, t::Float64)
    p = info.ps
    # diffusion = reduce(vcat, n.diffusion(([n.timedependent ? vcat(x, t) : [x] for x in u]...,), p.diffusion, st.diffusion)[1])
    diffusion = n.diffusion(u_in[1:end-1], p.diffusion, st.diffusion)[1]
    return vcat(diffusion, 0e0)
end

function augmented_diffusion_batch(n::LatentSDE, latent_dims::Int, batch_size::Int, st::NamedTuple, u_in_vec::Array{Float64}, info::ComponentVector{Float64}, t::Float64)
    p = info.ps
    # diffusion = reduce(vcat, n.diffusion(([n.timedependent ? vcat(x, t) : [x] for x in u]...,), p.diffusion, st.diffusion)[1])
    u_in = reshape(u_in_vec, latent_dims + 1, batch_size)
    diffusion = n.diffusion(u_in[1:end-1, :], p.diffusion, st.diffusion)[1]
    reshape(vcat(diffusion, zeros(size(u_in[end:end, :]))), (latent_dims + 1) * batch_size)
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
- `likelihood_dist`: Distribution for computing log-likelihoods (as a way of computing distance).
- `likelihood_scale`: Variance of `likelihood_dist`.
"""
function (n::LatentSDE)(timeseries::Timeseries, ps::ComponentVector, st;
    sense=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=false),
    ensemblemode=EnsembleThreads(),
    seed=nothing,
    noise=(seed, noise_size) -> nothing,
    likelihood_dist=Normal,
    likelihood_scale=0.01e0,
)
    latent_dimensions = n.initial_prior.out_dims ÷ 2
    batch_size = length(timeseries.u)

    # We are using matrices with the following dimensions:
    # 1 = latent space dimension
    # 2 = batch number
    # 3 = time step
    (eps, seed) = ChainRulesCore.ignore_derivatives() do
        if seed !== nothing
            Random.seed!(seed)
        end
        (rand(Normal{Float64}(0.0e0, 1.0e0), (latent_dimensions, batch_size)), rand(UInt32))
    end

    tsmatrix = ChainRulesCore.ignore_derivatives() do
        reduce(hcat, [reshape(map(only, u), 1, 1, :) for u in timeseries.u])
    end

    timecat = ChainRulesCore.ignore_derivatives() do
        function(x, y)
            cat(x, y; dims=3)
        end
    end

    # Lux recurrent uses batches / time the other way around...
    # time: dimension 3 => dimension 2
    # batches: dimension 2 => dimension 3
    tsmatrix_flipped = ChainRulesCore.ignore_derivatives() do
        reverse(permutedims(tsmatrix, (1, 3, 2)), dims=2)
    end

    precontext_flipped = reverse(n.encoder_recurrent(tsmatrix_flipped, ps.encoder_recurrent, st.encoder_recurrent)[1])
    context_flipped = [n.encoder_net(x, ps.encoder_net, st.encoder_net)[1] for x in precontext_flipped]

    # context_flipped is now a vector of 2-dim matrices
    # latent space: dimension 1
    # batch: dimension 2
    context_precomputed = reduce(timecat, context_flipped)

    initialdists_prior = get_distributions(n.initial_prior, ps.initial_prior, st.initial_prior, [1.0e0])

    initialdists_posterior = get_distributions(n.initial_posterior, ps.initial_posterior, st.initial_posterior, context_precomputed[:, :, 1])
    distributions_with_eps = [zip(dist, eps_batch) for (dist, eps_batch) in zip(eachslice(initialdists_posterior, dims=2), eachslice(eps, dims=2))]
    z0 = hcat([reshape([x.μ + ep * x.σ for (x, ep) in d], :, 1) for d in distributions_with_eps]...)

    augmented_z0 = vcat(z0, zeros(1, length(z0[1, :])))

    # Deriving operations with ComponentArray is not so easy, first we have to
    # grab the axes and then re-construct using a vector
    axes = ChainRulesCore.ignore_derivatives() do
        getaxes(ComponentArray((context=context_precomputed, ps=ps)))
    end

    vec_context = vec(context_precomputed)

    info = ComponentArray(vcat(vec_context, ps), axes)

    noise_instance = ChainRulesCore.ignore_derivatives() do
        noise(Int(floor(seed)), (latent_dimensions + 1) * batch_size)
    end
    sde_problem = SDEProblem{false}(
        (u, p, t) -> augmented_drift_batch(n, timeseries.t, latent_dimensions, batch_size, st, u, p, t),
        (u, p, t) -> augmented_diffusion_batch(n, latent_dimensions, batch_size, st, u, p, t),
        reshape(augmented_z0, (latent_dimensions + 1) * batch_size),
        n.tspan,
        info,
        seed=seed,
        noise=noise_instance
    )
    solution = solve(sde_problem, n.solver; sensealg=sense, saveat=range(n.tspan[1], n.tspan[end], n.datasize), dt=(n.tspan[end] / n.datasize), n.kwargs...)

    # If ts_start > 0, the timeseries starts after the latent sde, thus only score after ts_start
    # The timeseries could be irregularily sampled or have a different rate than the model, so search for appropiate points here
    # TODO: Interpolate this?
    ts_indices = ChainRulesCore.ignore_derivatives() do
        [searchsortedfirst(solution.t, t) for t in timeseries.t]
    end
    ts_start = ChainRulesCore.ignore_derivatives() do
        ts_indices[1]
    end
    
    sol = reduce(timecat, [reshape(x, latent_dimensions + 1, batch_size) for x in solution.u])
    posterior_latent = sol[1:end-1, :, :]
    kl_divergence_time = sol[end:end, :, :]

    initialdists_kl = reduce(hcat, [reshape([KullbackLeibler(a, b) for (a, b) in zip(initialdists_posterior[:, batch], initialdists_prior)], :, 1) for batch in eachindex(timeseries.u)])
    kl_divergence = sum(initialdists_kl, dims=1) .+ sol[:, :, end]

    projected_z0 = n.projector(z0, ps.projector, st.projector)[1]
    projected_ts = reduce(timecat, [n.projector(x, ps.projector, st.projector)[1] for x in eachslice(posterior_latent, dims=3)])

    logp = ChainRulesCore.ignore_derivatives() do
        function(x, y)
            loglikelihood(likelihood_dist(y, likelihood_scale), x)
        end
    end
    likelihoods_initial = if ts_start == 1
        [logp(x, y) for (x, y) in zip(tsmatrix[:, :, 1], z0[1:1, :])]
    else
        fill(0.0e0, size(projected_z0))
    end
    likelihoods_time = sum([logp(x, y) for (x, y) in zip(tsmatrix, projected_ts[:, :, ts_indices])], dims=3)[:, :, 1]
    likelihoods = likelihoods_initial .+ likelihoods_time

    return posterior_latent, projected_ts, kl_divergence_time, kl_divergence, likelihoods
end

function loss(n::LatentSDE, timeseries, ps, st, beta; kwargs...)
    posterior, projected_ts, logterm, kl_divergence, distance = n(timeseries, ps, st; kwargs...)
    return -distance .+ (beta * kl_divergence)
end
