using Lux, DiffEqFlux, Zygote, Optimisers
using ProgressMeter
using Statistics
using IPMeasures
using Random
using ComponentArrays
using ForwardDiff
using StatsBase: sample

include("neural.jl")

export NeuralSDEProblem, toy_neural_sde, trainproblem!

struct NeuralSDEProblem
    timeseries::Vector{NODEDataloader}
    u0::Array{Float32}

    neural_sde::Any
    optimiser::Optimisers.AbstractRule
end

function toy_neural_sde()
    n = 1000
    datasize = 30
    tspan = (0.0f0, 1f0)
    tsteps = range(tspan[1], tspan[2], datasize)
    ebm = ZeroDEnergyBalanceModelNonStochastic()
    NeuralSDEProblem(
        series(ebm, range(210.0f0, 320.0f0, n), tspan, datasize),
        range(0f0, 1f0, n), # FIXME ugly, works because data is normalized to starting range
        AugmentedNeuralODE(
            # Lux.Chain(Lux.Dense(1 => 6, tanh), Lux.Dense(6 => 6, tanh), Lux.Dense(6 => 1)),
            # Lux.Chain(Lux.Dense(1 => 10, swish), Lux.Dense(10 => 10, swish), Lux.Dense(10 => 1, init_bias=Lux.glorot_uniform, init_weight=((rng, out_dims, in_dims) -> Lux.glorot_uniform(rng, out_dims, in_dims, gain=2)))),
            Lux.Chain(Lux.Dense(1 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 1)),
            function (u, p, t, model, st)
                net, st = model(u, p, st)
                
                return net
                # return net
                # return [Float32(energy_in(u[1], ebm) - net[1])]
            end,
            # (u, t) -> drift(u, ZeroDEnergyBalanceModel(), t),
            # Lux.Scale(1, use_bias=false, init_weight=Lux.zeros32),
            # gpu(Chain(Flux.Scale(1; bias = false, init = Flux.glorot_uniform(gain=0.1)))),
            tspan,
            # SOSRA2(),
            # sensealg=SciMLSensitivity.InterpolatingAdjoint(),
            # sensealg=SciMLSensitivity.ForwardDiffSensitivity(),
        ),
        Optimisers.Adam(0.06),
    )
end

# function fit_derivative()
#     n = 1000
#     NeuralSDEProblem(
#         repeat(permutedims(hcat([drift([x], ZeroDEnergyBalanceModelNonStochastic(), nothing) for x in range(220f0, 300f0, n)]...)), outer=[1,n]),
#         range(0f0, 1f0, n), # FIXME ugly, works because data is normalized to starting range
#         Lux.Chain(Lux.Dense(1 => 50, tanh), Lux.Dense(50 => 1, init_bias=Lux.glorot_uniform, init_weight=((rng, out_dims, in_dims) -> Lux.glorot_uniform(rng, out_dims, in_dims, gain=10)))),
#         Optimisers.Adam(0.01),
#     )
# end


function trainproblem!(problem::NeuralSDEProblem)
    Base.exit_on_sigint(true)
    
    # timeseries = (problem.timeseries .- min(problem.timeseries[1,:]...)) / (max(problem.timeseries[1,:]...) - min(problem.timeseries[1,:]...))

    # data_means = mean(timeseries, dims=2)
    # data_vars = var(timeseries, mean=data_means, dims=2)

    # function get_means_and_vars(samples, neural_sde)
    #     u = repeat(reshape(problem.u0, :, 1), 1, samples)
    #     # sorry for the terrible expression, would be equivalent to
    #     # problem.neural_sde(u) if that used different seeds
    #     trajectories = hcat([neural_sde([x])[1, :] for x in u]...)

    #     neural_means = mean(vcat(trajectories), dims=2)
    #     neural_vars = var(vcat(trajectories), dims=2, mean=neural_means)

    #     neural_means, neural_vars, samples
    # end

    flatdata = vcat([[max(u...) for (t,u) in ts] for ts in problem.timeseries]...)
    datamax = max(flatdata...)
    datamin = min(flatdata...)

    function normalize(x)
        return (x - datamin) / (datamax - datamin)
    end

    function rescale(x)
        return (x * (datamax - datamin)) + datamin
    end

    rng = Random.default_rng()

    ps, st = Lux.setup(rng, problem.neural_sde)
    ps_inner, st_inner = Lux.setup(rng, problem.neural_sde.model)

    ps = ComponentArray(ps)

    function apply(ps, data)
        [Lux.apply(remake(problem.neural_sde, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st)[1] for (t, u) in data]
        # trajectories = []
        # for (t, u) in data
        #     trajectory, st_ = Lux.apply(remake(problem.neural_sde, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st)
        #     push!(trajectories, trajectory)
        #     st = merge(st, st_)
        # end
        # trajectories
    end
    
    function loss_neuralsde(ps, data)
        trajectories = apply(ps, data)

        # println(map(x -> x[2], data))

        sum(abs, map(v -> sum(abs2, v), map(x -> map(y -> rescale(first(y)), x.u), trajectories) .- map(x -> x[2], data)))

        # neural_means, neural_vars, _ = get_means_and_vars(samples, neural_sde)

        # loss
        # -mmd(IPMeasures.GaussianKernel(0.5), timeseries, trajectories)
        
        # neural_means = mean(vcat(trajectories), dims=2)
        # neural_vars = var(vcat(trajectories), dims=2, mean=neural_means)

        # sum(abs2, neural_means - data_means) + sum(abs2, neural_vars - data_vars)
    end

    
    function cb(data)
        # display(plot([(x,Lux.apply(problem.neural_sde.model, [x], ps_inner, st_inner)[1][1]) for x in -1:0.1:1]))

        # sorry for the terrible expression, would be equivalent to
        # problem.neural_sde(u) if that used different seeds
        # trajectories = hcat([Array(Lux.apply(problem.neural_sde, [x], ps, st)[1])[1, :] for x in problem.u0]...)
        trajectories = apply(ps, data)

        trajectories2 = []
        t = 0f0:0.01f0:3f0
        for u in 200f0:5f0:350f0
            trajectory, st_ = Lux.apply(remake(problem.neural_sde, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st)
            push!(trajectories2, trajectory)
            st = merge(st, st_)
        end

        # println(trajectories)
        # display(plot(trajectories .- minibatch))
        # sum(abs, (trajectories[horizon] .- minibatch[horizon]))

        # pl = plot(neural_means, linewidth=2, label="prediction", color="orange", dpi=700)
        pl = plot()
        for t in trajectories
            plot!(pl, (map(first, t.t), map(y -> rescale(first(y)), t.u)), linewidth=1, label="series (prediction)", color="orange", legend=false)
        end
        for t in trajectories2
            plot!(pl, (map(first, t.t), map(y -> rescale(first(y)), t.u)), linewidth=1, label="series (prediction)", color="green", legend=false)
        end

        # # plot!(pl, data_means, linewidth=2, label="data", color="blue")
        plot!(pl, data, linewidth=1, label="series (data)", legend=false, color="blue")
        display(pl)
    end

    learning_rate = 0.05

    for batch_size in 5:5:20
        opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)
        batch = map(x -> NODEDataloader(x, batch_size), problem.timeseries)
        println("Batch Size $batch_size")
        println("Batch $batch_size")
        @gif for step in 1:250
            minibatch = batch[sample(1:size(batch)[1], 10, replace=false)]
            data = reduce(vcat, [[x for x in minibatch[i]] for i in 1:size(minibatch)[1]])
            cb(data)
            loss = loss_neuralsde(ps, data)
            grads = Zygote.gradient((x -> loss_neuralsde(x, data)), ps)[1]
            # println("Loss: $loss, Grads: $grads")
            Optimisers.update!(opt_state, ps, grads)
        end
        learning_rate *= 0.5
    end
end
