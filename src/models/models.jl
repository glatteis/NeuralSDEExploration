using Lux, DiffEqFlux, Zygote, Optimisers
using ProgressMeter
using Statistics
using IPMeasures
using Random
using ComponentArrays
using ForwardDiff
using StatsBase: sample

include("neural.jl")
include("latent.jl")

export NeuralSDEProblem, toy_neural_sde, trainkullback!, trainmeansquare!

struct NeuralSDEProblem
    timeseries::Vector{NODEDataloader}

    neural_sde::Any
    # optimiser::Optimisers.AbstractRule
    # loss::Any
end

function toy_neural_ode()
    n = 1000
    datasize = 30
    tspan = (0.0f0, 1f0)
    ebm = ZeroDEnergyBalanceModelNonStochastic()
    NeuralSDEProblem(
        series(ebm, range(210.0f0, 320.0f0, n), tspan, datasize),
        AugmentedNeuralODE(
            Lux.Chain(Lux.Dense(1 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 1)),
            function (u, p, t, model, st)
                net, st = model(u, p, st)
                
                return net
            end,
            tspan,
            # SOSRA2(),
            # sensealg=SciMLSensitivity.InterpolatingAdjoint(),
            # sensealg=SciMLSensitivity.ForwardDiffSensitivity(),
            
        ),
    )
end

function toy_neural_sde()
    n = 2
    datasize = 20000
    tspan = (0.0f0, 1000f0)
    ebm = ZeroDEnergyBalanceModel()
    NeuralSDEProblem(
        series(ebm, range(210.0f0, 320.0f0, n), tspan, datasize, batchsize=500),
        AugmentedNeuralDSDE(
            Lux.Chain(Lux.Dense(1 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 10, tanh), Lux.Dense(10 => 1)),
            function (u, p, t, model, st)
                net, st = model(u, p, st)
                return net
            end,
            Lux.Chain(Lux.Dense(1 => 1, tanh, use_bias=false, init_weight=Lux.zeros32), Lux.Dense(1 => 1, use_bias=false)),
            function (u, p, t, model, st)
                net, st = model(u, p, st)
                return net
            end,
            tspan,
            SOSRA(),
            # sensealg=SciMLSensitivity.InterpolatingAdjoint(),
            # sensealg=SciMLSensitivity.ForwardDiffSensitivity(),
        ),
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


function trainmeansquare!(problem::NeuralSDEProblem)
    Base.exit_on_sigint(true)
    
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

    ps = ComponentArray(ps)

    function apply(ps, data)
        [Lux.apply(remake(problem.neural_sde, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st)[1] for (t, u) in data]
    end
    
    function loss_neuralsde(ps, data)
        trajectories = apply(ps, data)

        sum(abs, map(v -> sum(abs2, v), map(x -> map(y -> rescale(first(y)), x.u), trajectories) .- map((t, u) -> u, data)))

        # neural_means, neural_vars, _ = get_means_and_vars(samples, neural_sde)

        # loss
        # -mmd(IPMeasures.GaussianKernel(0.5), timeseries, trajectories)
        
        # neural_means = mean(vcat(trajectories), dims=2)
        # neural_vars = var(vcat(trajectories), dims=2, mean=neural_means)

        # sum(abs2, neural_means - data_means) + sum(abs2, neural_vars - data_vars)
    end

    
    function cb(data)
        trajectories = apply(ps, data)

        trajectories2 = []
        t = 0f0:0.01f0:3f0
        for u in 200f0:5f0:350f0
            trajectory, st_ = Lux.apply(remake(problem.neural_sde, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st)
            push!(trajectories2, trajectory)
            st = merge(st, st_)
        end

        pl = plot()
        for t in trajectories
            plot!(pl, (map(first, t.t), map(y -> rescale(first(y)), t.u)), linewidth=1, label="series (prediction)", color="orange", legend=false)
        end
        for t in trajectories2
            plot!(pl, (map(first, t.t), map(y -> rescale(first(y)), t.u)), linewidth=1, label="series (prediction)", color="green", legend=false)
        end

        plot!(pl, data, linewidth=1, label="series (data)", legend=false, color="blue")
        display(pl)
    end

    learning_rate = 0.05

    for batch_size in 5:5:20
        opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)
        batch = map(x -> NODEDataloader(x, batch_size), problem.timeseries)
        println("Batch Size $batch_size")
        @gif for step in 1:250
            minibatch = batch[sample(1:size(batch)[1], 10, replace=false)]
            data = reduce(vcat, [[x for x in minibatch[i]] for i in 1:size(minibatch)[1]])
            cb(data)
            loss = loss_neuralsde(ps, data)
            grads = Zygote.gradient((x -> loss_neuralsde(x, data)), ps)[1]
            Optimisers.update!(opt_state, ps, grads)
        end
        learning_rate *= 0.5
    end
end

function trainkullback!(problem::NeuralSDEProblem)
    Base.exit_on_sigint(true)
    
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

    ps = ComponentArray(ps)

    function apply(ps, data)
        [Lux.apply(remake(problem.neural_sde, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st)[1] for (t, u) in data]
    end
    
    function loss_neuralsde(ps, data)
        trajectories = apply(ps, data)

        sum(abs, map(v -> sum(abs2, v), map(x -> map(y -> rescale(first(y)), x.u), trajectories) .- map(x -> x[2], data)))

        # neural_means, neural_vars, _ = get_means_and_vars(samples, neural_sde)

        # loss
        # -mmd(IPMeasures.GaussianKernel(0.5), timeseries, trajectories)
        
        # neural_means = mean(vcat(trajectories), dims=2)
        # neural_vars = var(vcat(trajectories), dims=2, mean=neural_means)

        # sum(abs2, neural_means - data_means) + sum(abs2, neural_vars - data_vars)
    end

    
    function cb(data)
        # sorry for the terrible expression, would be equivalent to
        # problem.neural_sde(u) if that used different seeds
        # trajectories = hcat([Array(Lux.apply(problem.neural_sde, [x], ps, st)[1])[1, :] for x in problem.u0]...)
        trajectories = apply(ps, data)

        trajectories2 = []
        # t = 0f0:0.01f0:20f0
        # for u in 200f0:5f0:350f0
        #     trajectory, st_ = Lux.apply(remake(problem.neural_sde, tspan=(t[1], t[end]), saveat=range(t[1], t[end], size(t)[1])), [normalize(u[1])], ps, st)
        #     push!(trajectories2, trajectory)
        #     st = merge(st, st_)
        # end

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

    for batch_size in 100:500:1000
        opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), ps)
        batch = map(x -> NODEDataloader(x, batch_size), problem.timeseries)
        println("Batch Size $batch_size")
        @gif for step in 1:250
            minibatch = batch[sample(1:size(batch)[1], 1, replace=false)]
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

