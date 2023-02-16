using Flux, DiffEqFlux, Enzyme
using Statistics

struct NeuralSDEProblem
    timeseries :: Vector{Array{Float64}}
    u0 :: Array{Float64}
    
    neural_sde :: NeuralSDE
    optimiser :: Flux.Optimise.AbstractOptimiser
end

function toy_neural_sde()
    n = 50
    datasize = 4
    tspan = (0, 5)
    tsteps = range(tspan[1], tspan[2], length = datasize)
    NeuralSDEProblem(
        vec([Array(series(ZeroDEnergyBalanceModel(), 300.0, tspan, datasize)) for i in 1:n]),
        [300.0],
        NeuralSDE(
            gpu(Chain(Dense(1 => 20, tanh), Dense(20 => 1))),
            gpu(Dense(1 => 1, bias=false)),
            (0, 10),
            1,
            SOSRA(),
            # sensealg=SciMLSensitivity.InterpolatingAdjoint(; autojacvec=EnzymeVJP()),
            sensealg=SciMLSensitivity.InterpolatingAdjoint(),
            saveat=tsteps
        ),
        ADAM(0.1),
    )
end

function trainproblem!(problem :: NeuralSDEProblem)
    ps = Flux.params(problem.neural_sde)
    
    data_means = mean(problem.timeseries)
    data_vars = var(problem.timeseries, mean=data_means)

    function get_means_and_vars(n)
        u = repeat(reshape(problem.u0, :, 1), 1, n)
        # sorry for the terrible expression, would be equivalent to
        # problem.neural_sde(u) if that used different seeds
        samples = hcat([[x[1] for x in problem.neural_sde([x]).u][:,:] for x in u]...)

        neural_means = mean(vcat(samples), dims = 2)
        neural_vars = var(vcat(samples), dims = 2, mean = neural_means)
        
        neural_means, neural_vars
    end

    function loss_neuralsde(p; n = 4)
        neural_means, neural_vars = get_means_and_vars(n)
        loss = sum(abs2, data_means - neural_means) + sum(abs2, data_vars - neural_vars)

        loss
    end
    
    function cb()
        neural_means, neural_vars = get_means_and_vars(4)

        # currentprediction = predict(problem.u0)
        pl = plot(neural_means, ribbon=neural_vars ,label="prediction")
        # plot!(pl,currentprediction[1,:],label="prediction")
        plot!(pl, data_means, ribbon=data_vars, label="data")
        display(plot(pl))
    end
    cb()
    opt_state = Flux.setup(problem.optimiser, problem.neural_sde)
    Flux.train!(loss_neuralsde, ps, opt_state, problem.optimiser, cb=cb)
end
