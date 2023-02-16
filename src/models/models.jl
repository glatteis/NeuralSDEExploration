using Flux, DiffEqFlux, Enzyme
using Statistics

struct NeuralSDEProblem
    timeseries :: Vector{Array{Float64}}
    u0 :: Array{Float64}
    
    neural_sde :: NeuralSDE
    optimiser :: Flux.Optimise.AbstractOptimiser
end

function toy_neural_sde()
    n = 20
    datasize = 200
    tspan = (0, 50)
    tsteps = range(tspan[1], tspan[2], length = datasize)
    NeuralSDEProblem(
        vec([Array(series(ZeroDEnergyBalanceModel(), 270.0, tspan, datasize)) for i in 1:n]),
        [270.0],
        NeuralSDE(
            gpu(Chain(Dense(1 => 5, tanh), Dense(5 => 5, tanh), Dense(ones(1, 5), false))),
            gpu(Chain(Dense(zeros(1, 1), false))),
            tspan,
            1,
            SOSRA(),
            # sensealg=SciMLSensitivity.InterpolatingAdjoint(; autojacvec=EnzymeVJP()),
            # sensealg=SciMLSensitivity.InterpolatingAdjoint(),
            saveat=tsteps
        ),
        AdaDelta(0.1),
    )
end

function trainproblem!(problem :: NeuralSDEProblem)
    Base.exit_on_sigint(true)

    data_means = mean(problem.timeseries)
    data_vars = var(problem.timeseries, mean=data_means)

    function get_means_and_vars(samples, neural_sde)
        u = repeat(reshape(problem.u0, :, 1), 1, samples)
        # sorry for the terrible expression, would be equivalent to
        # problem.neural_sde(u) if that used different seeds
        samples = hcat([[x[1] for x in neural_sde([x]).u][:,:] for x in u]...)

        neural_means = mean(vcat(samples), dims = 2)
        neural_vars = var(vcat(samples), dims = 2, mean = neural_means)
        
        neural_means, neural_vars, samples
    end

    function loss_neuralsde(neural_sde, d...; samples = 20)
        neural_means, neural_vars, _ = get_means_and_vars(samples, neural_sde)
        loss = sum(abs2, data_means - neural_means) + sum(abs2, data_vars - neural_vars)

        loss
    end
    
    function cb(epoch)
        neural_means, neural_vars, samples = get_means_and_vars(20, problem.neural_sde)

        pl = plot(neural_means, linewidth=2, label="prediction", color = "orange", dpi=700)
        plot!(pl, samples, linewidth=0.3, label="series (prediction)", color = "orange", legend = false)

        plot!(pl, data_means, linewidth=2, label="data", color = "blue")
        plot!(pl, problem.timeseries, linewidth=0.3, label="series (data)", legend = false, color = "blue")

        display(pl)
        # savefig(pl, "plots/anim/$epoch.png")
    end
    cb(0)
    opt_state = Flux.setup(problem.optimiser, problem.neural_sde)
    for epoch in 1:100
        yield()
        println("epoch $epoch")
        cb(epoch)
        Flux.train!(loss_neuralsde, problem.neural_sde, 1:20, opt_state)
    end
end
