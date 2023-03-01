export AugmentedNeuralODE, AugmentedNeuralDSDE, remake

"""
A NeuralODE that can be augmented using known terms of the equation
"""
struct AugmentedNeuralODE{M,D,P,T,A,K} <: DiffEqFlux.NeuralDELayer
    model::M
    dudt::D
    p::P
    tspan::T
    args::A
    kwargs::K
end

function AugmentedNeuralODE(model::Lux.AbstractExplicitLayer, dudt, tspan, args...; p=nothing, kwargs...)
    AugmentedNeuralODE{typeof(model),typeof(dudt),typeof(p),
        typeof(tspan),typeof(args),typeof(kwargs)}(
        model, dudt, p, tspan, args, kwargs)
end

function remake(n::AugmentedNeuralODE, args...; tspan=n.tspan, kwargs...)
    return AugmentedNeuralODE(
        n.model,
        n.dudt,
        tspan,
        args...;
        p=n.p,
        kwargs...
    )
end

function (n::AugmentedNeuralODE{M})(x, p, st) where {M<:Lux.AbstractExplicitLayer}
    ff = ODEFunction{false}(((u, p, t) -> n.dudt(u, p, t, n.model, st)), tgrad=DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solution = solve(prob, n.args...; sensealg=sense, n.kwargs...)
    return solution, st
end

"""
A NeuralDSDE that can be augmented using known terms of the equation
"""
struct AugmentedNeuralDSDE{N,D,N2,D2,P,T,A,K} <: DiffEqFlux.NeuralSDELayer
    drift::N
    drift_dudt::D
    diffusion::N2
    diffusion_dudt::D2
    p::P
    tspan::T
    args::A
    kwargs::K
end

function AugmentedNeuralDSDE(drift::Lux.AbstractExplicitLayer, drift_dudt, diffusion::Lux.AbstractExplicitLayer, diffiusion_dudt, tspan, args...; p=nothing, kwargs...)
    AugmentedNeuralDSDE{typeof(drift),typeof(drift_dudt),
        typeof(diffusion),typeof(diffiusion_dudt),
        typeof(p),typeof(tspan),typeof(args),typeof(kwargs)}(drift, drift_dudt,
        diffusion, diffiusion_dudt, p, tspan, args, kwargs)
end

function remake(n::AugmentedNeuralDSDE, args...; tspan=n.tspan, kwargs...)
    return AugmentedNeuralDSDE(
        n.drift,
        n.drift_dudt,
        n.diffusion,
        n.diffusion_dudt,
        tspan,
        args...;
        p=n.p,
        kwargs...
    )
end

function (n::AugmentedNeuralDSDE)(x,p,st)
    drift = (u, p, t) -> n.drift_dudt(u, p.drift, t, n.drift, st.drift)
    diffusion = (u, p, t) -> n.diffusion_dudt(u, p.diffusion, t, n.diffusion, st.diffusion)

    # ff = SDEFunction{false}(drift,diffusion,tgrad=DiffEqFlux.basic_tgrad)
    prob = SDEProblem{false}(drift,diffusion,x,n.tspan,p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    return solve(prob,n.args...;sensealg=sense,n.kwargs...), st
end
