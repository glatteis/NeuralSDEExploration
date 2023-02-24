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
    return AugmentedNeuralODE{typeof(model),typeof(dudt),typeof(p),typeof(re),
        typeof(tspan),typeof(args),typeof(kwargs)}(
        n.model, n.dudt, n.p, n.re, tspan, n.args, kwargs)
end

function (n::AugmentedNeuralODE{M})(x, p, st) where {M<:Lux.AbstractExplicitLayer}
    ff = ODEFunction{false}(((u, p, t) -> n.dudt(u, p, t, n.model, st)), tgrad=DiffEqFlux.basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solution = solve(prob, n.args...; sensealg=sense, n.kwargs...), st
    return solution
end

# TODO
# """
# A NeuralDSDE that can be augmented using known terms of the equation
# """
# struct AugmentedNeuralDSDE{M,P,RE,M2,RE2,T,A,K} <: NeuralSDELayer
#     p::P
#     len::Int
#     drift::M
#     re1::RE
#     diffusion::M2
#     re2::RE2
#     tspan::T
#     args::A
#     kwargs::K
# end

# """
# A NeuralSDE that can be augmented using known terms of the equation
# """
# struct AugmentedNeuralSDE{P,M,RE,M2,RE2,T,A,K} <: NeuralSDELayer
#     p::P
#     len::Int
#     drift::M
#     re1::RE
#     diffusion::M2
#     re2::RE2
#     tspan::T
#     nbrown::Int
#     args::A
#     kwargs::K
# end

