using Dates
using DifferentialEquations
using NODEData

export series, timeseriesplot!

"A Timeseries provides timeseries data"
abstract type Timeseries end

"Get a series from a Timeseries"
function series(model :: Timeseries, u0, tspan, datasize, n) end

function timeseriesplot!(model :: Timeseries, u0s, tspan, datasize)
    fig = plot(0, 0, xlabel=xlabel(model), ylabel=ylabel(model), fmt=:png, dpi = 600, legend=nothing)
    for u0 in u0s
        plot!(fig, series(model, u0, tspan, datasize), legend=nothing)
    end
    folder = "plots/$(typeof(model))"
    Base.Filesystem.mkpath(folder)
    savefig(fig, "$folder/$(now()).png")
end
xlabel(model :: Timeseries) = "time"
ylabel(model :: Timeseries) = "value"

include("zero_d_energy_balance.jl")
include("fitzhugh_nagumo.jl")
