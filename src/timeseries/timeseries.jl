"A Timeseries provides timeseries data"
abstract type Timeseries end
using Dates

"Get a series from a Timeseries"
function series(model :: Timeseries, u0, tspan, datasize) end

function timeseriesplot!(model :: Timeseries, u0s, tspan, datasize)
    fig = plot(0, 0, xlabel=xlabel(model), ylabel=ylabel(model), legend=nothing)
    for u0 in u0s
        plot!(fig, series(model, u0, tspan, datasize), fmt=:png, legend=nothing)
    end
    folder = "plots/$(typeof(model))"
    Base.Filesystem.mkpath(folder)
    savefig(fig, "$folder/$(now()).png")
end
xlabel(model :: Timeseries) = "time"
ylabel(model :: Timeseries) = "value"

include("zero_d_energy_balance.jl")
