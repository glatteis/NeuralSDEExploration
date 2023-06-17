export series, timeseriesplot!

"A TimeseriesModel provides timeseries data"
abstract type TimeseriesModel end

"Get a series from a TimeseriesModel"
function series(model :: TimeseriesModel, u0, tspan, datasize; seed=rand(UInt32), noise=(seed) -> nothing, kwargs...) end

# function timeseriesplot!(model :: TimeseriesModel, u0s, tspan, datasize)
#     fig = plot(0, 0, xlabel=xlabel(model), ylabel=ylabel(model), fmt=:png, dpi = 600, legend=nothing)
#     for u0 in u0s
#         plot!(fig, series(model, u0, tspan, datasize), legend=nothing)
#     end
#     folder = "plots/$(typeof(model))"
#     Base.Filesystem.mkpath(folder)
#     savefig(fig, "$folder/$(now()).png")
# end
xlabel(model :: TimeseriesModel) = "time"
ylabel(model :: TimeseriesModel) = "value"

include("zero_d_energy_balance.jl")
include("fitzhugh_nagumo.jl")
include("ornstein_uhlenbeck.jl")
