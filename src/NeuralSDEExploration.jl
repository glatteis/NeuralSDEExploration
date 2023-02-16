module NeuralSDEExploration

using Plots; gr()

include("timeseries/timeseries.jl")
include("models/models.jl")

function plot_something()
    timeseriesplot!(ZeroDEnergyBalanceModel(), [230, 250])
end

end # module NeuralSDEExploration
