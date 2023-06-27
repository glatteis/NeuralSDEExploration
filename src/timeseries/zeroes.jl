export Zeroes

struct Zeroes <: TimeseriesModel
end

function series(model::Zeroes, u0s, tspan, datasize; seed=rand(UInt32), noise=(seed) -> nothing, kwargs...)
    Timeseries((u=repeat([[0.0] for x in range(tspan[1], tspan[2], datasize)], 10000), t=range(tspan[1], tspan[2], datasize)))
end
