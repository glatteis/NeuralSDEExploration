export Timeseries, filter_dims, map_dims, map_ts, select_ts

"""A collection of multivariate timeseries.
This type exists to consolidate the tons of different representations that are
used for timeseries:
    - vectors of named tuples
    - vectors of RODESolutions
    - EnsembleSolutions
and work with univariate and multivariate timeseries easily, be easily
plottable and that kind of stuff.
"""
struct Timeseries{T}
    t::Vector{T}
    u::Vector{Vector{Vector{T}}}
end

function Timeseries(vec::Vector)
    t = nothing
    u = []
    for entry in vec
        if t !== nothing && entry.t != t
            throw(ArgumentError("All provided time vectors must be the same"))
        end
        t = entry.t
        if length(entry.u) > 0 && entry.u[1] isa Number
            # Univariate timeseries
            push!(u, map(x -> [x], entry.u))
        else
            # Multivariate timeseries
            push!(u, entry.u)
        end
    end
    Timeseries{eltype(t)}(t, u)
end

function Timeseries(sol::EnsembleSolution)
    Timeseries([x for x in sol])
end

function select_ts(range, ts::Timeseries)
    Timeseries(ts.t, ts.u[range])
end

"""
Executes f on each datapoint by dimensions, so an input to f would be
    [<value in dimension 1>, <value in dimension 2>, ..., <value in dimension n>]
and f produces
    [<value in dimension 1>, <value in dimension 2>, ..., <value in dimension m>]
"""
function map_dims(f, ts::Timeseries)
    Timeseries(ts.t, map(x -> map(y -> f(y), x), ts.u))
end

"""
Removes dimensions from the timeseries by the range `dims`.
"""
function filter_dims(dims, ts::Timeseries)
    map_dims(x -> x[dims], ts)
end

"""
Executes f on each datapoint by timeseries, so an input to f would be
    [<value at dimension i in ts 1>, <value at dimension i at ts 2>, ..., <value at dimension i at ts n>]
and f produces
    [<value at dimension i in ts 1>, <value at dimension i at ts 2>, ..., <value at dimension i at ts m>]
"""
function map_ts(f, ts::Timeseries)
    dims = length(ts.u[1][1])
    result = Vector{Vector{Vector{eltype(ts.t)}}}()
    for dimension in 1:dims
        input_in_tuples = zip(map(case -> collect(map(el -> el[dimension], case)), ts.u)...)
        output = [f(collect(x)) for x in input_in_tuples]
        push!(result, output)
    end
    Timeseries(ts.t, result)
end

@recipe function f(ts::Timeseries)
    dims = length(ts.u[1][1])
    layout := @layout collect(repeat([1], dims))
    legend := false
    for dimension in 1:dims
        @series begin
            subplot := dimension
            this_dim = collect(map(case -> collect(map(el -> el[dimension], case)), ts.u))
            ts.t, this_dim
        end
    end
end

