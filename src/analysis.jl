export mean_and_var, mean_of_means, tipping_rate, tipping_rate_dim, timeseries_histogram

function mean_and_var(timeseries::Timeseries)
    map_ts((ts) -> [mean(ts), std(ts)], timeseries)
end

function mean_of_means(timeseries::Timeseries)
    mean_var = mean_and_var(timeseries)
    mean_of_means = []
    mean_of_means = []
    for dimension in 1:length(mean_var.u[1][1])
        push!(mean_of_means, mean([timeseries.u[1][time][dimension] for time in 1:length(timeseries.t)]))
    end
    mean_of_means
end

function tipping_rate(timeseries::Timeseries)
    # lines where tipping is registered if the timeseries passes
    lines = mean_of_means(timeseries)
    [tipping_rate_dim(timeseries, dim, lines[dim]) for dim in eachindex(lines)]
end

function tipping_rate_dim(timeseries::Timeseries, dim::Int, line)
    # vcat every timeseries together, because the tipping rate is an average
    # over all timeseries
    big_timeseries = reduce(vcat, [map(x -> x[dim], timeseries.u[i]) for i in 1:length(timeseries.u)])
    counted_tips = 0
    for i in 1:length(big_timeseries) - 1
        if (big_timeseries[i] < line && big_timeseries[i + 1] >= line) ||
            (big_timeseries[i] >= line && big_timeseries[i] < line)
            counted_tips += 1
        end
    end
    counted_tips / length(big_timeseries)
end

function timeseries_histogram(timeseries::Timeseries, range; dim=1)
    ts = vcat([vcat([x[dim] for x in u]...) for u in timeseries.u]...)
	fit(Histogram, ts, range)
end
