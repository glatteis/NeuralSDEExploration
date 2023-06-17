export timeseries_mean, timeseries_tipping_rate, timeseries_std, timeseries_histogram

function timeseries_mean(timeseries::Timeseries)
end

function timeseries_std(timeseries::Timeseries)
end

function timeseries_tipping_rate(timeseries::Timeseries)
end

function timeseries_histogram(timeseries::Timeseries, range; dim=1)
    ts = vcat([vcat([x[dim] for x in u]...) for u in timeseries.u]...)
	fit(Histogram, ts, range)
end
