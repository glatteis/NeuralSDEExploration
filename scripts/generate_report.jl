using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs, JLD2, Random, StatsBase, Random123, DiffEqNoiseProcess, Distributions, PGFPlotsX

ENV["GKSwstype"]="nul" # no GTK for plots

pgfplotsx()

for f in ARGS
    try
        # Extract everything from the file
        dict = load(f)["data"]

        latent_sde = dict["latent_sde"]
        ps = dict["ps"]
        st = dict["st"]
        model = dict["model"]
        timeseries = dict["timeseries"]
        
        datamin = dict["datamin"]
        datamax = dict["datamax"]
        function normalize(x)
            return 2f0 * (((x - datamin) / (datamax - datamin)) - 0.5f0)
        end

        seed = 1

        function savefigure(name, figure)
            path = join(split(f, ".")[1:end-1], ".")
            plot_name = path * "_" * name
            println("Saving plot $plot_name")
            savefig(figure, plot_name * ".tex")
            savefig(figure, plot_name * ".pdf")
        end

        function plot_prior(priorsamples; rng=rng, tspan=latent_sde.tspan, datasize=latent_sde.datasize)
            prior = NeuralSDEExploration.sample_prior_dataspace(latent_sde,ps,st;seed=abs(rand(rng, Int)),b=priorsamples, tspan=tspan, datasize=datasize)
            return plot(prior, linewidth=.5,color=:black,legend=false,title="projected prior")
        end

        function train_plot()
            viz_batch = select_ts(1:5, timeseries)
            posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(viz_batch, ps, st)

            savefigure("kl_div", plot(logterm_[1, :, :]', title="KL-Divergence", linewidth=2))

            function plotmodel()
                n = 5
                rng_plot = Xoshiro()
                nums = sample(rng_plot, 1:length(timeseries.u), n; replace=false)
                
                posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(select_ts(nums, timeseries), ps, st, seed=seed)
                
                priorsamples = 25
                priornums = sample(rng_plot, 1:length(timeseries.u), priorsamples; replace=false)
                priorplot = plot_prior(priorsamples, rng=rng_plot)

                posteriorplot = plot(timeseries.t, posterior_data[1, :,:]', linewidth=2, legend=false, title="projected posterior")
                dataplot = plot(select_ts(nums, timeseries), linewidth=2, legend=false, title="data")
                
                timeseriesplot = plot(select_ts(priornums, timeseries), linewidth=.5, color=:black, legend=false, title="data")
                
                l = @layout [a b ; c d]
                p = plot(dataplot, posteriorplot, timeseriesplot, priorplot, layout=l)
                p
            end

            savefigure("model", plotmodel())
        end

        # four times training tspan
        function extrapolate_tspan(tspan, factor)
            (tspan[1], tspan[2] + factor * (tspan[2] - tspan[1]))
        end
        extended_tspan = extrapolate_tspan(latent_sde.tspan, 3e0)
        datasize = latent_sde.datasize * 16

        sample_size = 500
        latent_sde_sample = NeuralSDEExploration.sample_prior_dataspace(latent_sde,ps,st;b=sample_size, tspan=extended_tspan, datasize=datasize)

        initial_condition = dict["initial_condition"][1:sample_size]
        data_model_sample = map_dims(x -> map(normalize, x), filter_dims(1:1, NeuralSDEExploration.series(model, initial_condition, extended_tspan, datasize)))

        function moment_analysis()
            mean_and_var_latent_sde = map_ts((ts) -> [mean(ts), std(ts)], latent_sde_sample)
            mean_and_var_data_model = map_ts((ts) -> [mean(ts), std(ts)], data_model_sample)

            p = plot(select_ts(1:1, mean_and_var_latent_sde), ribbon=2*map(only, mean_and_var_latent_sde.u[2]), label="Latent SDE")
            plot!(p, select_ts(1:1, mean_and_var_data_model), ribbon=2*map(only, mean_and_var_data_model.u[2]), label="Data Model")
            plot!(p, legend=true, title="mean and 95th percentile")

            savefigure("mean_var", p)
        end
        
        function plot_histograms()
            tspans = Dict(
                "training timespan" => latent_sde.tspan,
                "second half of training timespan" => (latent_sde.tspan[1] + 0.5 * (latent_sde.tspan[2] - latent_sde.tspan[1]), latent_sde.tspan[2]),
                "1x extrapolation" => (latent_sde.tspan[2], extrapolate_tspan(latent_sde.tspan, 1)[2]),
                "4x extrapolation" => (latent_sde.tspan[2], extrapolate_tspan(latent_sde.tspan, 4)[2]),
            )
            for (title, tspan) in tspans
                histogram_latent_sde = timeseries_histogram(select_tspan(tspan, latent_sde_sample), -2.0:0.04:2.0)
                histogram_data_model = timeseries_histogram(select_tspan(tspan, data_model_sample), -2.0:0.04:2.0)
                p_hist = plot([
                    histogram_latent_sde,
                    histogram_data_model,
                ], alpha=0.5, labels=["Latent SDE" "Data Model"], title=title)
                savefigure("histogram_" * replace(title, " " => "_"), p_hist)
            end
        end
        
        train_plot()
        moment_analysis()
        plot_histograms()

        function generate_json()
            function loss(ps, minibatch, seed)
                _, _, _, kl_divergence, likelihood = latent_sde(minibatch, ps, st, seed=seed)
                mean(kl_divergence), mean(likelihood)
            end

            kl_divergence, likelihood = loss(ps, select_ts(timeseries, 1:1000), seed)
        end
    catch y
        @warn "Could not generate report for $f because of $y"
    end
end
