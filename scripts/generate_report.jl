using NeuralSDEExploration, Plots, PlutoUI, PlutoArgs, JLD2, Random, StatsBase, Random123, DiffEqNoiseProcess, Distributions, PGFPlotsX

ENV["GKSwstype"]="nul" # no GTK for plots

pgfplotsx()

for f in ARGS
    # Extract everything from the file
    dict = load(f)["data"]

    latent_sde = dict["latent_sde"]
    ps = dict["ps"]
    st = dict["st"]
    model = dict["model"]
    timeseries = dict["timeseries"]
    
    # TODO replace with datamin/datamax from dict
    datamin = -7.2356f0
    datamax = 6.890989f0
    function normalize(x)
        return ((x - datamin) / (datamax - datamin))
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
    extended_tspan = extrapolate_tspan(latent_sde.tspan, 3f0)
    datasize = latent_sde.datasize * 16

    sample_size = 500
    latent_sde_sample = NeuralSDEExploration.sample_prior_dataspace(latent_sde,ps,st;b=sample_size, tspan=extended_tspan, datasize=datasize)

    # TODO replace with initial condition from .jld file
    initial_condition_hack = [[0f0, 0f0] for i in 1:sample_size]
    data_model_sample = map_dims(x -> map(normalize, x), filter_dims(1:1, NeuralSDEExploration.series(model, initial_condition_hack, extended_tspan, datasize)))

    function moment_analysis()
        mean_and_var_latent_sde = map_ts((ts) -> [mean(ts), std(ts)], latent_sde_sample)
        mean_and_var_data_model = map_ts((ts) -> [mean(ts), std(ts)], data_model_sample)

        p = plot(select_ts(1:1, mean_and_var_latent_sde), ribbon=2*map(only, mean_and_var_latent_sde.u[2]), label="Latent SDE")
        plot!(p, select_ts(1:1, mean_and_var_data_model), ribbon=2*map(only, mean_and_var_data_model.u[2]), label="Data Model")

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
            histogram_latent_sde = timeseries_histogram(select_tspan(tspan, latent_sde_sample), 0.0:0.01:1.0)
            histogram_data_model = timeseries_histogram(select_tspan(tspan, data_model_sample), 0.0:0.01:1.0)
            p_hist = plot([
                histogram_latent_sde,
                histogram_data_model,
            ], alpha=0.5, labels=["data model" "latent sde"], title=title)
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

        # ╔═╡ fe1ae4b3-2f1f-4b6c-a076-0d215f222e6c
        plot_prior(25, rng=Xoshiro(), tspan=(0f0, 10f0), datasize=100)

        # ╔═╡ 72045fd0-2769-4868-9b67-e7a41e3f1d7d
        plot(NeuralSDEExploration.sample_prior(latent_sde,ps,st;b=1,tspan=(0f0,10f0),datasize=5000))

        # ╔═╡ ae0c9dae-e490-4965-9353-c820a3ce3645
        plot(NeuralSDEExploration.sample_prior_dataspace(latent_sde,ps,st;b=1,tspan=(0f0,10f0),datasize=5000,seed=1), title="Neural SDE")

        # ╔═╡ 63e3e80c-6d03-4e42-a718-5bf30ad7182f
        plot(filter_dims(1:1, NeuralSDEExploration.series(model, [[0f0, 0f0]], (0f0, 10f0), 5000,seed=1)), title="FitzHugh-Nagumo")

        # ╔═╡ 9f8c49a0-2098-411b-976a-2b43cbb20a44
        plot(NeuralSDEExploration.series(model, [[0f0, 0f0]], (0f0, 10f0), 5000))

        # ╔═╡ ff5519b9-2a69-41aa-8f55-fc63fa176d3f
        plot(sample(timeseries, 25),linewidth=.5,color=:black,legend=false,title="data")
    end
end
