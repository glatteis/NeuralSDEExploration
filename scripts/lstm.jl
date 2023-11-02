using Zygote, LinearAlgebra, Random, Lux, CSV, ComponentArrays, Plots, PGFPlotsX, DataFrames, Optimisers

ENV["GKSwstype"]="nul" # no GTK for plots

pgfplotsx(size=(500, 250))

# inspired by https://lux.csail.mit.edu/dev/tutorials/beginner/3_SimpleRNN

struct TimeseriesLSTM{L, C} <:
    Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
        lstm_cell::L
        classifier::C
end

function TimeseriesLSTM(in_dims, hidden_dims, out_dims)
    return TimeseriesLSTM(Recurrence(LSTMCell(in_dims => hidden_dims)),
        Dense(hidden_dims => out_dims))
end

function (s::TimeseriesLSTM)(x::AbstractArray{T, 3},
    ps::NamedTuple,
    st::NamedTuple) where {T}
    lstm_output, st_ = s.lstm_cell(x, ps.lstm_cell, st.lstm_cell)
    output, st__ = s.classifier(lstm_output, ps.classifier, st.classifier)
    return output
end

function predict(s::TimeseriesLSTM, x::AbstractArray{T, 3}, ps::NamedTuple, st::NamedTuple, steps::Int64) where {T}
    for i in 1:steps
        lstm_output, st_ = s.lstm_cell(x, ps.lstm_cell, st.lstm_cell)
        output, st__ = s.classifier(lstm_output, ps.classifier, st.classifier)
        output = reshape(output, (1, size(output)...))
        x = hcat(x, output)
    end
    return x
end

# Constants
lookback = 20

df = CSV.File("energybalance.csv") |> DataFrame
data = Matrix{Float32}(df)[1:end, 2:end]

ts_length = size(data)[1]

lstm = TimeseriesLSTM(1, 100, 1)

rng = Xoshiro()

ps, st = Lux.setup(rng, lstm)

# p = plot(data)
# savefig(p, "plot.pdf")

num_ts = size(data)[2]

opt_state = Optimisers.setup(Optimisers.Adam(1f-2), ps)

train_loss = []
test_loss = []

for epoch in 1:10
    println("Epoch $epoch")
    for batch in 1:num_ts
    # for batch in 1:1
        function pass!(ts, do_update)
            windows = reduce(hcat, [ts[i:i + lookback - 1] for i in 1:(length(ts) - lookback + 1)])
            # add a first dimension
            windows = reshape(windows, (1, size(windows)...))

            inputs = collect(windows[:, 1:end-1, :])
            outputs = ts[lookback:end]

            function loss(ps)
                results = lstm(inputs, ps, st)
                sum(abs2, outputs - results[1, :])
            end
            if do_update
                dps = Zygote.gradient(ps -> loss(ps), ps)[1]
                Optimisers.update!(opt_state, ps, dps)
            end
            return loss(ps)
        end

        ts_train = data[1:ts_length÷2, batch]
        ts_test = data[ts_length÷2+1:end, batch]

        training_loss = pass!(ts_train, true)
        testing_loss = pass!(ts_test, false)
        
        push!(train_loss, training_loss)
        push!(test_loss, testing_loss)
        
        println("Training loss: $training_loss, Testing loss: $testing_loss")
    end

    input_length = 100
    prediction_length = 100

    beginning = data[ts_length÷2-input_length:ts_length÷2, 1:10]
    beginning = reshape(beginning, (1, size(beginning)...))
    prediction = predict(lstm, beginning, ps, st, prediction_length)

    p = plot((1:input_length)/10, prediction[1, 1:input_length, :], label="Ground Truth", color="blue", xlabel="Time \$t\$", ylabel="Normalized temperature \$T\$")
    plot!(p, (input_length+1:input_length+prediction_length)/10, prediction[1, input_length+1:input_length+prediction_length, :], label="LSTM Prediction", color="orange")
    # Label hacking
    for i in 1:20
        if i != 1 && i != 11
            p.series_list[i][:label] = ""
            # p.series_list[i][:linecolor] = RGBA{Float64}(0.0,0.0,0.0,0.0)
        end
    end
    savefig(p, "plot$epoch.pdf")
    savefig(p, "plot$epoch.tikz")
end
