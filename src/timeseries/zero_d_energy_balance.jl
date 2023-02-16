using DifferentialEquations

# From https://github.com/TUM-PIK-ESM/TUM-Dynamics-Lecture/blob/main/lectures/lecture-8/lecture8.ipynb

struct ZeroDEnergyBalanceModel <: Timeseries
    albedo_0 :: Float64
    albedo_var :: Float64
    solarconstant :: Float64
    radiation :: Float64
    noise_var :: Float64
end

ZeroDEnergyBalanceModel() = ZeroDEnergyBalanceModel(0.5, 0.4, 1363, 0.6 * 5.67e-8, 0.05)

# albedo decreases with increasing temperature
albedo(t, ebm :: ZeroDEnergyBalanceModel) = ebm.albedo_0 - (ebm.albedo_var / 2) * tanh(t - 273)

energy_in(t, ebm :: ZeroDEnergyBalanceModel) = (1 - albedo(t, ebm)) * (ebm.solarconstant / 4)

energy_out(t, ebm :: ZeroDEnergyBalanceModel) = ebm.radiation * t^4

drift(u, ebm :: ZeroDEnergyBalanceModel, t) = energy_in(u[1], ebm) - energy_out(u[1], ebm)

diffusion(u, ebm :: ZeroDEnergyBalanceModel, t) = ebm.noise_var * u

function series(ebm :: ZeroDEnergyBalanceModel, u0, tspan, datasize)
    problem = SDEProblem(drift, diffusion, u0, tspan, ebm)
    t = range(tspan[1],tspan[2],length=datasize)
    solve(problem, saveat=t)
end

ylabel(ebm :: ZeroDEnergyBalanceModel) = "temperature [K]"
