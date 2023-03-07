export ZeroDEnergyBalanceModel

# From https://github.com/TUM-PIK-ESM/TUM-Dynamics-Lecture/blob/main/lectures/lecture-8/lecture8.ipynb

struct ZeroDEnergyBalanceModel <: Timeseries
    albedo_0
    albedo_var
    solarconstant
    radiation
    noise_var
end

# from the lecture
ZeroDEnergyBalanceModelNonStochastic() = ZeroDEnergyBalanceModel(0.5, 0.4, 1363, 0.6 * 5.67e-8, 0)

# modified a bit for some nice tipping
ZeroDEnergyBalanceModel() = ZeroDEnergyBalanceModel(0.425, 0.4, 1363, 0.6 * 5.67e-8, 0.08)

# albedo decreases with increasing temperature
albedo(t, ebm::ZeroDEnergyBalanceModel) = ebm.albedo_0 - (ebm.albedo_var / 2) * tanh(t - 273)

energy_in(t, ebm::ZeroDEnergyBalanceModel) = (1 - albedo(t, ebm)) * (ebm.solarconstant / 4)

energy_out(t, ebm::ZeroDEnergyBalanceModel) = ebm.radiation * t^4

drift(u, ebm::ZeroDEnergyBalanceModel, t) = energy_in(u[1], ebm) - energy_out(u[1], ebm)

diffusion(u, ebm::ZeroDEnergyBalanceModel, t) = ebm.noise_var * u

function series(ebm::ZeroDEnergyBalanceModel, u0s, tspan, datasize)
    t = range(tspan[1], tspan[2], length=datasize)
    [solve(SDEProblem(drift, diffusion, u0, tspan, ebm, noise=WienerProcess(0.0, 0.0)), saveat=t) for u0 in u0s]
end

ylabel(ebm::ZeroDEnergyBalanceModel) = "temperature [K]"
