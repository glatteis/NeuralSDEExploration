Main questions:
- can neural sdes learn from bistable timeseries data and show the same statistical features?
    - what amount of data is needed?
    - how big are the networks?

Steps:

Minimal:
First steps:
- finish latent sde implementation, write tests (2 weeks)
    - goal: latent sde trains on simple data
- research methods to compare statistical features and apply them to latent sde / simple model (1 week)
    - goal: some way to quantify how good the model is
        - goal: some way to quantify how good the model is

- train generative models on advanced data
    - research more advanced models / why are we using these?
        - FitzHugh-Nagumo (and a simple model based on it?)
        - model by Keno Riechers: https://arxiv.org/abs/2303.04063
        - solar power data

- systematic research into multiple options
    - dimensions:
        - size of networks / dimensionality of latent space
        - amount of data
        - type of network used as encoder (latent sde)

Nice-to-have:
- sde-gan implementation
