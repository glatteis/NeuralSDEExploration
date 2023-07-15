# energy balance model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m sun --batch-size 128 --dt 0.05 --kl-anneal true --backsolve false --scale 0.01 --lr-cycle false  --tspan-start-data 0.0 --tspan-end-data 3.0 --tspan-start-train 0.0 --tspan-end-train 3.0 --tspan-start-model 0.0 --tspan-end-model 3.0 --context-size 8 --kidger true"

variable_args=(
    # eta
    "--eta 5.0 --learning-rate 0.001 --latent-dims 1 --hidden-size 32 --noise 0.05 --decay 0.999 --kl-rate 2500 --depth 1"
    "--eta 5.0 --learning-rate 0.001 --latent-dims 1 --hidden-size 32 --noise 0.10 --decay 0.999 --kl-rate 2500 --depth 1"
    "--eta 5.0 --learning-rate 0.001 --latent-dims 1 --hidden-size 32 --noise 0.15 --decay 0.999 --kl-rate 2500 --depth 1"
    "--eta 5.0 --learning-rate 0.001 --latent-dims 1 --hidden-size 32 --noise 0.20 --decay 0.999 --kl-rate 2500 --depth 1"

    # different noisiness
    "--eta 1.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 32 --noise 0.01 --decay 0.999"
    "--eta 0.1 --learning-rate 0.015 --latent-dims 1 --hidden-size 32 --noise 0.05"
    "--eta 8.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 32 --noise 0.1"
    "--eta 8.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 32 --noise 0.15"
    "--eta 8.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 32 --noise 0.2"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t2 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
