# energy balance model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m sun --batch-size 256 --dt 0.02 --kl-anneal true --backsolve true --scale 0.005 --lr-cycle false  --tspan-start-data 0.0 --tspan-end-data 0.5 --tspan-start-train 0.0 --tspan-end-train 0.5 --tspan-start-model 0.0 --tspan-end-model 0.5 --context-size 8 --kidger true"

variable_args=(
    # eta
    "--eta 0.5 --learning-rate 0.005 --latent-dims 2 --hidden-size 32 --noise 0.12 --decay 1.0 --kl-rate 1000 --depth 2"
    "--eta 1.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 32 --noise 0.12 --decay 1.0 --kl-rate 1000 --depth 2"
    "--eta 5.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 32 --noise 0.12 --decay 1.0 --kl-rate 1000 --depth 2"
    "--eta 10.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 32 --noise 0.12 --decay 1.0 --kl-rate 1000 --depth 2"
    "--eta 10.0 --learning-rate 0.005  --latent-dims 2 --hidden-size 32 --noise 0.12 --decay 1.0 --kl-rate 1000 --depth 2"
    "--eta 20.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 32 --noise 0.12 --decay 1.0 --kl-rate 1000 --depth 2"

    # different noisiness
    "--eta 1.0 --learning-rate 0.0015 --latent-dims 1 --hidden-size 32 --noise 0.01 --decay 0.999"
    "--eta 0.1 --learning-rate 0.0015 --latent-dims 1 --hidden-size 32 --noise 0.05"
    "--eta 8.0 --learning-rate 0.0015 --latent-dims 1 --hidden-size 32 --noise 0.1"
    "--eta 8.0 --learning-rate 0.0015 --latent-dims 1 --hidden-size 32 --noise 0.15"
    "--eta 8.0 --learning-rate 0.0015 --latent-dims 1 --hidden-size 32 --noise 0.2"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t2 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
