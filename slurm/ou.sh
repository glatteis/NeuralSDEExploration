# ornstein-uhlenbeck model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m ou --batch-size 128 --latent-dims 3 --stick-landing false --kl-rate 500 --kl-anneal true --lr-cycle false --tspan-start-data 0.0 --tspan-end-data 80.0 --tspan-start-train 0.0 --tspan-end-train 80.0 --tspan-start-model 0.0 --tspan-end-model 80.0 --dt 2.0 --backsolve true --scale 0.01 --decay 1.0"

variable_args=(
    # base experiment
    "--eta 0.1 --learning-rate 0.02 --latent-dims 3 --hidden-size 64"

    # beta cycling
    "--eta 0.01 --learning-rate 0.02 --latent-dims 3 --hidden-size 64"
    "--eta 0.5 --learning-rate 0.02 --latent-dims 3 --hidden-size 64"

    # learning rate
    "--eta 0.1 --learning-rate 0.01 --latent-dims 3 --hidden-size 64"
    "--eta 0.1 --learning-rate 0.05 --latent-dims 3 --hidden-size 64"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t16 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
