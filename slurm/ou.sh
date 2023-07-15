# ornstein-uhlenbeck model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m ou --batch-size 128 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 20.0 --tspan-start-train 0.0 --tspan-end-train 20.0 --tspan-start-model 0.0 --tspan-end-model 20.0 --dt 0.6 --backsolve true --scale 0.005 --kidger true --context-size 16"

variable_args=(
    # base experiment
    "--decay 0.999 --eta 0.001 --learning-rate 0.0005 --latent-dims 4 --hidden-size 32 --kl-rate 1500"
    "--decay 0.999 --eta 0.01 --learning-rate 0.0005 --latent-dims 4 --hidden-size 32 --kl-rate 1500"

    # learning rate
    "--decay 0.999 --eta 0.1 --learning-rate 0.005 --latent-dims 4 --hidden-size 32 --kl-rate 500"
    "--decay 0.999 --eta 1.0 --learning-rate 0.005 --latent-dims 4 --hidden-size 32 --kl-rate 500"

    # kl rate
    "--decay 0.995 --eta 0.01 --learning-rate 0.005 --latent-dims 4 --hidden-size 32 --kl-rate 250"
    "--decay 0.995 --eta 0.01 --learning-rate 0.005 --latent-dims 4 --hidden-size 32 --kl-rate 1000"

)

echo $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t16 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
