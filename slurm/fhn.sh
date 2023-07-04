# fitzhugh-nagumo model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m fhn --batch-size 128 --lr-cycle false --lr-rate 3000 --kl-rate 1000 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 2.0 --tspan-start-train 0.0 --tspan-end-train 2.0 --tspan-start-model 0.0 --tspan-end-model 2.0 --dt 0.08 --backsolve true --decay 0.9995 --kidger true --context-size 16"

variable_args=(
    # base experiment
    "--eta 10.0 --learning-rate 0.025 --latent-dims 3 --hidden-size 64"

    # beta cycling
    "--eta 5.0 --learning-rate 0.025 --latent-dims 3 --hidden-size 64"
    "--eta 50.0 --learning-rate 0.025 --latent-dims 3 --hidden-size 64"
    
    # latent dimension
    "--eta 50.0 --learning-rate 0.015 --latent-dims 3 --hidden-size 64"
    "--eta 50.0 --learning-rate 0.015 --latent-dims 4 --hidden-size 64"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t16 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
