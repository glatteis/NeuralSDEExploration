# fitzhugh-nagumo model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m fhn --batch-size 128 --lr-cycle false --lr-rate 3000 --kl-rate 1000 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 0.25 --tspan-start-train 0.0 --tspan-end-train 0.25 --tspan-start-model 0.0 --tspan-end-model 0.25 --dt 0.01 --backsolve true --decay 1.0 --kidger true"

variable_args=(
    # base experiment
    "--eta 50.0 --learning-rate 0.015 --latent-dims 2 --hidden-size 8"

    # beta cycling
    "--eta 10.0 --learning-rate 0.015 --latent-dims 2 --hidden-size 8"
    "--eta 100.0 --learning-rate 0.015 --latent-dims 2 --hidden-size 8"
    
    # latent dimension
    "--eta 50.0 --learning-rate 0.015 --latent-dims 3 --hidden-size 64"
    "--eta 50.0 --learning-rate 0.015 --latent-dims 4 --hidden-size 64"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t16 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
