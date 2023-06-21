# energy balance model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m sun --batch-size 128 --dt 0.05 --kl-rate 500 --kl-anneal true --backsolve true --scale 0.01 --depth 2 --decay 1.0 --lr-cycle false  --tspan-start-data 0.0 --tspan-end-data 2.0 --tspan-start-train 0.0 --tspan-end-train 2.0 --tspan-start-model 0.0 --tspan-end-model 2.0"

variable_args=(
    # base experiment
    "--eta 8.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 64"

    # beta cycling
    "--eta 4.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 64"
    "--eta 32.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 64"

    # network size 
    "--eta 8.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 8"
    "--eta 8.0 --learning-rate 0.015 --latent-dims 1 --hidden-size 16"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t16 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
