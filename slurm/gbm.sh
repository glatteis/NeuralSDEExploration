# ornstein-uhlenbeck model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m gbm --batch-size 128 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 1.0 --tspan-start-train 0.0 --tspan-end-train 1.0 --tspan-start-model 0.0 --tspan-end-model 1.0 --dt 0.05 --backsolve true --scale 0.01 --kidger true --context-size 4"

variable_args=(
    # base experiment
    "--decay 0.999 --eta 0.01 --learning-rate 0.005 --latent-dims 4 --hidden-size 100 --kl-rate 400 --depth 1"
    "--decay 0.999 --eta 0.1 --learning-rate 0.005 --latent-dims 4 --hidden-size 100 --kl-rate 400 --depth 1"
    "--decay 0.999 --eta 0.5 --learning-rate 0.005 --latent-dims 4 --hidden-size 100 --kl-rate 400 --depth 1"
    "--decay 0.999 --eta 1.0 --learning-rate 0.005 --latent-dims 4 --hidden-size 100 --kl-rate 400 --depth 1"
    "--decay 0.999 --eta 10.0 --learning-rate 0.005 --latent-dims 4 --hidden-size 100 --kl-rate 400 --depth 1"
)

echo $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t16 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
