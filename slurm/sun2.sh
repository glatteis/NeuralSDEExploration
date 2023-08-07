# energy balance model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m sun --batch-size 128 --dt 0.2 --kl-anneal true --backsolve true --scale 0.01 --lr-cycle false  --tspan-start-data 0.0 --tspan-end-data 4.0 --tspan-start-train 0.0 --tspan-end-train 4.0 --tspan-start-model 0.0 --tspan-end-model 4.0 --context-size 8 --kidger true"

variable_args=(
    # eta
    "--eta 0.0001 --learning-rate 0.005 --latent-dims 1 --hidden-size 50 --noise 0.15 --decay 0.999 --kl-rate 1000 --depth 2"
    "--eta 0.001 --learning-rate 0.005 --latent-dims 1 --hidden-size 50 --noise 0.15 --decay 0.999 --kl-rate 1000 --depth 2"
    "--eta 0.01 --learning-rate 0.005 --latent-dims 1 --hidden-size 50 --noise 0.15 --decay 0.999 --kl-rate 1000 --depth 2"
    "--eta 0.1 --learning-rate 0.005 --latent-dims 1 --hidden-size 50 --noise 0.15 --decay 0.999 --kl-rate 1000 --depth 2"
    "--eta 1.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 50 --noise 0.15 --decay 0.999 --kl-rate 1000 --depth 2"
    "--eta 10.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 50 --noise 0.15 --decay 0.999 --kl-rate 1000 --depth 2"
)

/home/linushe/julia-1.9.0/bin/julia -t4 --project=/home/linushe/neuralsdeexploration $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
