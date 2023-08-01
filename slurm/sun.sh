# energy balance model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m sun --batch-size 128 --dt 0.2 --kl-anneal true --backsolve true --scale 0.01 --lr-cycle false  --tspan-start-data 0.0 --tspan-end-data 4.0 --tspan-start-train 0.0 --tspan-end-train 4.0 --tspan-start-model 0.0 --tspan-end-model 4.0 --context-size 8 --kidger true"

variable_args=(
    # eta
    "--eta 20.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 150 --noise 0.15 --decay 0.999 --kl-rate 500 --depth 1"
    "--eta 50.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 150 --noise 0.15 --decay 0.999 --kl-rate 500 --depth 1"
    "--eta 100.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 150 --noise 0.15 --decay 0.999 --kl-rate 500 --depth 1"
    "--eta 500.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 150 --noise 0.15 --decay 0.999 --kl-rate 500 --depth 1"
    "--eta 1000.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 150 --noise 0.15 --decay 0.999 --kl-rate 500 --depth 1"
    "--eta 10000.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 150 --noise 0.15 --decay 0.999 --kl-rate 500 --depth 1"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t1 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
