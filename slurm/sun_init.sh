# energy balance model testcases

fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl -m sun --batch-size 256 --dt 1.0 --kl-anneal true --backsolve true --scale 0.02 --lr-cycle false  --tspan-start-data 0.0 --tspan-end-data 4.0 --tspan-start-train 0.0 --tspan-end-train 4.0 --tspan-start-model 0.0 --tspan-end-model 4.0 --context-size 8 --kidger true"

variable_args=(
    # eta
    "--eta 10.0 --learning-rate 0.0015 --latent-dims 1 --hidden-size 32 --noise 0.05 --decay 0.9999 --kl-rate 250 --depth 1"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t2 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
