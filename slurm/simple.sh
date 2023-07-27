# energy balance model testcases

fixed_args="/home/linushe/neuralsdeexploration_gpu/notebooks/sde_train.jl --batch-size 256 --dt 0.02 --kl-anneal true --backsolve true --scale 0.01 --lr-cycle false  --tspan-start-data 0.0 --tspan-end-data 0.5 --tspan-start-train 0.0 --tspan-end-train 0.5 --tspan-start-model 0.0 --tspan-end-model 0.5 --context-size 8 --kidger true"

variable_args=(
    "-m exp --eta 1.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m const --eta 1.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 1.0 --learning-rate 0.005 --latent-dims 1 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m sin --eta 1.0 --learning-rate 0.005 --latent-dims 3 --hidden-size 32 --noise 0.01 --decay 0.999"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration_gpu -t2 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}
