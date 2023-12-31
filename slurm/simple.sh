fixed_args="/home/linushe/neuralsdeexploration/notebooks/sde_train.jl --batch-size 128 --dt 0.02 --kl-anneal true --backsolve true --scale 0.01 --lr-cycle false  --tspan-start-data 0.0 --tspan-end-data 0.5 --tspan-start-train 0.0 --tspan-end-train 0.5 --tspan-start-model 0.0 --tspan-end-model 0.5 --context-size 8 --kidger true --kl-rate 1000"

variable_args=(
    "-m exp --eta 0.1 --learning-rate 0.01 --latent-dims 2 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m const --eta 0.1 --learning-rate 0.01 --latent-dims 2 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m sine --eta 0.1 --learning-rate 0.01 --latent-dims 3 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 0.1 --learning-rate 0.01 --latent-dims 3 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 1.0 --learning-rate 0.01 --latent-dims 3 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 5.0 --learning-rate 0.01 --latent-dims 3 --hidden-size 32 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 10.0 --learning-rate 0.01 --latent-dims 3 --hidden-size 32 --noise 0.01 --decay 0.999"

    "-m diffusion --eta 1.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 64 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 10.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 64 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 25.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 64 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 50.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 64 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 100.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 64 --noise 0.01 --decay 0.999"
    "-m diffusion --eta 1000.0 --learning-rate 0.005 --latent-dims 2 --hidden-size 64 --noise 0.01 --decay 0.999"
)

/home/linushe/julia-1.9.0/bin/julia --project=/home/linushe/neuralsdeexploration -t2 $fixed_args ${variable_args[$SLURM_ARRAY_TASK_ID]}

