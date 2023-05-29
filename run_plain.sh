#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/home/linushe/outputs/plain-%j.log

###
# This script is to submitted via "sbatch" on the cluster.
#
# Set --cpus-per-task above to match the size of your multiprocessing run, if any.
###

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Some initial setup
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
module purge

# Load a Julia module, if you're running Julia notebooks
# module load julia/1.8.2

# Start the script
#srun julia --project=. -t8 notebooks/sde_train.jl -m fhn --batch-size 128 --eta 10.0 --learning-rate 0.02 --latent-dims 2 --stick-landing false
#srun /home/linushe/julia-1.9.0/bin/julia --project=. -t5 notebooks/sde_train.jl -m ou --batch-size 128 --eta 0.05 --learning-rate 0.04 --latent-dims 4 --stick-landing false --kl-rate 5000 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 30.0 --tspan-start-train 0.0 --tspan-end-train 30.0 --tspan-start-model 0.0 --tspan-end-model 30.0 --dt 1.0 --hidden-size 32 --backsolve true
# normal sun
#srun /home/linushe/julia-1.9.0/bin/julia --project=. -t7 notebooks/sde_train.jl -m sun --batch-size 128 --eta 10.0 --learning-rate 0.02 --latent-dims 1 --stick-landing false --dt 0.04 --kl-rate 6000 --kl-anneal true --hidden-size 64 --backsolve true --noise 0.5
# long sun
#srun /home/linushe/julia-1.9.0/bin/julia --project=. -t4 notebooks/sde_train.jl -m sun --batch-size 128 --eta 10.0 --learning-rate 0.02 --latent-dims 1 --stick-landing false --dt 0.04 --kl-rate 8000 --kl-anneal true --hidden-size 64 --backsolve true --tspan-end-model 1.5 --tspan-end-data 1.5 --tspan-end-train 1.5 --backsolve true
srun /home/linushe/julia-1.9.0/bin/julia --project=. -t4 notebooks/sde_train.jl -m fhn --batch-size 128 --eta 1.0 --learning-rate 0.02 --latent-dims 2 --stick-landing false --kl-rate 4000 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 5.0 --tspan-start-train 3.0 --tspan-end-train 5.0 --tspan-start-model 2.7 --tspan-end-model 5.0 --dt 0.2 --backsolve true

#srun /home/linushe/julia-1.9.0/bin/julia --project=. -t5 -m ou --batch-size 128 --eta 0.1 --learning-rate 0.02 --latent-dims 2 --stick-landing false --kl-rate 4000 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 4.0 --tspan-start-train 0.0 --tspan-end-train 4.0 --tspan-start-model 0.0 --tspan-end-model 4.0 --dt 0.2 --hidden-size 16

# time dependent OU
#srun /home/linushe/julia-1.9.0/bin/julia --project=. -t4 notebooks/sde_train.jl -m ou --batch-size 128 --eta 0.1 --learning-rate 0.02 --latent-dims 1 --stick-landing false --kl-rate 4000 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 20.0 --tspan-start-train 0.0 --tspan-end-train 20.0 --tspan-start-model 0.0 --tspan-end-model 20.0 --dt 0.5 --hidden-size 32 --prior-size 8 --time-dep true 

# To stop the script, use 'scancel'
