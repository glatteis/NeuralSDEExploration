#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=16:00:00
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

srun /home/linushe/julia-1.9.0/bin/julia --project=. -t16 notebooks/sde_train.jl -m sun --batch-size 256 --eta 8.0 --learning-rate 0.015 --latent-dims 1 --stick-landing false --dt 0.05 --kl-rate 500 --kl-anneal true --hidden-size 64 --backsolve true --scale 0.01 --tspan-start-data 0.0 --tspan-end-data 2.0 --tspan-start-train 0.0 --tspan-end-train 2.0 --tspan-start-model 0.0 --tspan-end-model 2.0 --decay 1.0 --noise 0.1 --depth 2 --lr-cycle false
