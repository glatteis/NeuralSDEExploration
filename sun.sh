#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
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

srun /home/linushe/julia-1.9.0/bin/julia --project=. -t16 notebooks/sde_train.jl -m sun --batch-size 128 --eta 0.1 --learning-rate 0.02 --latent-dims 1 --stick-landing false --dt 0.03 --kl-rate 6000 --kl-anneal true --hidden-size 64 --backsolve true
