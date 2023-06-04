#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --cpus-per-task=10
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


# time dependent OU
srun /home/linushe/julia-1.9.0/bin/julia --project=. -t10 notebooks/sde_train.jl -m ou --batch-size 128 --eta 1.0 --learning-rate 0.02 --latent-dims 3 --stick-landing false --kl-rate 5000 --kl-anneal true --tspan-start-data 0.0 --tspan-end-data 20.0 --tspan-start-train 5.0 --tspan-end-train 20.0 --tspan-start-model 0.0 --tspan-end-model 20.0 --dt 0.4 --hidden-size 64 --backsolve true #--time-dep true
