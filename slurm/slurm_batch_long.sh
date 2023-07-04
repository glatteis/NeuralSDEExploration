#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/home/linushe/outputs/%x.%A_%4a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=linus.heck@rwth-aachen.de

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Some initial setup
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
module purge

# run specified script with cpus and task id
./$1
