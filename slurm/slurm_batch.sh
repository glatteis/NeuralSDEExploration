#!/bin/bash

#SBATCH --qos=gpushort
#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=12:00:00
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
