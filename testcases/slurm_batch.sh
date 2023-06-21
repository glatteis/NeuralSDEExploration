#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --cpus-per-task=$2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/home/linushe/outputs/plain-%j.log
#SBATCH --mail-user=linus.heck@rwth-aachen.de

# run specified script with cpus and task id
./$1 $2 $SLURM_ARRAY_TASK_ID 
