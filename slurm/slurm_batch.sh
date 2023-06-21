#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=/home/linushe/outputs/plain-%j.log
#SBATCH --mail-user=linus.heck@rwth-aachen.de

# run specified script with cpus and task id
./$1
