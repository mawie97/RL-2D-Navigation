#!/bin/bash
#SBATCH --job-name=navppo_train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=scavenge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

source ../venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs

python run_train_env.py
