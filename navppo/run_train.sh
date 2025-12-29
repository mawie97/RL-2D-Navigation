#!/bin/bash
#SBATCH --job-name=navppo_train
#SBATCH --output=logs/slurm.out
#SBATCH --error=logs/slurm.err
#SBATCH --partition=cores
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=7-00:00:00

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load Python/3.12.3-GCCcore-13.3.0

source /home/rusu/Thesis_Project/venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_train_env.py


