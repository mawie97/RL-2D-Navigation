#!/bin/bash
#SBATCH --job-name=navppo_eval
#SBATCH --output=logs/eval/slurm_%j.out
#SBATCH --error=logs/eval/slurm_%j.err
#SBATCH --partition=scavenge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/eval

module purge
module load Python/3.12.3-GCCcore-13.3.0
source /home/rusu/Thesis_Project/venv/bin/activate

# Avoid 3 processes each trying to use all 4 threads
export OMP_NUM_THREADS=1

python evaluate_trained_model.py > logs/eval/evaluate_trained_model_%j.log 2>&1 &
p1=$!
python evaluate_1.py > logs/eval/evaluate_1_%j.log 2>&1 &
p2=$!
python evaluate_2.py > logs/eval/evaluate_2_%j.log 2>&1 &
p3=$!

wait $p1
wait $p2
wait $p3
