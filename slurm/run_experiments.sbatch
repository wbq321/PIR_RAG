#!/bin/bash
#SBATCH --job-name=pir_rag_exp
#SBATCH --output=results/pir_rag_exp_%j.out
#SBATCH --error=results/pir_rag_exp_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "======================================================"
echo "PIR-RAG Experiment Job"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE MB"
echo "======================================================"

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Set up conda environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate pir_rag_env

# Run the experiment
echo "Starting PIR-RAG experiments..."
python -u experiments/run_experiments_v2.py \
    --config config.yaml \
    --output_dir ./results

echo "======================================================"
echo "Job finished at $(date)"
echo "======================================================"
