#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --output=results/prepare_data_%j.out
#SBATCH --error=results/prepare_data_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "======================================================"
echo "PIR-RAG Data Preparation Job"
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

# Run data preparation scripts
echo "Step 1: Downloading and preparing data..."
python -u scripts/download_data.py

echo "Step 2: Downloading model..."
python -u scripts/download_model.py

echo "Step 3: Generating embeddings..."
python -u scripts/generate_embeddings.py

echo "Step 4: Moving data to data directory..."
mkdir -p data
mv msmarco_data_prepared/* data/ 2>/dev/null || true

echo "Step 5: Preprocessing into size groups..."
python -u scripts/preprocess_size_groups.py \
    --corpus_path data/corpus_10000.csv \
    --embeddings_path data/embeddings_10000.npy \
    --output_dir data/size_groups

echo "======================================================"
echo "Data preparation finished at $(date)"
echo "======================================================"
