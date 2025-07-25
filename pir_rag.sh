#!/bin/bash
"""
PIR-RAG Project Management Script

This script helps manage the PIR-RAG project with common tasks.
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
SLURM_DIR="$PROJECT_ROOT/slurm"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"

# Ensure directories exist
mkdir -p "$DATA_DIR" "$RESULTS_DIR"

print_usage() {
    echo "PIR-RAG Project Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup           - Set up conda environment and install dependencies"
    echo "  download        - Download data and model (run on login node)"
    echo "  preprocess      - Preprocess data into size groups (run locally)"
    echo "  prepare         - Submit data preparation job to SLURM (includes preprocessing)"
    echo "  experiment      - Submit experiment job to SLURM"
    echo "  run-local       - Run experiments locally (not on SLURM)"
    echo "  clean           - Clean up generated files"
    echo "  status          - Check project status"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 download"
    echo "  $0 preprocess"
    echo "  $0 prepare"
    echo "  $0 experiment"
}

setup_environment() {
    echo -e "${BLUE}Setting up PIR-RAG environment...${NC}"
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Error: conda not found. Please install Anaconda/Miniconda first.${NC}"
        exit 1
    fi
    
    # Create environment if it doesn't exist
    if ! conda env list | grep -q "pir_rag_env"; then
        echo -e "${YELLOW}Creating conda environment 'pir_rag_env'...${NC}"
        conda create -n pir_rag_env python=3.8 -y
    else
        echo -e "${GREEN}Environment 'pir_rag_env' already exists.${NC}"
    fi
    
    # Activate environment and install dependencies
    echo -e "${YELLOW}Installing dependencies...${NC}"
    eval "$(conda shell.bash hook)"
    conda activate pir_rag_env
    pip install -r requirements.txt
    
    echo -e "${GREEN}Environment setup complete!${NC}"
}

download_data() {
    echo -e "${BLUE}Downloading data and model...${NC}"
    
    # Check if we're in the right environment
    if [[ "$CONDA_DEFAULT_ENV" != "pir_rag_env" ]]; then
        echo -e "${YELLOW}Activating pir_rag_env...${NC}"
        eval "$(conda shell.bash hook)"
        conda activate pir_rag_env
    fi
    
    # Download data
    echo -e "${YELLOW}Downloading MS MARCO data...${NC}"
    python "$SCRIPTS_DIR/download_data.py"
    
    # Download model
    echo -e "${YELLOW}Downloading BGE model...${NC}"
    python "$SCRIPTS_DIR/download_model.py"
    
    echo -e "${GREEN}Download complete!${NC}"
}

preprocess_data() {
    echo -e "${BLUE}Preprocessing data into size groups...${NC}"
    
    # Check if basic data exists
    if [[ ! -f "$DATA_DIR/embeddings_10000.npy" ]] || [[ ! -f "$DATA_DIR/corpus_10000.csv" ]]; then
        echo -e "${RED}Error: Basic data not found. Run '$0 download' and prepare embeddings first.${NC}"
        exit 1
    fi
    
    # Check if we're in the right environment
    if [[ "$CONDA_DEFAULT_ENV" != "pir_rag_env" ]]; then
        echo -e "${YELLOW}Activating pir_rag_env...${NC}"
        eval "$(conda shell.bash hook)"
        conda activate pir_rag_env
    fi
    
    # Run preprocessing
    python "$SCRIPTS_DIR/preprocess_size_groups.py" \
        --corpus_path "$DATA_DIR/corpus_10000.csv" \
        --embeddings_path "$DATA_DIR/embeddings_10000.npy" \
        --output_dir "$DATA_DIR/size_groups"
    
    echo -e "${GREEN}Data preprocessing complete!${NC}"
}

submit_data_preparation() {
    echo -e "${BLUE}Submitting data preparation job...${NC}"
    
    if [[ ! -f "$SLURM_DIR/prepare_data.sbatch" ]]; then
        echo -e "${RED}Error: SLURM script not found at $SLURM_DIR/prepare_data.sbatch${NC}"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    JOB_ID=$(sbatch "$SLURM_DIR/prepare_data.sbatch" | awk '{print $4}')
    
    echo -e "${GREEN}Data preparation job submitted with ID: $JOB_ID${NC}"
    echo -e "${YELLOW}This job will download data, generate embeddings, AND preprocess into size groups.${NC}"
    echo -e "${YELLOW}Monitor with: squeue -u $USER${NC}"
    echo -e "${YELLOW}Check logs in: $RESULTS_DIR/prepare_data_${JOB_ID}.out${NC}"
}

submit_experiment() {
    echo -e "${BLUE}Submitting experiment job...${NC}"
    
    # Check if data is ready (including size groups)
    if [[ ! -d "$DATA_DIR/size_groups" ]]; then
        echo -e "${RED}Error: Size groups not prepared. Run '$0 prepare' or '$0 preprocess' first.${NC}"
        exit 1
    fi
    
    if [[ ! -f "$SLURM_DIR/run_experiments.sbatch" ]]; then
        echo -e "${RED}Error: SLURM script not found at $SLURM_DIR/run_experiments.sbatch${NC}"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
    JOB_ID=$(sbatch "$SLURM_DIR/run_experiments.sbatch" | awk '{print $4}')
    
    echo -e "${GREEN}Experiment job submitted with ID: $JOB_ID${NC}"
    echo -e "${YELLOW}Monitor with: squeue -u $USER${NC}"
    echo -e "${YELLOW}Check logs in: $RESULTS_DIR/pir_rag_exp_${JOB_ID}.out${NC}"
}

run_local() {
    echo -e "${BLUE}Running experiments locally...${NC}"
    
    # Check if data is ready (including size groups)
    if [[ ! -d "$DATA_DIR/size_groups" ]]; then
        echo -e "${RED}Error: Size groups not prepared. Run '$0 prepare' or '$0 preprocess' first.${NC}"
        exit 1
    fi
    
    # Check if we're in the right environment
    if [[ "$CONDA_DEFAULT_ENV" != "pir_rag_env" ]]; then
        echo -e "${YELLOW}Activating pir_rag_env...${NC}"
        eval "$(conda shell.bash hook)"
        conda activate pir_rag_env
    fi
    
    cd "$PROJECT_ROOT"
    python experiments/run_experiments_v2.py \
        --config config.yaml \
        --output_dir "$RESULTS_DIR"
}

clean_files() {
    echo -e "${BLUE}Cleaning up generated files...${NC}"
    
    # Remove data files
    if [[ -d "$DATA_DIR" ]]; then
        echo -e "${YELLOW}Removing data directory...${NC}"
        rm -rf "$DATA_DIR"
    fi
    
    # Remove model
    if [[ -d "../shared_models/bge_model" ]]; then
        echo -e "${YELLOW}Removing shared model...${NC}"
        rm -rf "../shared_models/bge_model"
    fi
    
    # Remove cached data
    if [[ -d "./msmarco_data_prepared" ]]; then
        echo -e "${YELLOW}Removing prepared data cache...${NC}"
        rm -rf "./msmarco_data_prepared"
    fi
    
    echo -e "${GREEN}Cleanup complete!${NC}"
}

check_status() {
    echo -e "${BLUE}PIR-RAG Project Status${NC}"
    echo "========================"
    
    # Check environment
    if conda env list | grep -q "pir_rag_env"; then
        echo -e "Environment: ${GREEN}✓ pir_rag_env exists${NC}"
    else
        echo -e "Environment: ${RED}✗ pir_rag_env not found${NC}"
    fi
    
    # Check data
    if [[ -d "$DATA_DIR/size_groups" ]]; then
        SIZE_GROUPS=$(find "$DATA_DIR/size_groups" -name "*.csv" 2>/dev/null | wc -l)
        echo -e "Data: ${GREEN}✓ Ready (${SIZE_GROUPS} size groups)${NC}"
    elif [[ -f "$DATA_DIR/embeddings_10000.npy" ]] && [[ -f "$DATA_DIR/corpus_10000.csv" ]]; then
        echo -e "Data: ${YELLOW}⚠ Basic data ready, size groups need preprocessing${NC}"
    else
        echo -e "Data: ${RED}✗ Not prepared${NC}"
    fi
    
    # Check model
    if [[ -d "../shared_models/bge_model" ]]; then
        echo -e "Model: ${GREEN}✓ Downloaded${NC}"
    else
        echo -e "Model: ${RED}✗ Not downloaded${NC}"
    fi
    
    # Check recent results
    RECENT_RESULTS=$(find "$RESULTS_DIR" -name "pir_rag_results_*.csv" 2>/dev/null | wc -l)
    echo -e "Results: ${GREEN}$RECENT_RESULTS result files${NC}"
    
    # Check running jobs
    if command -v squeue &> /dev/null; then
        RUNNING_JOBS=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
        if [[ $RUNNING_JOBS -gt 0 ]]; then
            echo -e "SLURM Jobs: ${YELLOW}$RUNNING_JOBS running${NC}"
        else
            echo -e "SLURM Jobs: ${GREEN}None running${NC}"
        fi
    fi
}

# Main command handling
case "${1:-}" in
    setup)
        setup_environment
        ;;
    download)
        download_data
        ;;
    preprocess)
        preprocess_data
        ;;
    prepare)
        submit_data_preparation
        ;;
    experiment)
        submit_experiment
        ;;
    run-local)
        run_local
        ;;
    clean)
        clean_files
        ;;
    status)
        check_status
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
