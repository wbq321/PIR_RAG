#!/bin/bash

# Communication Comparison Experiment Runner
# Runs the comparison between PIR-RAG and Graph-PIR systems

echo "Starting Communication Efficiency Comparison..."
echo "============================================="

# Configuration
DATA_DIR="data"
RESULTS_DIR="results"
EMBEDDINGS_FILE="${DATA_DIR}/embeddings_10000.npy"
CORPUS_FILE="${DATA_DIR}/corpus_10000.csv"
MODEL_PATH="../shared_models/bge_model"
OUTPUT_FILE="${RESULTS_DIR}/communication_comparison_$(date +%Y%m%d_%H%M%S).json"

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# Check if data files exist
if [ ! -f "${EMBEDDINGS_FILE}" ]; then
    echo "Error: Embeddings file not found: ${EMBEDDINGS_FILE}"
    echo "Please run data preparation first:"
    echo "  python scripts/download_data.py"
    echo "  python scripts/generate_embeddings.py"
    exit 1
fi

if [ ! -f "${CORPUS_FILE}" ]; then
    echo "Error: Corpus file not found: ${CORPUS_FILE}"
    echo "Please run data preparation first"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model not found: ${MODEL_PATH}"
    echo "Please download the model first:"
    echo "  python scripts/download_model.py"
    exit 1
fi

echo "Data files verified ✓"
echo "Embeddings: ${EMBEDDINGS_FILE}"
echo "Corpus: ${CORPUS_FILE}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_FILE}"

# Run the test first
echo ""
echo "Running basic functionality test..."
python test_graph_pir.py

if [ $? -ne 0 ]; then
    echo "Basic test failed. Please check the implementation."
    exit 1
fi

echo ""
echo "Basic test passed ✓"

# Run the comparison
echo ""
echo "Starting communication comparison experiment..."
echo "This may take several minutes..."

python experiments/communication_comparison.py \
    --embeddings_path "${EMBEDDINGS_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --model_path "${MODEL_PATH}" \
    --output_path "${OUTPUT_FILE}" \
    --n_queries 30 \
    --top_k 10 \
    --max_docs 1000

if [ $? -eq 0 ]; then
    echo ""
    echo "Experiment completed successfully! ✓"
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""
    echo "To view detailed results:"
    echo "  cat ${OUTPUT_FILE} | jq '.comparison'"
    echo ""
    echo "Key metrics are also printed above."
else
    echo ""
    echo "Experiment failed ✗"
    echo "Check the error messages above for debugging information."
    exit 1
fi
