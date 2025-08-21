#!/bin/bash

# Quick Communication Comparison with 1000 documents
# For faster testing and development

echo "Quick Communication Comparison (1000 documents)"
echo "==============================================="

# Configuration
DATA_DIR="data"
RESULTS_DIR="results"
EMBEDDINGS_FILE="${DATA_DIR}/embeddings_10000.npy"
CORPUS_FILE="${DATA_DIR}/corpus_10000.csv"
MODEL_PATH="../shared_models/bge_model"
OUTPUT_FILE="${RESULTS_DIR}/quick_comparison_$(date +%Y%m%d_%H%M%S).json"

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

echo "Running quick comparison with first 1000 documents..."
echo "Embeddings: ${EMBEDDINGS_FILE}"
echo "Corpus: ${CORPUS_FILE}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_FILE}"

# Run the comparison with smaller dataset
python experiments/communication_comparison.py \
    --embeddings_path "${EMBEDDINGS_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --model_path "${MODEL_PATH}" \
    --output_path "${OUTPUT_FILE}" \
    --n_queries 10 \
    --top_k 5 \
    --max_docs 1000

if [ $? -eq 0 ]; then
    echo ""
    echo "Quick comparison completed successfully! ✓"
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""
    echo "Key results:"
    if command -v jq &> /dev/null; then
        echo "Communication efficiency ratio:"
        cat ${OUTPUT_FILE} | jq '.comparison.communication_efficiency_ratio'
        echo "Relative savings:"
        cat ${OUTPUT_FILE} | jq '.comparison.relative_savings'
    else
        echo "Install 'jq' to see formatted results, or view the full JSON file:"
        echo "  cat ${OUTPUT_FILE}"
    fi
else
    echo ""
    echo "Quick comparison failed ✗"
    echo "Check the error messages above."
    exit 1
fi
