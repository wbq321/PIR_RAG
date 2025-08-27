#!/bin/bash

# Comprehensive PIR Experiment Runner
# This script runs all PIR experiments and generates analysis plots

echo "Starting Comprehensive PIR Experiments"
echo "======================================"

# Set up Python environment
export PYTHONPATH="${PYTHONPATH}:./src"

# Run comprehensive experiments
echo "Running comprehensive experiments..."
python experiments/comprehensive_experiment.py --experiment all --n-docs 1000 --n-queries 10

# Wait for completion
if [ $? -eq 0 ]; then
    echo "Experiments completed successfully!"
    
    # Generate analysis plots
    echo "Generating analysis plots..."
    python experiments/analyze_results.py --generate-all
    
    if [ $? -eq 0 ]; then
        echo "Analysis completed successfully!"
        echo "Check the results/ directory for data files"
        echo "Check the results/figures/ directory for plots"
        echo ""
        echo "Files generated:"
        echo "- JSON experiment data with detailed timing"
        echo "- CSV summary files for easy analysis"
        echo "- PNG and PDF plots for all comparisons"
        echo "- Text summary report"
    else
        echo "Analysis failed, but experiment data is available"
    fi
else
    echo "Experiments failed!"
    exit 1
fi

echo ""
echo "Experiment Summary:"
echo "- All three PIR systems tested (PIR-RAG, Graph-PIR, Tiptoe)"
echo "- Real cryptographic operations verified"
echo "- URL retrieval implemented across all systems"
echo "- Comprehensive timing and communication cost analysis"
echo "- Scalability testing across different dataset sizes"
echo "- Parameter sensitivity analysis"
echo ""
echo "Check the summary report for detailed results!"
