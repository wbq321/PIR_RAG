#!/usr/bin/env python3
"""
Example Usage Guide for comprehensive_experiment.py with System-Specific Arguments

This file shows examples of how to run comprehensive_experiment.py with 
all the new system-specific arguments for PIR-RAG, Graph-PIR, and Tiptoe.
"""

print("""
🚀 COMPREHENSIVE EXPERIMENT USAGE EXAMPLES

1. BASIC QUICK TEST (Small parameters for testing):
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment single \\
  --n-docs 50 \\
  --n-queries 3 \\
  --top-k 5 \\
  --output-dir "results/quick_test"

2. PIR-RAG FOCUSED TEST (Custom PIR-RAG parameters):
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment single \\
  --n-docs 500 \\
  --n-queries 10 \\
  --pir-rag-k-clusters 8 \\
  --pir-rag-cluster-top-k 5 \\
  --top-k 10 \\
  --output-dir "results/pir_rag_test"

3. GRAPH-PIR FOCUSED TEST (Custom Graph-PIR parameters):
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment single \\
  --n-docs 500 \\
  --n-queries 10 \\
  --graph-pir-k-neighbors 32 \\
  --graph-pir-ef-construction 400 \\
  --graph-pir-max-connections 32 \\
  --graph-pir-ef-search 100 \\
  --top-k 10 \\
  --output-dir "results/graph_pir_test"

4. TIPTOE FOCUSED TEST (Custom Tiptoe parameters):
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment single \\
  --n-docs 500 \\
  --n-queries 10 \\
  --tiptoe-k-clusters 10 \\
  --tiptoe-use-real-crypto \\
  --tiptoe-poly-degree 16384 \\
  --tiptoe-plain-modulus 2048 \\
  --top-k 10 \\
  --output-dir "results/tiptoe_test"

5. FULL CUSTOMIZATION (All systems with custom parameters):
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment all \\
  --n-docs 1000 \\
  --n-queries 15 \\
  --embed-dim 384 \\
  --top-k 10 \\
  --output-dir "results/full_custom" \\
  --pir-rag-k-clusters 10 \\
  --pir-rag-cluster-top-k 5 \\
  --graph-pir-k-neighbors 64 \\
  --graph-pir-ef-construction 500 \\
  --graph-pir-max-connections 64 \\
  --graph-pir-ef-search 150 \\
  --tiptoe-k-clusters 8 \\
  --tiptoe-use-real-crypto \\
  --tiptoe-poly-degree 8192 \\
  --tiptoe-plain-modulus 1024

6. SCALABILITY TEST (Custom parameters for all systems):
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment scalability \\
  --n-queries 5 \\
  --embed-dim 384 \\
  --pir-rag-k-clusters 5 \\
  --pir-rag-cluster-top-k 3 \\
  --graph-pir-k-neighbors 16 \\
  --graph-pir-ef-construction 200 \\
  --tiptoe-k-clusters 5 \\
  --tiptoe-use-real-crypto \\
  --output-dir "results/scalability_custom"

7. PARAMETER SENSITIVITY TEST:
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment sensitivity \\
  --n-docs 1000 \\
  --pir-rag-k-clusters 8 \\
  --pir-rag-cluster-top-k 4 \\
  --graph-pir-k-neighbors 32 \\
  --graph-pir-ef-construction 300 \\
  --output-dir "results/sensitivity_test"

8. RETRIEVAL PERFORMANCE TEST:
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment retrieval \\
  --n-docs 500 \\
  --n-queries 20 \\
  --top-k 10 \\
  --pir-rag-k-clusters 5 \\
  --pir-rag-cluster-top-k 3 \\
  --output-dir "results/retrieval_test"

9. WITH REAL DATA (if you have MS MARCO embeddings):
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment single \\
  --embeddings-path "data/embeddings_10000.npy" \\
  --corpus-path "data/corpus_10000.csv" \\
  --n-docs 2000 \\
  --n-queries 10 \\
  --pir-rag-k-clusters 10 \\
  --graph-pir-k-neighbors 32 \\
  --tiptoe-k-clusters 8 \\
  --output-dir "results/real_data_test"

10. DISABLE TIPTOE CRYPTO (for faster testing):
================================================================================
python experiments/comprehensive_experiment.py \\
  --experiment single \\
  --n-docs 200 \\
  --n-queries 5 \\
  --tiptoe-no-real-crypto \\
  --output-dir "results/fast_test"

📋 ARGUMENT REFERENCE:
================================================================================

GENERAL ARGUMENTS:
--experiment        : single, scalability, sensitivity, retrieval, all
--output-dir        : Directory for results (default: "results")
--n-docs           : Number of documents (default: 1000)
--n-queries        : Number of queries (default: 10)
--embed-dim        : Embedding dimension for synthetic data (default: 384)
--embeddings-path  : Path to real embeddings (.npy file)
--corpus-path      : Path to real corpus (.csv file)
--top-k           : Top-K for retrieval (default: 10)

PIR-RAG ARGUMENTS:
--pir-rag-k-clusters      : Number of clusters (default: n_docs/20)
--pir-rag-cluster-top-k   : Top clusters to retrieve (default: 3)

GRAPH-PIR ARGUMENTS:
--graph-pir-k-neighbors       : HNSW neighbors (default: 16)
--graph-pir-ef-construction   : HNSW construction parameter (default: 200)
--graph-pir-max-connections   : Max connections per node (default: 16)
--graph-pir-ef-search        : HNSW search parameter (default: 50)

TIPTOE ARGUMENTS:
--tiptoe-k-clusters          : Number of clusters (default: n_docs/20)
--tiptoe-use-real-crypto     : Enable real crypto (default: True)
--tiptoe-no-real-crypto      : Disable real crypto
--tiptoe-poly-degree         : Polynomial degree for HE (default: 8192)
--tiptoe-plain-modulus       : Plain modulus for HE (default: 1024)

💡 TIPS:
- Start with small --n-docs (50-200) for quick testing
- Use --tiptoe-no-real-crypto for faster development/debugging
- Increase --graph-pir-k-neighbors for better accuracy but slower performance
- PIR-RAG and Tiptoe k-clusters automatically scale with dataset size (n_docs/20)
- Override automatic k-clusters with explicit values if needed
- Use real data paths for more realistic experiments
""")
