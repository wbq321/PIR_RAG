# How to Analyze Retrieval Performance Results

This guide shows you how to run retrieval experiments and analyze the results.

## Step 1: Run Retrieval Performance Experiments

First, run the retrieval performance experiments to generate data:

```bash
cd "d:\UW\Research\rag encryption\PIR_RAG"

# Run retrieval experiments with default settings
python experiments/comprehensive_experiment.py --experiment retrieval

# Or with custom parameters
python experiments/comprehensive_experiment.py --experiment retrieval \
    --n-docs 1000 \
    --n-queries 20 \
    --top-k 10 \
    --pir-rag-k-clusters 10
```

This will generate JSON files with names like:
- `retrieval_performance_YYYYMMDD-HHMMSS.json`

## Step 2: Analyze the Results

### Option A: Use the Dedicated Retrieval Analyzer (Recommended)

```bash
# Analyze all retrieval results (finds most recent automatically)
python analyze_retrieval_results.py

# Analyze a specific result file
python analyze_retrieval_results.py --file results/retrieval_performance_20250827-123456.json

# Specify custom results directory
python analyze_retrieval_results.py --results-dir ./my_results
```

### Option B: Use the General Results Analyzer

```bash
# Generate all plots including retrieval analysis
python experiments/analyze_results.py --generate-all

# Analyze specific retrieval file
python experiments/analyze_results.py --file retrieval_performance_20250827-123456.json
```

## What You'll Get

### 1. Console Output Analysis
```
====================================================================== 
RETRIEVAL PERFORMANCE ANALYSIS SUMMARY
======================================================================
📊 Experiment Overview:
  Total experiments: 3
  Successful: 3
  Failed: 0

🏆 System Performance Ranking:
System       Precision@K  Recall@K   NDCG@K     Query Time  
------------------------------------------------------------
PIR-RAG      0.650        0.650      0.742      2.341s      
Graph-PIR    0.890        0.890      0.923      4.523s      
Tiptoe       0.720        0.720      0.801      3.102s      

🥇 Best Performance:
  Highest Precision@K: Graph-PIR
  Highest Recall@K: Graph-PIR  
  Highest NDCG@K: Graph-PIR

⚡ Efficiency Analysis:
  PIR-RAG: 0.275 (precision per second per KB)
  Graph-PIR: 0.195 (precision per second per KB)
  Tiptoe: 0.230 (precision per second per KB)
```

### 2. Visualization Files
Generated in `results/figures/`:

- **`retrieval_performance_analysis_TIMESTAMP.png`**: 
  - 4-panel plot showing retrieval quality, query timing, communication overhead, setup time
  
- **`retrieval_metrics_table_TIMESTAMP.csv`**:
  - Detailed CSV table with all metrics for further analysis

### 3. Key Metrics Explained

#### Retrieval Quality Metrics:
- **Precision@K**: Fraction of retrieved docs that are relevant (higher = better)
- **Recall@K**: Fraction of relevant docs that were retrieved (higher = better)  
- **NDCG@K**: Normalized Discounted Cumulative Gain (considers ranking order, higher = better)

#### Performance Metrics:
- **Setup Time**: One-time cost to initialize the system
- **Query Time**: Average time per query (for PIR operations)
- **Simulation Time**: Time for plaintext quality calculation (hybrid approach)
- **Communication**: Data transfer overhead per query

## Understanding the Results

### Good Results Indicate:
- **High Precision/Recall/NDCG** (> 0.7): System finds relevant documents effectively
- **Low Query Time** (< 5s): Fast enough for interactive use
- **Reasonable Communication** (< 100KB): Network-friendly for remote PIR

### Watch Out For:
- **Perfect scores (1.0)**: May indicate data leakage or overfitting
- **Very low scores (< 0.3)**: System may not be working properly
- **Huge performance differences**: Check if parameters are comparable

### Troubleshooting

If you get errors:
1. **No result files found**: Run retrieval experiments first
2. **Import errors**: Dependencies are expected (analysis still works)
3. **JSON parsing errors**: Check if files are complete (experiments may have crashed)

### Advanced Analysis

For deeper analysis, you can load the JSON directly:

```python
import json
import pandas as pd

# Load results
with open('results/retrieval_performance_20250827-123456.json', 'r') as f:
    data = json.load(f)

# Extract per-query results for detailed analysis
pir_rag_queries = data['pir_rag']['query_results']
graph_pir_queries = data['graph_pir']['query_results'] 

# Analyze variance, outliers, etc.
```

## Result Interpretation Guide

### System Comparison:
- **PIR-RAG**: Usually fastest setup, moderate quality, good for large datasets
- **Graph-PIR**: Often highest quality, slower setup, good for accuracy-focused use cases
- **Tiptoe**: Balanced performance, good for general-purpose RAG with privacy

### Parameter Tuning:
- Increase `n_queries` for more reliable statistics (20-50 recommended)
- Increase `top_k` to test broader retrieval scenarios
- Adjust cluster parameters based on your dataset characteristics
