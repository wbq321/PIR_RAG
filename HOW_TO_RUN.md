# PIR Comparison - How to Run

## Problem: Import Errors
If you get import errors like "attempted relative import beyond top-level package", use one of these solutions:

## Solution 1: Run from PIR_RAG root directory (Recommended)
```bash
cd /path/to/PIR_RAG
python -m experiments.run_pir_comparison --embeddings_path data/embeddings_10000.npy --corpus_path data/corpus_10000.csv --data_size 100 --n_queries 1 --n_clusters 10 --model_path ../shared_models/bge_model
```

## Solution 2: Use the wrapper script
```bash
cd /path/to/PIR_RAG
python run_pir_comparison_wrapper.py --embeddings_path data/embeddings_10000.npy --corpus_path data/corpus_10000.csv --data_size 100 --n_queries 1 --n_clusters 10 --model_path ../shared_models/bge_model
```

## Solution 3: Set PYTHONPATH
```bash
export PYTHONPATH=/path/to/PIR_RAG/src:$PYTHONPATH
cd /path/to/PIR_RAG
python experiments/run_pir_comparison.py --embeddings_path data/embeddings_10000.npy --corpus_path data/corpus_10000.csv --data_size 100 --n_queries 1 --n_clusters 10 --model_path ../shared_models/bge_model
```

## Recommended Command (Solution 1):
```bash
cd /scratch/user/u.bw269205/PIR_RAG
python -m experiments.run_pir_comparison \
    --embeddings_path data/embeddings_10000.npy \
    --corpus_path data/corpus_10000.csv \
    --data_size 100 \
    --n_queries 1 \
    --n_clusters 10 \
    --model_path ../shared_models/bge_model
```
