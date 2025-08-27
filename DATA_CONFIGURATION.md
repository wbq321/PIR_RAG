# Data Configuration Guide

## 🎯 **Quick Start - No Setup Required**

The experiment framework works **immediately** without any data setup:

```bash
# Run experiments with synthetic data (default)
python experiments/comprehensive_experiment.py --experiment all --n-docs 1000

# Test retrieval performance
python test_retrieval_performance.py

# Analyze results
python experiments/analyze_results.py --generate-all
```

## 📊 **Data Options**

### Option 1: Synthetic Data (Default - Zero Setup)

**✅ Recommended for:** Performance benchmarking, system comparison, development

The framework automatically generates realistic synthetic data:
- Random embeddings with configurable dimensions
- Synthetic documents with realistic text patterns
- Consistent across all PIR systems for fair comparison
- Perfect for timing and scalability analysis

**Usage:**
```bash
# Uses synthetic data automatically
python experiments/comprehensive_experiment.py --experiment all
```

### Option 2: Default MS MARCO Data (Auto-Detection)

**✅ Recommended for:** Research validation with real data

If you have prepared MS MARCO data in the expected location:

```
data/
├── embeddings_10000.npy    # Pre-computed embeddings
└── corpus_10000.csv        # Documents with 'text' column
```

The framework will **automatically detect and use** this data:

```bash
# Will auto-detect and use MS MARCO data if available
python experiments/comprehensive_experiment.py --experiment all --n-docs 5000
```

### Option 3: Custom Data Paths

**✅ Recommended for:** Custom datasets, different embedding models

Specify your own data files:

```bash
python experiments/comprehensive_experiment.py \
    --experiment all \
    --embeddings-path "path/to/your_embeddings.npy" \
    --corpus-path "path/to/your_corpus.csv" \
    --n-docs 2000
```

**Requirements:**
- Embeddings: `.npy` file with shape `[n_documents, embedding_dim]`
- Corpus: `.csv` file with a `text` column containing document text

## 🛠️ **Data Preparation (Optional)**

### For MS MARCO Dataset:

1. **Download MS MARCO:**
   ```bash
   python scripts/download_data.py
   ```

2. **Generate Embeddings:**
   ```bash
   python scripts/generate_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
   ```

3. **Files Created:**
   ```
   data/
   ├── embeddings_10000.npy
   ├── corpus_10000.csv
   └── query_embeddings.npy
   ```

### For Custom Dataset:

1. **Prepare your data:**
   ```python
   import numpy as np
   import pandas as pd
   
   # Your embeddings as numpy array
   embeddings = np.array(your_embeddings)  # Shape: [n_docs, embed_dim]
   np.save("my_embeddings.npy", embeddings)
   
   # Your documents as CSV
   df = pd.DataFrame({"text": your_documents})
   df.to_csv("my_corpus.csv", index=False)
   ```

2. **Run experiments:**
   ```bash
   python experiments/comprehensive_experiment.py \
       --embeddings-path "my_embeddings.npy" \
       --corpus-path "my_corpus.csv"
   ```

## 🎛️ **Configuration Examples**

### Quick Performance Test
```bash
# 500 docs, synthetic data, all experiments
python experiments/comprehensive_experiment.py --experiment all --n-docs 500
```

### Scalability Test with Real Data
```bash
# Use MS MARCO data if available, otherwise synthetic
python experiments/comprehensive_experiment.py --experiment scalability
```

### Custom Data Analysis
```bash
# Your own embeddings and documents
python experiments/comprehensive_experiment.py \
    --experiment all \
    --embeddings-path "research_data/paper_embeddings.npy" \
    --corpus-path "research_data/paper_corpus.csv" \
    --n-docs 2000 \
    --n-queries 20
```

### Retrieval Quality Evaluation
```bash
# Test retrieval performance with specific data
python test_retrieval_performance.py  # Uses synthetic data by default
```

## 🔍 **Data Fallback Logic**

The framework uses this priority order:

1. **Custom paths specified** → Use your files
2. **Default MS MARCO exists** → Use `data/embeddings_10000.npy` and `data/corpus_10000.csv`
3. **Fallback** → Generate synthetic data automatically

**All options produce clean, measurable results!**

## 📈 **Performance Recommendations**

| Use Case | Data Type | Command |
|----------|-----------|---------|
| **Development & Debugging** | Synthetic | `python experiments/comprehensive_experiment.py --n-docs 100` |
| **Performance Benchmarking** | Synthetic | `python experiments/comprehensive_experiment.py --experiment all` |
| **Research Validation** | Real (MS MARCO) | Place data in `data/` folder, run normally |
| **Publication Results** | Real (Custom) | Use `--embeddings-path` and `--corpus-path` |

## ✅ **Verification**

Check what data is being used by looking at the experiment output:

```
Loading real data from data/embeddings_10000.npy and data/corpus_10000.csv...
Loaded 1000 documents, embedding dim: 384
```

or

```
Generating synthetic test data: 1000 documents, 384D embeddings
```

**Ready to run experiments? No data setup required - just start experimenting!**
