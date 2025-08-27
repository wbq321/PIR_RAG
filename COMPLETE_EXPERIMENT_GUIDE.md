# Complete PIR Experiment Framework

## 🎯 **Overview**

This comprehensive framework provides **complete experimental evaluation** for PIR-based RAG systems, including:

- ⏱️ **Performance Benchmarking** - Timing, scalability, communication costs
- 🎯 **Retrieval Quality** - Precision@K, Recall@K, NDCG@K evaluation  
- 📊 **Comprehensive Analysis** - Automated plotting and reporting
- 🔄 **Full System Comparison** - PIR-RAG, Graph-PIR, Tiptoe side-by-side

## 🚀 **Quick Start**

### **Zero Setup - Instant Results**
```bash
# Run complete experiment suite with synthetic data (no setup required)
python run_complete_experiments.py --quick

# Analyze existing results
python run_complete_experiments.py --analysis-only
```

### **Full Scale Evaluation**
```bash
# Comprehensive experiments with all systems and metrics
python run_complete_experiments.py --full --n-docs 2000 --n-queries 50
```

## 📋 **Experiment Types**

### 1. **Performance Benchmarking**

**What it measures:**
- Setup time (clustering, indexing, preprocessing)
- Query time (per-step breakdown)
- Communication cost (upload/download bytes)
- Scalability (performance vs dataset size)
- Parameter sensitivity (optimal configurations)

**Command:**
```bash
python experiments/comprehensive_experiment.py --experiment all --n-docs 1000
```

**Outputs:**
- `results/single_experiment_*.json` - Detailed timing data
- `results/scalability_*.json` - Performance across dataset sizes
- `results/parameter_sensitivity_*.json` - Parameter optimization data

### 2. **Retrieval Quality Evaluation**

**What it measures:**
- Precision@K - Fraction of retrieved docs that are relevant
- Recall@K - Fraction of relevant docs that are retrieved
- NDCG@K - Normalized discounted cumulative gain
- Query throughput (QPS)
- Retrieval consistency

**Command:**
```bash
python test_retrieval_performance.py
# OR integrated:
python experiments/comprehensive_experiment.py --experiment retrieval
```

**Outputs:**
- `results/retrieval_performance_*.json` - IR quality metrics
- Per-query detailed analysis
- System comparison tables

### 3. **Comprehensive Analysis & Plotting**

**Performance Analysis:**
```bash
python experiments/analyze_results.py --generate-all
```

**Retrieval Analysis:**
```bash
python experiments/analyze_retrieval_performance.py --generate-all
```

**Generated Plots:**
- 📊 System comparison charts
- 📈 Scalability curves  
- ⚙️ Parameter sensitivity heatmaps
- 🎯 Retrieval quality comparisons
- 📊 Query performance distributions

## 📊 **Data Configuration**

### **Option 1: Synthetic Data (Default)**
✅ **No setup required** - Framework generates realistic synthetic data automatically.

```bash
# Works immediately
python run_complete_experiments.py
```

### **Option 2: Real Data (MS MARCO)**
Place data files in expected location:
```
data/
├── embeddings_10000.npy    # Pre-computed embeddings
└── corpus_10000.csv        # Documents with 'text' column
```

Framework **automatically detects** and uses real data when available.

### **Option 3: Custom Data**
```bash
python experiments/comprehensive_experiment.py \
    --embeddings-path "your_embeddings.npy" \
    --corpus-path "your_corpus.csv"
```

**See `DATA_CONFIGURATION.md` for detailed setup instructions.**

## 🎛️ **Command Reference**

### **Core Experiments**

```bash
# Quick test (small scale)
python run_complete_experiments.py --quick

# Single system comparison
python experiments/comprehensive_experiment.py --experiment single

# Scalability analysis
python experiments/comprehensive_experiment.py --experiment scalability

# Parameter optimization
python experiments/comprehensive_experiment.py --experiment sensitivity

# Retrieval quality evaluation
python experiments/comprehensive_experiment.py --experiment retrieval

# Everything at once
python experiments/comprehensive_experiment.py --experiment all
```

### **Analysis & Plotting**

```bash
# Complete analysis suite
python experiments/analyze_results.py --generate-all

# Retrieval quality analysis
python experiments/analyze_retrieval_performance.py --generate-all

# Specific plot types
python experiments/analyze_results.py --generate-timing
python experiments/analyze_results.py --generate-scalability
```

### **Data Testing**

```bash
# Test data loading options
python test_data_loading.py

# Test retrieval performance only
python test_retrieval_performance.py
```

## 📈 **Output Structure**

```
results/
├── single_experiment_*.json           # Timing data
├── scalability_*.json                 # Performance vs size
├── retrieval_performance_*.json       # IR quality metrics
├── *_summary_*.csv                    # CSV summaries
├── *_summary_*.txt                    # Human-readable reports
└── figures/                           # All plots
    ├── system_comparison_*.png
    ├── scalability_*.png
    ├── retrieval_quality_*.png
    └── parameter_sensitivity_*.png
```

## 🔬 **Research Use Cases**

### **Performance Research**
```bash
# Large-scale performance evaluation
python run_complete_experiments.py --full --n-docs 5000 --n-queries 100

# Parameter optimization study
python experiments/comprehensive_experiment.py --experiment sensitivity --n-docs 2000
```

### **Retrieval Quality Research**
```bash
# Comprehensive IR evaluation with real data
python experiments/comprehensive_experiment.py --experiment retrieval \
    --embeddings-path "data/embeddings_10000.npy" \
    --corpus-path "data/corpus_10000.csv" \
    --n-queries 100 --top-k 20
```

### **System Comparison Study**
```bash
# Head-to-head comparison
python experiments/comprehensive_experiment.py --experiment all --n-docs 3000
python experiments/analyze_results.py --generate-all
python experiments/analyze_retrieval_performance.py --generate-all
```

## ⚡ **Performance Tips**

### **Quick Development**
- Use `--quick` flag for rapid iteration
- Start with synthetic data (no setup required)
- Use `--n-docs 100 --n-queries 5` for debugging

### **Publication Quality**
- Use `--full` flag for comprehensive evaluation
- Include real data for domain-specific validation
- Run multiple seeds for statistical significance

### **Large Scale**
- Increase `--n-docs` and `--n-queries` gradually
- Monitor memory usage with large embeddings
- Use `--analysis-only` to re-generate plots without re-running experiments

## 🎯 **Example Workflows**

### **1. Quick System Verification**
```bash
python test_data_loading.py                    # Verify setup
python run_complete_experiments.py --quick     # Quick test
```

### **2. Development & Debugging**
```bash
python experiments/comprehensive_experiment.py --experiment single --n-docs 100
python experiments/analyze_results.py --generate-summary
```

### **3. Research Evaluation**
```bash
python run_complete_experiments.py --full --n-docs 3000 --n-queries 100
# Review results/figures/ for all plots and reports
```

### **4. Publication Preparation**
```bash
# Run with real data
python experiments/comprehensive_experiment.py --experiment all \
    --embeddings-path "data/embeddings_10000.npy" \
    --corpus-path "data/corpus_10000.csv" \
    --n-docs 5000 --n-queries 200

# Generate publication-quality plots
python experiments/analyze_results.py --generate-all
python experiments/analyze_retrieval_performance.py --generate-all
```

## 🏆 **Key Features**

✅ **Zero Configuration** - Works immediately with synthetic data  
✅ **Flexible Data** - Synthetic, MS MARCO, or custom datasets  
✅ **Complete Metrics** - Performance + retrieval quality  
✅ **Automated Analysis** - Plots and reports generated automatically  
✅ **Three PIR Systems** - PIR-RAG, Graph-PIR, Tiptoe comparison  
✅ **Scalable** - From 100 to 10,000+ documents  
✅ **Research Ready** - Publication-quality outputs  

## 🆘 **Troubleshooting**

**No results found:**
```bash
python test_data_loading.py  # Check data loading
ls results/                  # Check if experiments ran
```

**Import errors:**
```bash
pip install -r requirements.txt
python test_imports.py
```

**Performance issues:**
```bash
# Start small and scale up
python run_complete_experiments.py --quick
```

**Need help:**
```bash
python experiments/comprehensive_experiment.py --help
python experiments/analyze_results.py --help
```

---

**🎉 Ready to evaluate your PIR systems comprehensively!**

The framework provides everything needed for thorough experimental evaluation, from quick development tests to publication-ready research results.
