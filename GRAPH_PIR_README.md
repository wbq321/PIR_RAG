# Graph-PIR Implementation for Communication Efficiency Comparison

This document describes the new Graph-PIR implementation added to compare communication efficiency with PIR-RAG.

## What Was Added

### 1. Graph-PIR Core Implementation (`src/graph_pir/`)

- **`piano_pir.py`**: Simplified PianoPIR implementation based on the Go version from private-search-temp
  - `PianoPIRClient`: Handles query generation and response decoding
  - `PianoPIRServer`: Processes PIR queries and returns encrypted responses  
  - `SimpleBatchPianoPIR`: Manages batch processing across multiple partitions

- **`graph_search.py`**: Graph-based ANN search with PIR integration
  - `GraphSearchEngine`: Implements graph traversal with private vertex access
  - Builds graph structure with neighbor relationships
  - Uses PIR for private vertex retrieval during search

- **`system.py`**: Unified interface compatible with PIR-RAG comparison
  - `GraphPIRSystem`: Main system class with setup/search methods
  - Compatible interface for fair comparison with PIR-RAG
  - Comprehensive performance tracking and statistics

### 2. Comparison Framework

- **`experiments/communication_comparison.py`**: Complete comparison experiment
  - Runs both PIR-RAG and Graph-PIR on the same queries
  - Measures communication costs, query times, and other metrics
  - Generates detailed comparison reports and statistics

- **`test_graph_pir.py`**: Validation and testing script
  - Tests basic functionality with synthetic data
  - Validates both private and non-private modes
  - Checks integration with comparison framework

- **`run_comparison.sh`**: Automated execution script
  - Runs complete comparison experiment
  - Checks for required data files and dependencies
  - Generates timestamped results

## How It Works

### Graph-PIR Approach
1. **Setup Phase**: 
   - Build graph structure connecting similar documents
   - Package vector embeddings + neighbor info into PIR database
   - Initialize batch PIR system with partitioning

2. **Query Phase**:
   - Start from random vertices in the graph
   - Use PIR to privately access vertex information (vector + neighbors)
   - Traverse graph by following edges to most similar vertices
   - Collect top-k results through iterative exploration

### Communication Efficiency Focus
- **PIR-RAG**: Encrypts queries for entire clusters, downloads full cluster contents
- **Graph-PIR**: Only accesses specific vertices during traversal, potentially more selective

## Usage

### Quick Test
```bash
cd PIR_RAG
python test_graph_pir.py
```

### Full Comparison
```bash
cd PIR_RAG
./run_comparison.sh
```

### Manual Comparison
```bash
python experiments/communication_comparison.py \
    --embeddings_path data/embeddings_10000.npy \
    --corpus_path data/corpus_10000.csv \
    --model_path ../shared_models/bge_model \
    --n_queries 30 \
    --top_k 10
```

## Key Metrics Compared

1. **Communication Efficiency**:
   - Total bytes uploaded/downloaded
   - Average communication per query
   - Upload vs download breakdown

2. **Performance**:
   - Query latency
   - Setup time
   - Preprocessing overhead

3. **Search Quality**:
   - Number of results found
   - Search exploration patterns

## Expected Results

The comparison will show trade-offs between the two approaches:
- **PIR-RAG**: Higher communication per query but consistent costs
- **Graph-PIR**: Variable communication based on graph traversal depth

This implementation enables researchers to evaluate which approach is more suitable for different privacy-preserving search scenarios.

## Files Structure

```
PIR_RAG/
├── src/graph_pir/              # Graph-PIR implementation
│   ├── __init__.py
│   ├── piano_pir.py           # PianoPIR core
│   ├── graph_search.py        # Graph search engine
│   └── system.py              # Unified interface
├── experiments/
│   └── communication_comparison.py  # Comparison experiment
├── test_graph_pir.py          # Testing script
└── run_comparison.sh          # Execution script
```
