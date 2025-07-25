# PIR-RAG: Private Information Retrieval for RAG Systems

A privacy-preserving Retrieval-Augmented Generation (RAG) system that uses homomorphic encryption to enable private document retrieval while maintaining high-quality semantic search.

## Project Structure

```
PIR_RAG/
├── src/pir_rag/                 # Core PIR-RAG library
│   ├── __init__.py             # Package initialization
│   ├── client.py               # Client-side implementation
│   ├── server.py               # Server-side implementation
│   └── utils.py                # Utility functions
├── scripts/                     # Data preparation scripts
│   ├── download_data.py        # Download MS MARCO dataset
│   ├── download_model.py       # Download sentence transformer model
│   ├── generate_embeddings.py  # Generate document embeddings
│   └── text_len_dis.py         # Analyze document length distribution
├── experiments/                 # Experiment scripts
│   └── run_experiments.py      # Main experiment runner
├── slurm/                      # SLURM batch scripts
│   ├── prepare_data.sbatch     # Data preparation job
│   └── run_experiments.sbatch  # Experiment execution job
├── data/                       # Data directory (created after data prep)
├── results/                    # Experiment results and logs
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n pir_rag_env python=3.8
conda activate pir_rag_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

On a node with internet access:

```bash
# Download data and model (run on login node)
python scripts/download_data.py
python scripts/download_model.py

# Generate embeddings (submit to compute node)
sbatch slurm/prepare_data.sbatch
```

### 3. Run Experiments

```bash
# Submit experiment job
sbatch slurm/run_experiments.sbatch

# Or run locally
python experiments/run_experiments.py \
    --embeddings_path data/embeddings_10000.npy \
    --corpus_path data/corpus_10000.csv \
    --model_path ../shared_models/bge_model
```

## System Architecture

### PIR-RAG Workflow

1. **Server Setup**: 
   - Clusters documents using K-means on embeddings
   - Encodes documents into fixed-size chunks for PIR
   - Sets up PIR database structure

2. **Client Query**:
   - Finds relevant clusters using semantic similarity
   - Generates encrypted PIR queries (Paillier encryption)
   - Receives encrypted document chunks
   - Decrypts and re-ranks final results

3. **Privacy Guarantees**:
   - Server never learns which clusters/documents client wants
   - Client only learns documents in requested clusters
   - Uses homomorphic encryption for private computation

### Key Components

- **PIRRAGServer**: Handles document clustering and encrypted query processing
- **PIRRAGClient**: Manages key generation, query encryption, and result decryption
- **Utilities**: Document encoding/decoding, evaluation data preparation

## Configuration

Modify `config.yaml` to customize experiments:

```yaml
experiments:
  document_size:
    enabled: true
    doc_sizes: [1500, 2500, 3500, 4500]
  
  cluster_count:
    enabled: false
    cluster_counts: [100, 500, 1000, 5000]
```

## Experiments

The system evaluates several factors:

- **Document Size Impact**: How document size affects performance
- **Cluster Count**: Trade-offs between privacy and efficiency
- **Top-K Retrieval**: Number of clusters to query
- **Key Length**: Encryption strength vs. performance

### Metrics

- **Privacy Costs**: Upload/download bandwidth, encryption times
- **Performance**: Query latency, server computation time  
- **Effectiveness**: Recall@10 for retrieval quality

## Results

Results are saved in `results/` directory with timestamps:
- `pir_rag_results_YYYYMMDD-HHMMSS.csv`: Experiment metrics
- `pir_rag_exp_<job_id>.out/.err`: SLURM job logs

## Dependencies

- Python 3.8+
- PyTorch
- sentence-transformers
- scikit-learn
- phe (Paillier Homomorphic Encryption)
- pandas, numpy, tqdm

## Usage Examples

### Custom Experiment

```python
from src.pir_rag import PIRRAGClient, PIRRAGServer
from src.pir_rag.utils import prepare_evaluation_data

# Load your data
embeddings = np.load("data/embeddings.npy")
corpus_texts = pd.read_csv("data/corpus.csv")['text'].tolist()

# Set up system
server = PIRRAGServer()
server.setup(embeddings, corpus_texts, n_clusters=1000)

client = PIRRAGClient()
client.setup(server.centroids, key_length=2048)

# Query
query_embedding = model.encode(["your query here"])
cluster_indices = client.find_relevant_clusters(query_embedding, top_k=3)
docs, metrics = client.pir_retrieve(server, cluster_indices)
```

## Research Context

This project investigates privacy-preserving RAG systems, addressing:
- How to maintain user privacy in document retrieval
- Performance trade-offs of homomorphic encryption
- Scalability of PIR for large document collections
- Practical deployment considerations

## Contributing

1. Follow the modular structure in `src/pir_rag/`
2. Add new experiments to `experiments/`
3. Update configuration in `config.yaml`
4. Add tests for new functionality

## License

[Add your license here]
