"""
Communication Efficiency Comparison: PIR-RAG vs Graph-PIR vs Tiptoe

Compares the three systems on the same dataset to evaluate:
- Upload/download bytes
- Query latency
- Preprocessing costs
- Retrieval quality
"""

import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List
import sys
import os
import argparse

# Add PIR_RAG to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from pir_rag import PIRRAGClient, PIRRAGServer
from graph_pir import GraphPIRSystem
from tiptoe import TiptoeSystem
from sentence_transformers import SentenceTransformer


def load_test_data(embeddings_path: str = None, corpus_path: str = None, data_size: int = 1000):
    """Load test data from specified paths or generate synthetic data."""
    try:
        # Try to load real data if paths provided
        if embeddings_path and corpus_path and os.path.exists(embeddings_path) and os.path.exists(corpus_path):
            print(f"Loading real data from {embeddings_path} and {corpus_path} (first {data_size} samples)...")
            embeddings = np.load(embeddings_path)[:data_size]
            corpus_df = pd.read_csv(corpus_path)
            documents = corpus_df['text'].iloc[:data_size].tolist()
            print(f"Loaded {len(documents)} documents, embedding dim: {embeddings.shape[1]}")

        # Try default paths if no paths specified
        elif embeddings_path is None and corpus_path is None:
            default_embeddings = "data/embeddings_10000.npy"
            default_corpus = "data/corpus_10000.csv"

            if os.path.exists(default_embeddings) and os.path.exists(default_corpus):
                print(f"Loading default MS MARCO data (first {data_size} samples)...")
                embeddings = np.load(default_embeddings)[:data_size]
                corpus_df = pd.read_csv(default_corpus)
                documents = corpus_df['text'].iloc[:data_size].tolist()
                print(f"Loaded {len(documents)} documents, embedding dim: {embeddings.shape[1]}")
            else:
                raise FileNotFoundError("Default data files not found")
        else:
            raise FileNotFoundError("Specified data files not found")

    except Exception as e:
        print(f"Could not load real data ({e}), generating synthetic data for {data_size} documents...")
        # Generate synthetic data
        embedding_dim = 384
        np.random.seed(42)
        embeddings = np.random.randn(data_size, embedding_dim).astype(np.float32)
        documents = [
            f"This is document {i} about topic {i%10}. " +
            f"It contains information relevant to query types and has some content for testing. " * 3
            for i in range(data_size)
        ]
        print(f"Generated {len(documents)} documents, embedding dim: {embeddings.shape[1]}")

    return embeddings, documents


def generate_test_queries(documents: List[str], n_queries: int = 10) -> List[str]:
    """Generate realistic test queries by sampling from corpus documents."""
    if len(documents) == 0:
        raise ValueError("Cannot generate queries from empty document list")

    # Sample random documents to use as queries
    np.random.seed(42)  # For reproducible results
    query_indices = np.random.choice(len(documents), size=min(n_queries, len(documents)), replace=False)

    selected_queries = []
    for idx in query_indices:
        doc_text = documents[idx]
        # Take first sentence or first 100 characters as query
        if '.' in doc_text:
            # Use first sentence
            query = doc_text.split('.')[0].strip() + '.'
        else:
            # Use first 100 characters
            query = doc_text[:100].strip()
            if len(doc_text) > 100:
                query += "..."

        # Ensure query is not too short
        if len(query.strip()) < 10:
            query = doc_text[:50].strip() + "..."

        selected_queries.append(query)

    print(f"Generated {len(selected_queries)} queries from corpus documents")
    return selected_queries


def load_model(model_path: str) -> SentenceTransformer:
    """Load sentence transformer model for query encoding."""
    try:
        print(f"Loading sentence transformer model from {model_path}...")
        model = SentenceTransformer(model_path)
        print(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Trying default model...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Loaded default model: all-MiniLM-L6-v2")
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load any model: {e2}")


def run_pir_rag_experiment(embeddings: np.ndarray, documents: List[str],
                          n_clusters: int, n_queries: int = 10, key_length: int = 2048,
                          model: SentenceTransformer = None) -> Dict[str, Any]:
    """Run PIR-RAG experiment."""
    print(f"\n=== PIR-RAG Experiment (clusters={n_clusters}) ===")

    # Setup
    setup_start = time.perf_counter()
    server = PIRRAGServer()
    server_metrics = server.setup(embeddings, documents, n_clusters)

    client = PIRRAGClient()
    client_metrics = client.setup(server.centroids, key_length=key_length)
    setup_time = time.perf_counter() - setup_start

    print(f"Setup complete in {setup_time:.2f}s")

    # Run queries
    all_metrics = []
    total_upload = 0
    total_download = 0
    total_query_time = 0

    # Generate test queries
    test_queries = generate_test_queries(documents, n_queries)

    for i, query_text in enumerate(test_queries):
        # Encode query using the model
        if model is not None:
            query_embedding = torch.tensor(model.encode([query_text])[0])
        else:
            # Fallback to random if no model (for testing only)
            query_embedding = torch.tensor(np.random.randn(embeddings.shape[1]).astype(np.float32))
            print(f"Warning: Using random query embedding for query {i+1} (no model provided)")

        # Find relevant clusters
        cluster_indices = client.find_relevant_clusters(query_embedding, top_k=3)

        # Perform PIR retrieval
        query_start = time.perf_counter()
        retrieved_docs, query_metrics = client.pir_retrieve(server, cluster_indices)
        query_time = time.perf_counter() - query_start

        total_upload += query_metrics["total_upload_bytes"]
        total_download += query_metrics["total_download_bytes"]
        total_query_time += query_time

        if i == 0:
            print(f"First query: '{query_text[:50]}...'")
            print(f"Retrieved {len(retrieved_docs)} docs in {query_time:.3f}s")

    avg_metrics = {
        "system": "PIR-RAG",
        "setup_time": setup_time,
        "avg_query_time": total_query_time / n_queries,
        "avg_upload_bytes": total_upload / n_queries,
        "avg_download_bytes": total_download / n_queries,
        "total_upload_bytes": total_upload,
        "total_download_bytes": total_download,
        "n_clusters": n_clusters,
        "n_queries": n_queries,
        "server_setup_metrics": server_metrics,
        "client_setup_metrics": client_metrics
    }

    print(f"PIR-RAG Results:")
    print(f"  Avg query time: {avg_metrics['avg_query_time']:.3f}s")
    print(f"  Avg upload: {avg_metrics['avg_upload_bytes']:,} bytes")
    print(f"  Avg download: {avg_metrics['avg_download_bytes']:,} bytes")

    return avg_metrics


def run_graph_pir_experiment(embeddings: np.ndarray, documents: List[str],
                           n_queries: int = 10, top_k: int = 10,
                           model: SentenceTransformer = None) -> Dict[str, Any]:
    """Run Graph-PIR experiment."""
    print(f"\n=== Graph-PIR Experiment ===")

    # Setup
    setup_start = time.perf_counter()
    graph_pir = GraphPIRSystem()
    setup_metrics = graph_pir.setup(embeddings, documents)
    setup_time = time.perf_counter() - setup_start

    print(f"Setup complete in {setup_time:.2f}s")

    # Run queries
    total_upload = 0
    total_download = 0
    total_query_time = 0

    # Generate test queries
    test_queries = generate_test_queries(documents, n_queries)

    for i, query_text in enumerate(test_queries):
        # Encode query using the model
        if model is not None:
            query_embedding = model.encode([query_text])[0].astype(np.float32)
        else:
            # Fallback to random if no model (for testing only)
            query_embedding = np.random.randn(embeddings.shape[1]).astype(np.float32)
            print(f"Warning: Using random query embedding for query {i+1} (no model provided)")

        # Perform Graph-PIR query
        query_start = time.perf_counter()
        retrieved_docs, query_metrics = graph_pir.query(query_embedding, top_k=top_k)
        query_time = time.perf_counter() - query_start

        total_upload += query_metrics["phase1_upload_bytes"] + query_metrics["phase2_upload_bytes"]
        total_download += query_metrics["phase1_download_bytes"] + query_metrics["phase2_download_bytes"]
        total_query_time += query_time

        if i == 0:
            print(f"First query: '{query_text[:50]}...'")
            print(f"Retrieved {len(retrieved_docs)} docs in {query_time:.3f}s")
            print(f"  Phase 1: {query_metrics['phase1_time']:.3f}s, Phase 2: {query_metrics['phase2_time']:.3f}s")

    avg_metrics = {
        "system": "Graph-PIR",
        "setup_time": setup_time,
        "avg_query_time": total_query_time / n_queries,
        "avg_upload_bytes": total_upload / n_queries,
        "avg_download_bytes": total_download / n_queries,
        "total_upload_bytes": total_upload,
        "total_download_bytes": total_download,
        "n_queries": n_queries,
        "setup_metrics": setup_metrics
    }

    print(f"Graph-PIR Results:")
    print(f"  Avg query time: {avg_metrics['avg_query_time']:.3f}s")
    print(f"  Avg upload: {avg_metrics['avg_upload_bytes']:,} bytes")
    print(f"  Avg download: {avg_metrics['avg_download_bytes']:,} bytes")

    return avg_metrics


def run_tiptoe_experiment(embeddings: np.ndarray, documents: List[str],
                         test_queries: List[str], n_clusters: int = 100,
                         top_k: int = 10, model: SentenceTransformer = None) -> Dict[str, Any]:
    """Run Tiptoe private search experiment."""
    print(f"\n=== Tiptoe Experiment ===")
    print(f"[Tiptoe] Setting up Tiptoe system...")
    print(f"[Tiptoe] Documents: {len(documents)}, Embeddings: {embeddings.shape}")
    
    # Setup
    tiptoe = TiptoeSystem(target_dim=192, n_clusters=n_clusters)
    setup_start = time.perf_counter()
    setup_metrics = tiptoe.setup(embeddings, documents)
    setup_time = time.perf_counter() - setup_start
    
    print(f"Setup complete in {setup_time:.2f}s")
    
    # Generate test queries
    test_queries = generate_test_queries(documents, len(test_queries))
    print(f"Generated {len(test_queries)} queries from corpus documents")
    
    # Run queries
    total_upload = 0
    total_download = 0
    total_query_time = 0
    
    for i, query_text in enumerate(test_queries):
        # Encode query using the model
        if model is not None:
            query_embedding = model.encode([query_text])[0].astype(np.float32)
        else:
            # Fallback to random if no model (for testing only)
            query_embedding = np.random.randn(embeddings.shape[1]).astype(np.float32)
            print(f"Warning: Using random query embedding for query {i+1} (no model provided)")
        
        # Perform Tiptoe query
        query_start = time.perf_counter()
        retrieved_docs, query_metrics = tiptoe.query(query_embedding, top_k=top_k)
        query_time = time.perf_counter() - query_start
        
        # Accumulate metrics
        total_query_time += query_time
        total_upload += query_metrics.get('phase1_upload_bytes', 0) + query_metrics.get('phase2_upload_bytes', 0)
        total_download += query_metrics.get('phase1_download_bytes', 0) + query_metrics.get('phase2_download_bytes', 0)
        
        print(f"Query {i+1}: {query_time:.3f}s, {len(retrieved_docs)} docs, cluster {query_metrics.get('selected_cluster', 'N/A')}")
        if i < 3:  # Show detailed breakdown for first few queries
            print(f"  Phase 1: {query_metrics.get('phase1_time', 0):.3f}s, Phase 2: {query_metrics.get('phase2_time', 0):.3f}s")
    
    avg_metrics = {
        "system": "Tiptoe",
        "setup_time": setup_time,
        "avg_query_time": total_query_time / len(test_queries),
        "avg_upload_bytes": total_upload / len(test_queries),
        "avg_download_bytes": total_download / len(test_queries),
        "total_upload_bytes": total_upload,
        "total_download_bytes": total_download,
        "n_queries": len(test_queries),
        "setup_metrics": setup_metrics
    }
    
    print(f"Tiptoe Results:")
    print(f"  Avg query time: {avg_metrics['avg_query_time']:.3f}s")
    print(f"  Avg upload: {avg_metrics['avg_upload_bytes']:,} bytes")
    print(f"  Avg download: {avg_metrics['avg_download_bytes']:,} bytes")
    
    return avg_metrics


def compare_systems(pir_rag_results: Dict, graph_pir_results: Dict, tiptoe_results: Dict = None):
    """Compare results between the three systems."""
    print(f"\n=== COMPARISON SUMMARY ===")
    
    systems = ["PIR-RAG", "Graph-PIR"]
    results = [pir_rag_results, graph_pir_results]
    
    if tiptoe_results is not None:
        systems.append("Tiptoe")
        results.append(tiptoe_results)

    # Setup time comparison
    print(f"Setup Time:")
    for i, (system, result) in enumerate(zip(systems, results)):
        print(f"  {system}: {result['setup_time']:.2f}s")
    
    if len(results) >= 2:
        print(f"  Ratio (Graph/PIR): {results[1]['setup_time']/results[0]['setup_time']:.2f}x")
    if len(results) >= 3:
        print(f"  Ratio (Tiptoe/PIR): {results[2]['setup_time']/results[0]['setup_time']:.2f}x")

    # Query time comparison
    print(f"\nAverage Query Time:")
    for system, result in zip(systems, results):
        print(f"  {system}: {result['avg_query_time']:.3f}s")
    
    if len(results) >= 2:
        print(f"  Ratio (Graph/PIR): {results[1]['avg_query_time']/results[0]['avg_query_time']:.2f}x")
    if len(results) >= 3:
        print(f"  Ratio (Tiptoe/PIR): {results[2]['avg_query_time']/results[0]['avg_query_time']:.2f}x")

    # Communication cost comparison
    print(f"\nCommunication Costs (per query):")
    for system, result in zip(systems, results):
        total_comm = result["avg_upload_bytes"] + result["avg_download_bytes"]
        print(f"  {system}:")
        print(f"    Upload: {result['avg_upload_bytes']:,} bytes")
        print(f"    Download: {result['avg_download_bytes']:,} bytes")
        print(f"    Total: {total_comm:,} bytes")
    
    if len(results) >= 2:
        pir_total = results[0]["avg_upload_bytes"] + results[0]["avg_download_bytes"]
        graph_total = results[1]["avg_upload_bytes"] + results[1]["avg_download_bytes"]
        print(f"  Communication Ratio (Graph/PIR): {graph_total/pir_total:.2f}x")
    
    if len(results) >= 3:
        tiptoe_total = results[2]["avg_upload_bytes"] + results[2]["avg_download_bytes"]
        print(f"  Communication Ratio (Tiptoe/PIR): {tiptoe_total/pir_total:.2f}x")

    # Summary
    print(f"\n=== KEY INSIGHTS ===")
    if len(results) >= 2:
        if results[1]["setup_time"] < results[0]["setup_time"]:
            print(f"✓ Graph-PIR has {results[0]['setup_time']/results[1]['setup_time']:.1f}x faster setup")
        else:
            print(f"✓ PIR-RAG has {results[1]['setup_time']/results[0]['setup_time']:.1f}x faster setup")

        if results[1]["avg_query_time"] < results[0]["avg_query_time"]:
            print(f"✓ Graph-PIR has {results[0]['avg_query_time']/results[1]['avg_query_time']:.1f}x faster queries")
        else:
            print(f"✓ PIR-RAG has {results[1]['avg_query_time']/results[0]['avg_query_time']:.1f}x faster queries")

        if graph_total < pir_total:
            print(f"✓ Graph-PIR has {pir_total/graph_total:.1f}x lower communication cost")
        else:
            print(f"✓ PIR-RAG has {graph_total/pir_total:.1f}x lower communication cost")
    
    if len(results) >= 3:
        print(f"\n=== THREE-WAY COMPARISON ===")
        setup_times = [(systems[i], results[i]['setup_time']) for i in range(len(results))]
        setup_times.sort(key=lambda x: x[1])
        print(f"Setup time ranking: {' < '.join([f'{name} ({time:.2f}s)' for name, time in setup_times])}")
        
        query_times = [(systems[i], results[i]['avg_query_time']) for i in range(len(results))]
        query_times.sort(key=lambda x: x[1])
        print(f"Query time ranking: {' < '.join([f'{name} ({time:.3f}s)' for name, time in query_times])}")


def main():
    """Main experiment runner with command-line arguments."""
    parser = argparse.ArgumentParser(description='Compare PIR-RAG vs Graph-PIR vs Tiptoe communication efficiency')

    # Data arguments
    parser.add_argument('--embeddings_path', type=str, default=None,
                       help='Path to embeddings file (.npy format)')
    parser.add_argument('--corpus_path', type=str, default=None,
                       help='Path to corpus file (.csv format with "text" column)')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to sentence transformer model (optional, for reranking)')
    parser.add_argument('--data_size', type=int, default=1000,
                       help='Number of documents to use (default: 1000)')

    # Experiment arguments
    parser.add_argument('--n_queries', type=int, default=5,
                       help='Number of test queries to run (default: 5)')
    parser.add_argument('--n_clusters', type=int, default=100,
                       help='Number of clusters for PIR-RAG (default: 100)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of documents to retrieve per query (default: 10)')
    parser.add_argument('--key_length', type=int, default=1024,
                       help='Paillier key length for PIR-RAG (default: 1024)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results (default: results)')
    parser.add_argument('--output_prefix', type=str, default='pir_comparison',
                       help='Prefix for output files (default: pir_comparison)')

    # System arguments
    parser.add_argument('--systems', type=str, default='all',
                       choices=['all', 'pir-rag', 'graph-pir', 'tiptoe', 'pir-rag,graph-pir', 'pir-rag,tiptoe', 'graph-pir,tiptoe'],
                       help='Systems to compare: all, pir-rag, graph-pir, tiptoe, or comma-separated combinations (default: all)')
    parser.add_argument('--skip_pir_rag', action='store_true',
                       help='Skip PIR-RAG experiment (deprecated, use --systems instead)')
    parser.add_argument('--skip_graph_pir', action='store_true',
                       help='Skip Graph-PIR experiment (deprecated, use --systems instead)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    print("=== PIR-RAG vs Graph-PIR Communication Efficiency Comparison ===")

    # Print configuration
    print(f"Experiment configuration:")
    print(f"  Dataset size: {args.data_size} documents")
    if args.embeddings_path:
        print(f"  Embeddings: {args.embeddings_path}")
    if args.corpus_path:
        print(f"  Corpus: {args.corpus_path}")
    if args.model_path:
        print(f"  Model: {args.model_path}")
    print(f"  PIR-RAG clusters: {args.n_clusters}")
    print(f"  Test queries: {args.n_queries}")
    print(f"  Top-k retrieval: {args.top_k}")
    print(f"  Key length: {args.key_length}")
    print(f"  Output directory: {args.output_dir}")

    # Load data
    embeddings, documents = load_test_data(args.embeddings_path, args.corpus_path, args.data_size)

    # Load model for query encoding
    model = None
    if args.model_path:
        model = load_model(args.model_path)
    else:
        print("Warning: No model path provided. Will use random query embeddings.")
        print("For realistic results, provide --model_path argument.")

    # Determine which systems to run
    if args.systems == 'all':
        systems_to_run = ['pir-rag', 'graph-pir', 'tiptoe']
    else:
        systems_to_run = [s.strip() for s in args.systems.split(',')]
    
    # Handle deprecated skip flags
    if args.skip_pir_rag and 'pir-rag' in systems_to_run:
        systems_to_run.remove('pir-rag')
    if args.skip_graph_pir and 'graph-pir' in systems_to_run:
        systems_to_run.remove('graph-pir')
    
    print(f"Running experiments for: {', '.join(systems_to_run)}")

    # Run experiments
    results = []
    systems = []
    
    try:
        if 'pir-rag' in systems_to_run:
            print("\n=== Running PIR-RAG Experiment ===")
            pir_rag_results = run_pir_rag_experiment(
                embeddings, documents, args.n_clusters, args.n_queries, args.key_length, model
            )
            results.append(pir_rag_results)
            systems.append('PIR-RAG')

        if 'graph-pir' in systems_to_run:
            print("\n=== Running Graph-PIR Experiment ===")
            graph_pir_results = run_graph_pir_experiment(
                embeddings, documents, args.n_queries, args.top_k, model
            )
            results.append(graph_pir_results)
            systems.append('Graph-PIR')
            
        if 'tiptoe' in systems_to_run:
            print("\n=== Running Tiptoe Experiment ===")
            tiptoe_results = run_tiptoe_experiment(
                embeddings, documents, args.n_clusters, args.n_queries, model
            )
            results.append(tiptoe_results)
            systems.append('Tiptoe')

        # Compare results
        if len(results) >= 2:
            print(f"\n=== COMPARISON RESULTS ===")
            compare_systems(systems, results)
        elif len(results) == 1:
            print(f"\n=== {systems[0]} Results Only ===")
            print(f"Setup time: {results[0]['setup_time']:.2f}s")
            print(f"Avg query time: {results[0]['avg_query_time']:.3f}s")
            print(f"Avg communication: {results[0]['avg_upload_bytes'] + results[0]['avg_download_bytes']:,} bytes")

        # Save results
        if results:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
            results_df = pd.DataFrame(results)

            os.makedirs(args.output_dir, exist_ok=True)
            results_file = os.path.join(args.output_dir, f"{args.output_prefix}_{timestamp}.csv")
            results_df.to_csv(results_file, index=False)
            print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
