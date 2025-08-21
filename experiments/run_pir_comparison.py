"""
Communication Efficiency Comparison: PIR-RAG vs Graph-PIR

Compares the two systems on the same dataset to evaluate:
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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pir_rag import PIRRAGClient, PIRRAGServer
from graph_pir import GraphPIRSystem


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


def run_pir_rag_experiment(embeddings: np.ndarray, documents: List[str], 
                          n_clusters: int, n_queries: int = 10, key_length: int = 2048) -> Dict[str, Any]:
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
    
    np.random.seed(42)
    for i in range(n_queries):
        # Generate random query
        query_embedding = torch.tensor(np.random.randn(embeddings.shape[1]).astype(np.float32))
        
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
            print(f"First query: retrieved {len(retrieved_docs)} docs in {query_time:.3f}s")
    
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
                           n_queries: int = 10, top_k: int = 10) -> Dict[str, Any]:
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
    
    np.random.seed(42)
    for i in range(n_queries):
        # Generate random query (same as PIR-RAG for fair comparison)
        query_embedding = np.random.randn(embeddings.shape[1]).astype(np.float32)
        
        # Perform Graph-PIR query
        query_start = time.perf_counter()
        retrieved_docs, query_metrics = graph_pir.query(query_embedding, top_k=top_k)
        query_time = time.perf_counter() - query_start
        
        total_upload += query_metrics["phase1_upload_bytes"] + query_metrics["phase2_upload_bytes"]
        total_download += query_metrics["phase1_download_bytes"] + query_metrics["phase2_download_bytes"]
        total_query_time += query_time
        
        if i == 0:
            print(f"First query: retrieved {len(retrieved_docs)} docs in {query_time:.3f}s")
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


def compare_systems(pir_rag_results: Dict, graph_pir_results: Dict):
    """Compare results between the two systems."""
    print(f"\n=== COMPARISON SUMMARY ===")
    
    # Setup time comparison
    pir_setup = pir_rag_results["setup_time"]
    graph_setup = graph_pir_results["setup_time"]
    print(f"Setup Time:")
    print(f"  PIR-RAG: {pir_setup:.2f}s")
    print(f"  Graph-PIR: {graph_setup:.2f}s")
    print(f"  Ratio (Graph/PIR): {graph_setup/pir_setup:.2f}x")
    
    # Query time comparison
    pir_query = pir_rag_results["avg_query_time"]
    graph_query = graph_pir_results["avg_query_time"]
    print(f"\nAverage Query Time:")
    print(f"  PIR-RAG: {pir_query:.3f}s")
    print(f"  Graph-PIR: {graph_query:.3f}s")
    print(f"  Ratio (Graph/PIR): {graph_query/pir_query:.2f}x")
    
    # Communication cost comparison
    pir_total_comm = pir_rag_results["avg_upload_bytes"] + pir_rag_results["avg_download_bytes"]
    graph_total_comm = graph_pir_results["avg_upload_bytes"] + graph_pir_results["avg_download_bytes"]
    
    print(f"\nCommunication Costs (per query):")
    print(f"  PIR-RAG:")
    print(f"    Upload: {pir_rag_results['avg_upload_bytes']:,} bytes")
    print(f"    Download: {pir_rag_results['avg_download_bytes']:,} bytes")
    print(f"    Total: {pir_total_comm:,} bytes")
    print(f"  Graph-PIR:")
    print(f"    Upload: {graph_pir_results['avg_upload_bytes']:,} bytes")
    print(f"    Download: {graph_pir_results['avg_download_bytes']:,} bytes")
    print(f"    Total: {graph_total_comm:,} bytes")
    print(f"  Communication Ratio (Graph/PIR): {graph_total_comm/pir_total_comm:.2f}x")
    
    # Summary
    print(f"\n=== KEY INSIGHTS ===")
    if graph_setup < pir_setup:
        print(f"✓ Graph-PIR has {pir_setup/graph_setup:.1f}x faster setup")
    else:
        print(f"✓ PIR-RAG has {graph_setup/pir_setup:.1f}x faster setup")
        
    if graph_query < pir_query:
        print(f"✓ Graph-PIR has {pir_query/graph_query:.1f}x faster queries")
    else:
        print(f"✓ PIR-RAG has {graph_query/pir_query:.1f}x faster queries")
        
    if graph_total_comm < pir_total_comm:
        print(f"✓ Graph-PIR has {pir_total_comm/graph_total_comm:.1f}x lower communication cost")
    else:
        print(f"✓ PIR-RAG has {graph_total_comm/pir_total_comm:.1f}x lower communication cost")


def main():
    """Main experiment runner with command-line arguments."""
    parser = argparse.ArgumentParser(description='Compare PIR-RAG vs Graph-PIR communication efficiency')
    
    # Data arguments
    parser.add_argument('--embeddings_path', type=str, default=None,
                       help='Path to embeddings file (.npy format)')
    parser.add_argument('--corpus_path', type=str, default=None,
                       help='Path to corpus file (.csv format with "text" column)')
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
    parser.add_argument('--skip_pir_rag', action='store_true',
                       help='Skip PIR-RAG experiment (only run Graph-PIR)')
    parser.add_argument('--skip_graph_pir', action='store_true',
                       help='Skip Graph-PIR experiment (only run PIR-RAG)')
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
    print(f"  PIR-RAG clusters: {args.n_clusters}")
    print(f"  Test queries: {args.n_queries}")
    print(f"  Top-k retrieval: {args.top_k}")
    print(f"  Key length: {args.key_length}")
    print(f"  Output directory: {args.output_dir}")
    
    # Load data
    embeddings, documents = load_test_data(args.embeddings_path, args.corpus_path, args.data_size)
    
    # Run experiments
    results = []
    try:
        if not args.skip_pir_rag:
            pir_rag_results = run_pir_rag_experiment(
                embeddings, documents, args.n_clusters, args.n_queries, args.key_length
            )
            results.append(pir_rag_results)
        
        if not args.skip_graph_pir:
            graph_pir_results = run_graph_pir_experiment(
                embeddings, documents, args.n_queries, args.top_k
            )
            results.append(graph_pir_results)
        
        # Compare results if both systems were run
        if len(results) == 2:
            compare_systems(results[0], results[1])
        elif len(results) == 1:
            print(f"\n=== {results[0]['system']} Results Only ===")
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
