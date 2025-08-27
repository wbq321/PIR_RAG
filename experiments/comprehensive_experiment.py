"""
Comprehensive PIR Experiment Framework

This script conducts detailed performance and scalability experiments for 
all three PIR systems (PIR-RAG, Graph-PIR, Tiptoe) and collects comprehensive
timing data for analysis and plotting.
"""

import time
import numpy as np
import pandas as pd
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import argparse
from pathlib import Path

# Add PIR_RAG to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from pir_rag import PIRRAGClient, PIRRAGServer
from graph_pir import GraphPIRSystem
from tiptoe import TiptoeSystem
from sentence_transformers import SentenceTransformer

# Import retrieval performance testing
try:
    from test_retrieval_performance import RetrievalPerformanceTester
    RETRIEVAL_TESTING_AVAILABLE = True
except ImportError:
    RETRIEVAL_TESTING_AVAILABLE = False
    print("⚠️  Retrieval performance testing not available. Install required dependencies.")


class PIRExperimentRunner:
    """Comprehensive experiment runner for all PIR systems."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    def load_or_generate_data(self, embeddings_path: str = None, corpus_path: str = None, 
                             n_docs: int = 1000, embed_dim: int = 384, seed: int = 42) -> Tuple[np.ndarray, List[str]]:
        """Load real data or generate synthetic data if paths not provided."""
        
        # Try to load real data if paths provided
        if embeddings_path and corpus_path:
            try:
                if os.path.exists(embeddings_path) and os.path.exists(corpus_path):
                    print(f"Loading real data from {embeddings_path} and {corpus_path}...")
                    embeddings = np.load(embeddings_path)[:n_docs]
                    corpus_df = pd.read_csv(corpus_path)
                    documents = corpus_df['text'].iloc[:n_docs].tolist()
                    print(f"Loaded {len(documents)} documents, embedding dim: {embeddings.shape[1]}")
                    return embeddings, documents
                else:
                    print(f"Data files not found, falling back to synthetic data...")
            except Exception as e:
                print(f"Error loading data: {e}, falling back to synthetic data...")
        
        # Try default paths if no paths specified
        elif embeddings_path is None and corpus_path is None:
            default_embeddings = "data/embeddings_10000.npy"
            default_corpus = "data/corpus_10000.csv"
            
            if os.path.exists(default_embeddings) and os.path.exists(default_corpus):
                try:
                    print(f"Loading default MS MARCO data (first {n_docs} samples)...")
                    embeddings = np.load(default_embeddings)[:n_docs]
                    corpus_df = pd.read_csv(default_corpus)
                    documents = corpus_df['text'].iloc[:n_docs].tolist()
                    print(f"Loaded {len(documents)} documents, embedding dim: {embeddings.shape[1]}")
                    return embeddings, documents
                except Exception as e:
                    print(f"Error loading default data: {e}, falling back to synthetic data...")
        
        # Generate synthetic data as fallback
        print(f"Generating synthetic test data: {n_docs} documents, {embed_dim}D embeddings")
        return self.generate_test_data(n_docs, embed_dim, seed)
        
    def generate_test_data(self, n_docs: int, embed_dim: int = 384, seed: int = 42) -> Tuple[np.ndarray, List[str]]:
        """Generate synthetic test data."""
        np.random.seed(seed)
        embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
        documents = [
            f"Document {i}: This is test document {i} with content about topic {i % 10}. "
            f"It contains information relevant to research area {(i // 10) % 5}. "
            f"Additional context: {' '.join([f'keyword_{j}' for j in range(i % 7)])}"
            for i in range(n_docs)
        ]
        return embeddings, documents
    
    def run_pir_rag_experiment(self, embeddings: np.ndarray, documents: List[str], 
                              queries: List[np.ndarray], k_clusters: int = 5, 
                              top_k: int = 10) -> Dict[str, Any]:
        """Run PIR-RAG experiment with detailed timing."""
        print(f"Running PIR-RAG experiment (k_clusters={k_clusters})")
        
        # Setup phase
        setup_start = time.perf_counter()
        client = PIRRAGClient()
        server = PIRRAGServer()
        
        # Setup timing breakdown
        cluster_start = time.perf_counter()
        # Server does clustering and setup first
        server_setup_result = server.setup(embeddings, documents, k_clusters)
        clustering_time = time.perf_counter() - cluster_start
        
        server_setup_start = time.perf_counter()
        # Client setup with centroids from server
        client_setup_result = client.setup(server.centroids)
        server_setup_time = time.perf_counter() - server_setup_start
        
        total_setup_time = time.perf_counter() - setup_start
        
        # Query phase - detailed timing for each step
        query_times = []
        step_times = []
        communication_costs = []
        
        for i, query_embedding in enumerate(queries):
            print(f"  Query {i+1}/{len(queries)}")
            
            query_start = time.perf_counter()
            
            # Step 1: Find relevant clusters
            cluster_start = time.perf_counter()
            relevant_clusters = client.find_relevant_clusters(query_embedding, top_clusters=3)
            cluster_time = time.perf_counter() - cluster_start
            
            # Step 2: PIR retrieval
            pir_start = time.perf_counter()
            urls, pir_metrics = client.pir_retrieve(relevant_clusters, server)
            pir_time = time.perf_counter() - pir_start
            
            # Step 3: Reranking
            rerank_start = time.perf_counter()
            final_results = client.rerank_documents(query_embedding, urls, top_k=top_k)
            rerank_time = time.perf_counter() - rerank_start
            
            total_query_time = time.perf_counter() - query_start
            
            query_times.append(total_query_time)
            step_times.append({
                'cluster_selection_time': cluster_time,
                'pir_retrieval_time': pir_time,
                'reranking_time': rerank_time,
                'query_gen_time': pir_metrics.get('total_query_gen_time', 0),
                'server_time': pir_metrics.get('total_server_time', 0),
                'decode_time': pir_metrics.get('total_decode_time', 0)
            })
            communication_costs.append({
                'upload_bytes': pir_metrics.get('total_upload_bytes', 0),
                'download_bytes': pir_metrics.get('total_download_bytes', 0)
            })
        
        return {
            'system': 'PIR-RAG',
            'setup_time': total_setup_time,
            'clustering_time': clustering_time,
            'server_setup_time': server_setup_time,
            'avg_query_time': np.mean(query_times),
            'std_query_time': np.std(query_times),
            'query_times': query_times,
            'step_times': step_times,
            'avg_upload_bytes': np.mean([c['upload_bytes'] for c in communication_costs]),
            'avg_download_bytes': np.mean([c['download_bytes'] for c in communication_costs]),
            'communication_costs': communication_costs,
            'parameters': {'k_clusters': k_clusters, 'top_k': top_k},
            'n_documents': len(documents),
            'embedding_dim': embeddings.shape[1]
        }
    
    def run_graph_pir_experiment(self, embeddings: np.ndarray, documents: List[str], 
                                queries: List[np.ndarray], graph_params: Dict = None,
                                top_k: int = 10) -> Dict[str, Any]:
        """Run Graph-PIR experiment with detailed timing."""
        print(f"Running Graph-PIR experiment")
        
        if graph_params is None:
            graph_params = {'k_neighbors': 16, 'ef_construction': 200, 'max_connections': 16}
        
        # Setup phase
        setup_start = time.perf_counter()
        system = GraphPIRSystem()
        setup_metrics = system.setup(embeddings, documents, graph_params=graph_params)
        setup_time = time.perf_counter() - setup_start
        
        # Query phase - detailed timing
        query_times = []
        step_times = []
        communication_costs = []
        
        for i, query_embedding in enumerate(queries):
            print(f"  Query {i+1}/{len(queries)}")
            
            query_start = time.perf_counter()
            urls, query_metrics = system.query(query_embedding, top_k=top_k)
            query_time = time.perf_counter() - query_start
            
            query_times.append(query_time)
            step_times.append({
                'phase1_time': query_metrics.get('phase1_time', 0),
                'phase2_time': query_metrics.get('phase2_time', 0),
                'graph_traversal_time': query_metrics.get('graph_traversal_time', 0),
                'pir_time': query_metrics.get('pir_time', 0),
                'encryption_time': query_metrics.get('encryption_time', 0),
                'decryption_time': query_metrics.get('decryption_time', 0)
            })
            communication_costs.append({
                'upload_bytes': query_metrics.get('phase1_upload_bytes', 0) + query_metrics.get('phase2_upload_bytes', 0),
                'download_bytes': query_metrics.get('phase1_download_bytes', 0) + query_metrics.get('phase2_download_bytes', 0),
                'phase1_upload': query_metrics.get('phase1_upload_bytes', 0),
                'phase1_download': query_metrics.get('phase1_download_bytes', 0),
                'phase2_upload': query_metrics.get('phase2_upload_bytes', 0),
                'phase2_download': query_metrics.get('phase2_download_bytes', 0)
            })
        
        return {
            'system': 'Graph-PIR',
            'setup_time': setup_time,
            'graph_setup_time': setup_metrics.get('graph_setup_time', 0),
            'vector_pir_setup_time': setup_metrics.get('vector_pir_setup_time', 0),
            'doc_pir_setup_time': setup_metrics.get('doc_pir_setup_time', 0),
            'avg_query_time': np.mean(query_times),
            'std_query_time': np.std(query_times),
            'query_times': query_times,
            'step_times': step_times,
            'avg_upload_bytes': np.mean([c['upload_bytes'] for c in communication_costs]),
            'avg_download_bytes': np.mean([c['download_bytes'] for c in communication_costs]),
            'communication_costs': communication_costs,
            'parameters': graph_params,
            'n_documents': len(documents),
            'embedding_dim': embeddings.shape[1]
        }
    
    def run_tiptoe_experiment(self, embeddings: np.ndarray, documents: List[str], 
                             queries: List[np.ndarray], tiptoe_params: Dict = None,
                             top_k: int = 10) -> Dict[str, Any]:
        """Run Tiptoe experiment with detailed timing."""
        print(f"Running Tiptoe experiment")
        
        if tiptoe_params is None:
            tiptoe_params = {'k_clusters': 5, 'use_real_crypto': True}
        
        # Setup phase
        setup_start = time.perf_counter()
        system = TiptoeSystem()
        setup_metrics = system.setup(embeddings, documents, **tiptoe_params)
        setup_time = time.perf_counter() - setup_start
        
        # Query phase - detailed timing
        query_times = []
        step_times = []
        communication_costs = []
        
        for i, query_embedding in enumerate(queries):
            print(f"  Query {i+1}/{len(queries)}")
            
            query_start = time.perf_counter()
            urls, query_metrics = system.query(query_embedding, top_k=top_k)
            query_time = time.perf_counter() - query_start
            
            query_times.append(query_time)
            step_times.append({
                'cluster_selection_time': query_metrics.get('cluster_selection_time', 0),
                'ranking_time': query_metrics.get('ranking_time', 0),
                'pir_retrieval_time': query_metrics.get('retrieval_time', 0),
                'total_ranking_time': query_metrics.get('total_ranking_time', 0)
            })
            communication_costs.append({
                'upload_bytes': query_metrics.get('total_upload_bytes', 0),
                'download_bytes': query_metrics.get('total_download_bytes', 0),
                'pir_communication': query_metrics.get('pir_communication', 0)
            })
        
        return {
            'system': 'Tiptoe',
            'setup_time': setup_time,
            'crypto_setup_time': setup_metrics.get('crypto_setup_time', 0),
            'clustering_time': setup_metrics.get('clustering_time', 0),
            'avg_query_time': np.mean(query_times),
            'std_query_time': np.std(query_times),
            'query_times': query_times,
            'step_times': step_times,
            'avg_upload_bytes': np.mean([c['upload_bytes'] for c in communication_costs]),
            'avg_download_bytes': np.mean([c['download_bytes'] for c in communication_costs]),
            'communication_costs': communication_costs,
            'parameters': tiptoe_params,
            'n_documents': len(documents),
            'embedding_dim': embeddings.shape[1]
        }
    
    def run_scalability_experiment(self, doc_sizes: List[int] = [100, 500, 1000, 2000], 
                                  n_queries: int = 5, embed_dim: int = 384,
                                  embeddings_path: str = None, corpus_path: str = None) -> Dict[str, Any]:
        """Run scalability experiments across different dataset sizes."""
        print("Running scalability experiments...")
        
        scalability_results = {
            'doc_sizes': doc_sizes,
            'pir_rag_results': [],
            'graph_pir_results': [],
            'tiptoe_results': []
        }
        
        for n_docs in doc_sizes:
            print(f"\n=== Testing with {n_docs} documents ===")
            
            # Load or generate test data
            embeddings, documents = self.load_or_generate_data(
                embeddings_path, corpus_path, n_docs, embed_dim
            )
            queries = [np.random.randn(embeddings.shape[1]).astype(np.float32) for _ in range(n_queries)]
            
            # Test each system
            try:
                pir_rag_result = self.run_pir_rag_experiment(embeddings, documents, queries, k_clusters=min(5, n_docs//20))
                scalability_results['pir_rag_results'].append(pir_rag_result)
            except Exception as e:
                print(f"PIR-RAG failed for {n_docs} docs: {e}")
                scalability_results['pir_rag_results'].append(None)
            
            try:
                graph_pir_result = self.run_graph_pir_experiment(embeddings, documents, queries)
                scalability_results['graph_pir_results'].append(graph_pir_result)
            except Exception as e:
                print(f"Graph-PIR failed for {n_docs} docs: {e}")
                scalability_results['graph_pir_results'].append(None)
            
            try:
                tiptoe_result = self.run_tiptoe_experiment(embeddings, documents, queries, 
                                                         {'k_clusters': min(5, n_docs//20), 'use_real_crypto': True})
                scalability_results['tiptoe_results'].append(tiptoe_result)
            except Exception as e:
                print(f"Tiptoe failed for {n_docs} docs: {e}")
                scalability_results['tiptoe_results'].append(None)
        
        return scalability_results
    
    def run_parameter_sensitivity_experiment(self, n_docs: int = 1000, n_queries: int = 10,
                                           embeddings_path: str = None, corpus_path: str = None) -> Dict[str, Any]:
        """Run parameter sensitivity experiments."""
        print("Running parameter sensitivity experiments...")
        
        # Load or generate test data
        embeddings, documents = self.load_or_generate_data(
            embeddings_path, corpus_path, n_docs
        )
        queries = [np.random.randn(embeddings.shape[1]).astype(np.float32) for _ in range(n_queries)]
        
        sensitivity_results = {}
        
        # PIR-RAG: k_clusters sensitivity
        print("\nTesting PIR-RAG k_clusters sensitivity...")
        k_clusters_values = [3, 5, 10, 15, 20]
        pir_rag_sensitivity = []
        
        for k in k_clusters_values:
            if k > n_docs // 10:  # Skip if too many clusters
                continue
            try:
                result = self.run_pir_rag_experiment(embeddings, documents, queries[:3], k_clusters=k)
                pir_rag_sensitivity.append(result)
            except Exception as e:
                print(f"PIR-RAG failed for k_clusters={k}: {e}")
        
        sensitivity_results['pir_rag_k_clusters'] = pir_rag_sensitivity
        
        # Graph-PIR: k_neighbors sensitivity
        print("\nTesting Graph-PIR k_neighbors sensitivity...")
        k_neighbors_values = [8, 16, 32, 64]
        graph_pir_sensitivity = []
        
        for k in k_neighbors_values:
            try:
                graph_params = {'k_neighbors': k, 'ef_construction': 200, 'max_connections': k}
                result = self.run_graph_pir_experiment(embeddings, documents, queries[:3], graph_params=graph_params)
                graph_pir_sensitivity.append(result)
            except Exception as e:
                print(f"Graph-PIR failed for k_neighbors={k}: {e}")
        
        sensitivity_results['graph_pir_k_neighbors'] = graph_pir_sensitivity
        
        return sensitivity_results
    
    def run_retrieval_performance_experiment(self, n_docs: int = 1000, n_queries: int = 50, 
                                           embeddings_path: str = None, corpus_path: str = None,
                                           embed_dim: int = 384, top_k: int = 10) -> Dict[str, Any]:
        """Run retrieval performance experiments measuring IR quality metrics."""
        if not RETRIEVAL_TESTING_AVAILABLE:
            print("❌ Retrieval performance testing not available")
            return {}
            
        print(f"\n{'='*60}")
        print(f"Running Retrieval Performance Experiment")
        print(f"Documents: {n_docs}, Queries: {n_queries}, Top-K: {top_k}")
        print(f"{'='*60}")
        
        # Load or generate data
        embeddings, documents = self.load_or_generate_data(
            embeddings_path, corpus_path, n_docs, embed_dim
        )
        
        # Create retrieval performance tester
        tester = RetrievalPerformanceTester()
        
        # Generate queries for testing
        np.random.seed(42)
        queries = tester.generate_realistic_queries(embeddings, n_queries)
        
        # Test each system individually and collect results
        retrieval_results = {
            'experiment_info': {
                'timestamp': self.timestamp,
                'n_docs': n_docs,
                'n_queries': n_queries,
                'top_k': top_k,
                'embed_dim': embeddings.shape[1],
                'data_type': 'real' if embeddings_path else 'synthetic'
            }
        }
        
        # Test PIR-RAG
        try:
            pir_rag_client = PIRRAGClient()
            pir_rag_server = PIRRAGServer()
            
            # Setup server first (it does the clustering)
            k_clusters = min(5, len(documents)//20)
            server_setup_result = pir_rag_server.setup(embeddings, documents, k_clusters)
            
            # Setup client with centroids from server
            client_setup_result = pir_rag_client.setup(pir_rag_server.centroids)
            
            pir_rag_system = (pir_rag_client, pir_rag_server)
            pir_rag_results = tester.test_retrieval_performance(
                "PIR-RAG", pir_rag_system, embeddings, documents, queries, top_k
            )
            retrieval_results['pir_rag'] = pir_rag_results
            print(f"✅ PIR-RAG retrieval test completed")
        except Exception as e:
            print(f"❌ PIR-RAG retrieval test failed: {e}")
            retrieval_results['pir_rag'] = None
        
        # Test Graph-PIR
        try:
            graph_pir_system = GraphPIRSystem()
            graph_pir_results = tester.test_retrieval_performance(
                "Graph-PIR", graph_pir_system, embeddings, documents, queries, top_k
            )
            retrieval_results['graph_pir'] = graph_pir_results
            print(f"✅ Graph-PIR retrieval test completed")
        except Exception as e:
            print(f"❌ Graph-PIR retrieval test failed: {e}")
            retrieval_results['graph_pir'] = None
        
        # Test Tiptoe
        try:
            tiptoe_system = TiptoeSystem()
            tiptoe_results = tester.test_retrieval_performance(
                "Tiptoe", tiptoe_system, embeddings, documents, queries, top_k
            )
            retrieval_results['tiptoe'] = tiptoe_results
            print(f"✅ Tiptoe retrieval test completed")
        except Exception as e:
            print(f"❌ Tiptoe retrieval test failed: {e}")
            retrieval_results['tiptoe'] = None
        
        return retrieval_results
    
    def save_results(self, results: Dict[str, Any], experiment_name: str):
        """Save experiment results to files."""
        
        # Save detailed JSON results
        json_path = self.output_dir / f"{experiment_name}_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json.dump(convert_numpy(results), f, indent=2)
        
        # Save CSV summary
        csv_path = self.output_dir / f"{experiment_name}_summary_{self.timestamp}.csv"
        
        if experiment_name == "scalability":
            # Create scalability summary CSV
            scalability_data = []
            for i, n_docs in enumerate(results['doc_sizes']):
                row = {'n_documents': n_docs}
                
                for system, key in [('PIR-RAG', 'pir_rag_results'), 
                                   ('Graph-PIR', 'graph_pir_results'),
                                   ('Tiptoe', 'tiptoe_results')]:
                    if i < len(results[key]) and results[key][i] is not None:
                        result = results[key][i]
                        row.update({
                            f'{system}_setup_time': result['setup_time'],
                            f'{system}_avg_query_time': result['avg_query_time'],
                            f'{system}_std_query_time': result['std_query_time'],
                            f'{system}_avg_upload_bytes': result['avg_upload_bytes'],
                            f'{system}_avg_download_bytes': result['avg_download_bytes']
                        })
                    else:
                        row.update({
                            f'{system}_setup_time': None,
                            f'{system}_avg_query_time': None,
                            f'{system}_std_query_time': None,
                            f'{system}_avg_upload_bytes': None,
                            f'{system}_avg_download_bytes': None
                        })
                
                scalability_data.append(row)
            
            pd.DataFrame(scalability_data).to_csv(csv_path, index=False)
        
        print(f"Results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Comprehensive PIR Experiments")
    parser.add_argument("--experiment", choices=["single", "scalability", "sensitivity", "retrieval", "all"], 
                       default="all", help="Type of experiment to run")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--n-docs", type=int, default=1000, help="Number of documents for single experiment")
    parser.add_argument("--n-queries", type=int, default=10, help="Number of queries to test")
    parser.add_argument("--embed-dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--embeddings-path", type=str, default=None, 
                       help="Path to embeddings file (.npy format)")
    parser.add_argument("--corpus-path", type=str, default=None,
                       help="Path to corpus file (.csv format with 'text' column)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval evaluation")
    
    args = parser.parse_args()
    
    runner = PIRExperimentRunner(args.output_dir)
    
    if args.experiment in ["single", "all"]:
        print(f"Running single experiment with {args.n_docs} documents...")
        
        # Load or generate data with flexible options
        embeddings, documents = runner.load_or_generate_data(
            embeddings_path=args.embeddings_path,
            corpus_path=args.corpus_path,
            n_docs=args.n_docs,
            embed_dim=args.embed_dim
        )
        queries = [np.random.randn(embeddings.shape[1]).astype(np.float32) for _ in range(args.n_queries)]
        
        # Run all three systems
        results = {}
        
        try:
            results['pir_rag'] = runner.run_pir_rag_experiment(embeddings, documents, queries)
        except Exception as e:
            print(f"PIR-RAG experiment failed: {e}")
            
        try:
            results['graph_pir'] = runner.run_graph_pir_experiment(embeddings, documents, queries)
        except Exception as e:
            print(f"Graph-PIR experiment failed: {e}")
            
        try:
            results['tiptoe'] = runner.run_tiptoe_experiment(embeddings, documents, queries)
        except Exception as e:
            print(f"Tiptoe experiment failed: {e}")
        
        runner.save_results(results, "single_experiment")
    
    if args.experiment in ["scalability", "all"]:
        print("Running scalability experiments...")
        scalability_results = runner.run_scalability_experiment(
            doc_sizes=[100, 500, 1000, 2000], 
            n_queries=5, 
            embed_dim=args.embed_dim,
            embeddings_path=args.embeddings_path,
            corpus_path=args.corpus_path
        )
        runner.save_results(scalability_results, "scalability")
    
    if args.experiment in ["sensitivity", "all"]:
        print("Running parameter sensitivity experiments...")
        sensitivity_results = runner.run_parameter_sensitivity_experiment(
            n_docs=args.n_docs, 
            n_queries=5,
            embeddings_path=args.embeddings_path,
            corpus_path=args.corpus_path
        )
        runner.save_results(sensitivity_results, "parameter_sensitivity")
    
    if args.experiment in ["retrieval", "all"]:
        print("Running retrieval performance experiments...")
        retrieval_results = runner.run_retrieval_performance_experiment(
            n_docs=args.n_docs,
            n_queries=args.n_queries * 5,  # More queries for retrieval eval
            embeddings_path=args.embeddings_path,
            corpus_path=args.corpus_path,
            embed_dim=args.embed_dim,
            top_k=args.top_k
        )
        if retrieval_results:
            runner.save_results(retrieval_results, "retrieval_performance")
    
    print("All experiments completed!")


if __name__ == "__main__":
    main()
