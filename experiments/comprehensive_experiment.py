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
import torch
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


def generate_test_queries(documents: List[str], n_queries: int = 10, seed: int = 42) -> List[str]:
    """Generate realistic test queries by sampling from corpus documents."""
    if len(documents) == 0:
        raise ValueError("Cannot generate queries from empty document list")

    # Sample random documents to use as queries with local random generator
    rng = np.random.RandomState(seed)  # Use local generator to avoid state conflicts
    query_indices = rng.choice(len(documents), size=min(n_queries, len(documents)), replace=False)

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


class PIRExperimentRunner:
    """Comprehensive experiment runner for all PIR systems."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_default_k_clusters(self, n_docs: int, user_value: int = None) -> int:
        """Calculate default k_clusters value as n_docs/20, with minimum of 1."""
        if user_value is not None:
            return user_value
        return max(1, n_docs // 20)

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
        """Generate synthetic test data using local random generator to avoid state conflicts."""
        # Use local random generator to avoid affecting global random state
        rng = np.random.RandomState(seed)
        embeddings = rng.randn(n_docs, embed_dim).astype(np.float32)
        documents = [
            f"Document {i}: This is test document {i} with content about topic {i % 10}. "
            f"It contains information relevant to research area {(i // 10) % 5}. "
            f"Additional context: {' '.join([f'keyword_{j}' for j in range(i % 7)])}"
            for i in range(n_docs)
        ]
        return embeddings, documents

    def run_pir_rag_experiment(self, embeddings: np.ndarray, documents: List[str],
                              queries: List[np.ndarray], k_clusters: int = 5,
                              cluster_top_k: int = 3, top_k: int = 10) -> Dict[str, Any]:
        """Run PIR-RAG experiment with detailed timing."""
        print(f"Running PIR-RAG experiment (k_clusters={k_clusters}, cluster_top_k={cluster_top_k})")

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
            query_tensor = torch.tensor(query_embedding) if not isinstance(query_embedding, torch.Tensor) else query_embedding
            relevant_clusters = client.find_relevant_clusters(query_tensor, top_k=cluster_top_k)
            cluster_time = time.perf_counter() - cluster_start

            # Step 2: PIR retrieval (now returns URLs and embeddings together)
            pir_start = time.perf_counter()
            doc_tuples, pir_metrics = client.pir_retrieve(relevant_clusters, server)
            pir_time = time.perf_counter() - pir_start

            # Step 3: Reranking (using embeddings from PIR, no server request)
            rerank_start = time.perf_counter()
            final_results = client.rerank_documents(query_tensor, doc_tuples, top_k=top_k)
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
            'parameters': {'k_clusters': k_clusters, 'cluster_top_k': cluster_top_k, 'top_k': top_k},
            'n_documents': len(documents),
            'embedding_dim': embeddings.shape[1]
        }

    def run_graph_pir_experiment(self, embeddings: np.ndarray, documents: List[str],
                                queries: List[np.ndarray], graph_params: Dict = None,
                                top_k: int = 10) -> Dict[str, Any]:
        """Run Graph-PIR experiment with detailed timing."""
        print(f"Running Graph-PIR experiment with {len(documents)} docs")

        if graph_params is None:
            graph_params = {'k_neighbors': 16, 'ef_construction': 20, 'max_connections': 16,
                           'max_iterations': 5, 'parallel': 1, 'ef_search': 30}

        print(f"  Graph params: {graph_params}")

        # Setup phase
        setup_start = time.perf_counter()
        try:
            system = GraphPIRSystem()
            print(f"  ✅ GraphPIRSystem created")
            setup_metrics = system.setup(embeddings, documents, graph_params=graph_params)
            print(f"  ✅ GraphPIR setup completed")
        except Exception as e:
            print(f"  ❌ GraphPIR setup failed: {e}")
            raise
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
        print(f"Running Tiptoe experiment with {len(documents)} docs")

        if tiptoe_params is None:
            tiptoe_params = {'k_clusters': 5, 'use_real_crypto': True}

        print(f"  Tiptoe params: {tiptoe_params}")

        # Setup phase
        setup_start = time.perf_counter()
        try:
            system = TiptoeSystem()
            print(f"  ✅ TiptoeSystem created")
            setup_metrics = system.setup(embeddings, documents, **tiptoe_params)
            print(f"  ✅ Tiptoe setup completed")
        except Exception as e:
            print(f"  ❌ Tiptoe setup failed: {e}")
            raise
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
                'upload_bytes': query_metrics.get('upload_bytes', 0),
                'download_bytes': query_metrics.get('download_bytes', 0),
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

    def run_scalability_experiment(self, doc_sizes: List[int] = [500, 1000, 2000, 5000],
                                  n_queries: int = 5, embed_dim: int = 384,
                                  embeddings_path: str = None, corpus_path: str = None,
                                  pir_rag_params: Dict = None, graph_pir_params: Dict = None,
                                  tiptoe_params: Dict = None) -> Dict[str, Any]:
        """Run scalability experiments across different dataset sizes."""
        print("Running scalability experiments...")

        # Set default parameters if not provided
        if pir_rag_params is None:
            pir_rag_params = {'k_clusters': None, 'cluster_top_k': 3}  # None will be calculated per doc size
        if graph_pir_params is None:
            graph_pir_params = {'k_neighbors': 16, 'ef_construction': 100, 'max_connections': 16,
                               'max_iterations': 10, 'parallel': 1, 'ef_search': 30,
                               'max_neighbors_per_step': 10}  # FIXED: Limit PIR queries per step for consistent performance
        if tiptoe_params is None:
            # FIXED: Use fixed cluster count for fair scalability comparison
            # PIR-RAG scales clusters with dataset size (adaptive clustering)
            # Tiptoe uses fixed clusters to test scaling performance with constant cluster complexity
            tiptoe_params = {'k_clusters': 50, 'use_real_crypto': True}  # Fixed 50 clusters for all sizes

        # OVERRIDE: Force fixed cluster count for scalability testing even if provided via args
        # This ensures fair comparison across dataset sizes
        if 'k_clusters' not in tiptoe_params or tiptoe_params['k_clusters'] is None:
            tiptoe_params['k_clusters'] = 50

        scalability_results = {
            'doc_sizes': doc_sizes,
            'pir_rag_results': [],
            'graph_pir_results': [],
            'tiptoe_results': []
        }

        # FIXED: For real data, generate realistic queries from actual embeddings
        # First load a sample of embeddings to generate realistic queries
        if embeddings_path and corpus_path and os.path.exists(embeddings_path):
            print("Using real dataset - generating realistic queries from actual embeddings...")
            sample_embeddings = np.load(embeddings_path)[:max(doc_sizes)]  # Load largest needed size
            sample_documents = pd.read_csv(corpus_path)['text'].iloc[:max(doc_sizes)].tolist()

            # Generate realistic queries by sampling from actual embeddings
            np.random.seed(12345)  # Fixed seed for consistency
            query_indices = np.random.choice(len(sample_embeddings), size=n_queries, replace=False)
            test_queries = [sample_embeddings[i] for i in query_indices]
            print(f"Generated {len(test_queries)} realistic queries from dataset (indices: {query_indices})")
        else:
            # Fallback to synthetic queries for synthetic data
            print("Using synthetic data - generating synthetic queries...")
            np.random.seed(12345)  # Different seed from data generation to avoid conflicts
            test_queries = [np.random.randn(embed_dim).astype(np.float32) for _ in range(n_queries)]
            print(f"Generated {len(test_queries)} synthetic test queries (seed=12345)")

        for n_docs in doc_sizes:
            print(f"\n=== Testing with {n_docs} documents ===")

            # Load or generate test data
            embeddings, documents = self.load_or_generate_data(
                embeddings_path, corpus_path, n_docs, embed_dim
            )
            # Use the pre-generated consistent queries
            queries = test_queries

            # Test each system
            try:
                print(f"  Running PIR-RAG with {n_docs} documents...")
                # Adjust cluster numbers for dataset size
                adjusted_pir_rag_params = pir_rag_params.copy()
                adjusted_pir_rag_params['k_clusters'] = self.get_default_k_clusters(n_docs, adjusted_pir_rag_params['k_clusters'])
                print(f"  PIR-RAG using {adjusted_pir_rag_params['k_clusters']} clusters")

                pir_rag_result = self.run_pir_rag_experiment(
                    embeddings, documents, queries,
                    k_clusters=adjusted_pir_rag_params['k_clusters'],
                    cluster_top_k=adjusted_pir_rag_params['cluster_top_k']
                )
                scalability_results['pir_rag_results'].append(pir_rag_result)
                print(f"  ✅ PIR-RAG completed for {n_docs} docs")
            except Exception as e:
                print(f"  ❌ PIR-RAG failed for {n_docs} docs: {e}")
                import traceback
                traceback.print_exc()
                scalability_results['pir_rag_results'].append(None)

            try:
                print(f"  Running Graph-PIR with {n_docs} documents...")
                graph_pir_result = self.run_graph_pir_experiment(
                    embeddings, documents, queries,
                    graph_params=graph_pir_params
                )
                scalability_results['graph_pir_results'].append(graph_pir_result)
                print(f"  ✅ Graph-PIR completed for {n_docs} docs")
            except Exception as e:
                print(f"  ❌ Graph-PIR failed for {n_docs} docs: {e}")
                import traceback
                traceback.print_exc()
                scalability_results['graph_pir_results'].append(None)

            try:
                print(f"  Running Tiptoe with {n_docs} documents...")
                # FIXED: Keep cluster count fixed for fair comparison across dataset sizes
                adjusted_tiptoe_params = tiptoe_params.copy()
                # Don't adjust cluster count - use fixed value for all dataset sizes
                print(f"  Tiptoe using FIXED {adjusted_tiptoe_params['k_clusters']} clusters (consistent across all sizes)")

                tiptoe_result = self.run_tiptoe_experiment(
                    embeddings, documents, queries,
                    tiptoe_params=adjusted_tiptoe_params
                )
                scalability_results['tiptoe_results'].append(tiptoe_result)
                print(f"  ✅ Tiptoe completed for {n_docs} docs")
            except Exception as e:
                print(f"  ❌ Tiptoe failed for {n_docs} docs: {e}")
                import traceback
                traceback.print_exc()
                scalability_results['tiptoe_results'].append(None)

        return scalability_results

    def run_parameter_sensitivity_experiment(self, n_docs: int = 1000, n_queries: int = 10,
                                           embeddings_path: str = None, corpus_path: str = None,
                                           base_pir_rag_params: Dict = None,
                                           base_graph_pir_params: Dict = None) -> Dict[str, Any]:
        """Run parameter sensitivity experiments."""
        print("Running parameter sensitivity experiments...")

        # Set default base parameters if not provided
        if base_pir_rag_params is None:
            base_pir_rag_params = {'k_clusters': None, 'cluster_top_k': 3}  # None will be calculated based on n_docs
        if base_graph_pir_params is None:
            base_graph_pir_params = {'k_neighbors': 16, 'ef_construction': 100, 'max_connections': 16,
                                   'max_iterations': 5, 'parallel': 1, 'ef_search': 30}

        # Load or generate test data
        embeddings, documents = self.load_or_generate_data(
            embeddings_path, corpus_path, n_docs
        )

        # Calculate default k_clusters if not provided
        if base_pir_rag_params['k_clusters'] is None:
            base_pir_rag_params['k_clusters'] = self.get_default_k_clusters(n_docs, None)

        # Generate consistent test queries with fixed seed
        np.random.seed(54321)  # Fixed seed for parameter sensitivity testing
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
                result = self.run_pir_rag_experiment(
                    embeddings, documents, queries[:3],
                    k_clusters=k,
                    cluster_top_k=base_pir_rag_params['cluster_top_k']
                )
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
                graph_params = base_graph_pir_params.copy()
                graph_params['k_neighbors'] = k
                graph_params['max_connections'] = k  # Adjust max_connections to match k_neighbors

                result = self.run_graph_pir_experiment(
                    embeddings, documents, queries[:3],
                    graph_params=graph_params
                )
                graph_pir_sensitivity.append(result)
            except Exception as e:
                print(f"Graph-PIR failed for k_neighbors={k}: {e}")

        sensitivity_results['graph_pir_k_neighbors'] = graph_pir_sensitivity

        return sensitivity_results

    def run_retrieval_performance_experiment(self, n_docs: int = 1000, n_queries: int = 50,
                                           embeddings_path: str = None, corpus_path: str = None,
                                           embed_dim: int = 384, top_k: int = 10,
                                           pir_rag_k_clusters: int = 5,
                                           pir_rag_cluster_top_k: int = 3,
                                           graph_params: Dict = None,
                                           tiptoe_k_clusters: int = None) -> Dict[str, Any]:
        """Run retrieval performance experiments measuring IR quality metrics."""
        if not RETRIEVAL_TESTING_AVAILABLE:
            print("❌ Retrieval performance testing not available")
            return {}

        print(f"\n{'='*60}")
        print(f"Running Retrieval Performance Experiment (HYBRID APPROACH)")
        print(f"• Phase 1: Plaintext simulation for accurate retrieval quality")
        print(f"• Phase 2: Real PIR operations for realistic performance metrics")
        print(f"Documents: {n_docs}, Queries: {n_queries}, Top-K: {top_k}")
        print(f"{'='*60}")

        # Load or generate data
        embeddings, documents = self.load_or_generate_data(
            embeddings_path, corpus_path, n_docs, embed_dim
        )

        # Create retrieval performance tester
        tester = RetrievalPerformanceTester()

        # Generate queries for testing with fixed seed for consistency
        np.random.seed(67890)  # Fixed seed for retrieval performance testing
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
            pir_rag_system = (pir_rag_client, pir_rag_server)

            # FIXED: Remove redundant setup - let test_retrieval_performance handle it
            pir_rag_results = tester.test_retrieval_performance(
                "PIR-RAG", pir_rag_system, embeddings, documents, queries, top_k,
                pir_rag_k_clusters=pir_rag_k_clusters
            )
            retrieval_results['pir_rag'] = pir_rag_results
            print(f"✅ PIR-RAG retrieval test completed")
        except Exception as e:
            print(f"❌ PIR-RAG retrieval test failed: {e}")
            retrieval_results['pir_rag'] = None

        # Test Graph-PIR
        try:
            graph_pir_system = GraphPIRSystem()
            # Pass graph_params to the test method instead of pre-setting up
            graph_pir_results = tester.test_retrieval_performance(
                "Graph-PIR", graph_pir_system, embeddings, documents, queries, top_k,
                graph_params=graph_params
            )
            retrieval_results['graph_pir'] = graph_pir_results
            print(f"✅ Graph-PIR retrieval test completed")
        except Exception as e:
            print(f"❌ Graph-PIR retrieval test failed: {e}")
            retrieval_results['graph_pir'] = None

        # Test Tiptoe
        try:
            tiptoe_system = TiptoeSystem()
            # FIXED: Remove redundant setup - let test_retrieval_performance handle it
            tiptoe_results = tester.test_retrieval_performance(
                "Tiptoe", tiptoe_system, embeddings, documents, queries, top_k,
                tiptoe_k_clusters=tiptoe_k_clusters
            )
            retrieval_results['tiptoe'] = tiptoe_results
            print(f"✅ Tiptoe retrieval test completed")
        except Exception as e:
            print(f"❌ Tiptoe retrieval test failed: {e}")
            retrieval_results['tiptoe'] = None

        # Print hybrid approach summary
        self._print_hybrid_retrieval_summary(retrieval_results)

        return retrieval_results

    def _print_hybrid_retrieval_summary(self, results: Dict[str, Any]):
        """Print a summary of hybrid retrieval performance results."""

        print(f"\n{'='*70}")
        print(f"HYBRID RETRIEVAL PERFORMANCE SUMMARY")
        print(f"{'='*70}")

        experiment_info = results.get('experiment_info', {})
        print(f"Test Configuration:")
        print(f"  Documents: {experiment_info.get('n_docs', 'N/A'):,}")
        print(f"  Queries: {experiment_info.get('n_queries', 'N/A')}")
        print(f"  Top-K: {experiment_info.get('top_k', 'N/A')}")
        print(f"  Data Type: {experiment_info.get('data_type', 'N/A')}")

        # Table header
        print(f"\nHybrid Approach Results:")
        print(f"{'System':<12} {'Setup(s)':<10} {'Sim.Time(s)':<12} {'PIR Time(s)':<12} {'P@K':<8} {'R@K':<8} {'NDCG@K':<10} {'Comm(KB)':<10}")
        print("-" * 82)

        for system_name in ['pir_rag', 'graph_pir', 'tiptoe']:
            system_results = results.get(system_name)
            if system_results is None:
                print(f"{system_name.upper():<12} {'ERROR':<10} {'--':<12} {'--':<12} {'--':<8} {'--':<8} {'--':<10} {'--':<10}")
                continue

            # Check if this is hybrid approach results
            if system_results.get('hybrid_approach', False):
                setup_time = f"{system_results.get('setup_time', 0):.2f}"
                sim_time = f"{system_results.get('avg_quality_simulation_time', 0):.3f}"
                pir_time = f"{system_results.get('avg_pir_performance_time', 0):.3f}"
                precision = f"{system_results.get('avg_precision_at_k', 0):.3f}"
                recall = f"{system_results.get('avg_recall_at_k', 0):.3f}"
                ndcg = f"{system_results.get('avg_ndcg_at_k', 0):.3f}"
                comm = f"{system_results.get('avg_communication_bytes', 0)/1024:.1f}"

                print(f"{system_name.upper():<12} {setup_time:<10} {sim_time:<12} {pir_time:<12} {precision:<8} {recall:<8} {ndcg:<10} {comm:<10}")
            else:
                print(f"{system_name.upper():<12} {'OLD':<10} {'--':<12} {'--':<12} {'--':<8} {'--':<8} {'--':<10} {'--':<10}")

        print(f"\nHybrid Approach Benefits:")
        print(f"✅ Accurate quality metrics (no PIR corruption)")
        print(f"✅ Realistic performance measurements")
        print(f"✅ Preserved original PIR implementations")
        print(f"✅ Simulation Time: Plaintext quality calculation")
        print(f"✅ PIR Time: Real encrypted operations")

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

        elif experiment_name == "retrieval_performance":
            # Create retrieval performance summary CSV
            retrieval_data = []

            # Extract experiment info
            exp_info = results.get('experiment_info', {})
            base_row = {
                'n_documents': exp_info.get('n_docs', 'N/A'),
                'n_queries': exp_info.get('n_queries', 'N/A'),
                'top_k': exp_info.get('top_k', 'N/A'),
                'embed_dim': exp_info.get('embed_dim', 'N/A'),
                'data_type': exp_info.get('data_type', 'N/A')
            }

            # Process each system's results
            for system_key, system_name in [('pir_rag', 'PIR-RAG'),
                                          ('graph_pir', 'Graph-PIR'),
                                          ('tiptoe', 'Tiptoe')]:
                row = base_row.copy()
                row['system'] = system_name

                system_results = results.get(system_key)
                if system_results is not None:
                    row.update({
                        'setup_time': system_results.get('setup_time', 'N/A'),
                        'avg_query_time': system_results.get('avg_query_time', 'N/A'),
                        'std_query_time': system_results.get('std_query_time', 'N/A'),
                        'avg_upload_bytes': system_results.get('avg_upload_bytes', 'N/A'),
                        'avg_download_bytes': system_results.get('avg_download_bytes', 'N/A'),
                        'recall_at_1': system_results.get('recall_at_1', 'N/A'),
                        'recall_at_5': system_results.get('recall_at_5', 'N/A'),
                        'recall_at_10': system_results.get('recall_at_10', 'N/A'),
                        'precision_at_1': system_results.get('precision_at_1', 'N/A'),
                        'precision_at_5': system_results.get('precision_at_5', 'N/A'),
                        'precision_at_10': system_results.get('precision_at_10', 'N/A'),
                        'ndcg_at_10': system_results.get('ndcg_at_10', 'N/A'),
                        'mrr': system_results.get('mrr', 'N/A')
                    })
                else:
                    row.update({
                        'setup_time': 'Error',
                        'avg_query_time': 'Error',
                        'std_query_time': 'Error',
                        'avg_upload_bytes': 'Error',
                        'avg_download_bytes': 'Error',
                        'recall_at_1': 'Error',
                        'recall_at_5': 'Error',
                        'recall_at_10': 'Error',
                        'precision_at_1': 'Error',
                        'precision_at_5': 'Error',
                        'precision_at_10': 'Error',
                        'ndcg_at_10': 'Error',
                        'mrr': 'Error'
                    })

                retrieval_data.append(row)

            pd.DataFrame(retrieval_data).to_csv(csv_path, index=False)

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

    # PIR-RAG specific arguments
    parser.add_argument("--pir-rag-k-clusters", type=int, default=None,
                       help="Number of clusters for PIR-RAG (default: n_docs/20)")
    parser.add_argument("--pir-rag-cluster-top-k", type=int, default=3,
                       help="Number of top clusters to retrieve in PIR-RAG")

    # Graph-PIR specific arguments
    parser.add_argument("--graph-pir-k-neighbors", type=int, default=16,
                       help="Number of neighbors in HNSW graph for Graph-PIR")
    parser.add_argument("--graph-pir-ef-construction", type=int, default=100,
                       help="ef_construction parameter for HNSW graph building")
    parser.add_argument("--graph-pir-max-connections", type=int, default=16,
                       help="Maximum connections per node in HNSW graph")
    parser.add_argument("--graph-pir-ef-search", type=int, default=30,
                       help="ef parameter for graph search")
    parser.add_argument("--graph-pir-max-iterations", type=int, default=5,
                       help="Maximum number of graph traversal iterations/turns (GraphANN maxStep)")
    parser.add_argument("--graph-pir-parallel", type=int, default=1,
                       help="Number of parallel explorations per step (GraphANN parallel)")
    parser.add_argument("--graph-pir-max-neighbors-per-step", type=int, default=5,
                       help="Maximum neighbors to PIR query per step for consistent performance")

    # Tiptoe specific arguments
    parser.add_argument("--tiptoe-k-clusters", type=int, default=None,
                       help="Number of clusters for Tiptoe (default: n_docs/20)")
    parser.add_argument("--tiptoe-use-real-crypto", action="store_true", default=True,
                       help="Use real cryptography for Tiptoe (default: True)")
    parser.add_argument("--tiptoe-no-real-crypto", dest="tiptoe_use_real_crypto",
                       action="store_false", help="Disable real cryptography for Tiptoe")
    parser.add_argument("--tiptoe-poly-degree", type=int, default=8192,
                       help="Polynomial degree for Tiptoe homomorphic encryption")
    parser.add_argument("--tiptoe-plain-modulus", type=int, default=1024,
                       help="Plain modulus for Tiptoe homomorphic encryption")

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

        # Generate consistent test queries with fixed seed
        np.random.seed(98765)  # Fixed seed for main experiment testing
        queries = [np.random.randn(embeddings.shape[1]).astype(np.float32) for _ in range(args.n_queries)]

        # Run all three systems with configured parameters
        results = {}

        try:
            # PIR-RAG with custom parameters
            k_clusters = runner.get_default_k_clusters(args.n_docs, args.pir_rag_k_clusters)
            results['pir_rag'] = runner.run_pir_rag_experiment(
                embeddings, documents, queries,
                k_clusters=k_clusters,
                cluster_top_k=args.pir_rag_cluster_top_k,
                top_k=args.top_k
            )
            print(f"✅ PIR-RAG completed with k_clusters={k_clusters}")
        except Exception as e:
            print(f"❌ PIR-RAG experiment failed: {e}")

        try:
            # Graph-PIR with custom parameters
            graph_params = {
                'k_neighbors': args.graph_pir_k_neighbors,
                'ef_construction': args.graph_pir_ef_construction,
                'max_connections': args.graph_pir_max_connections,
                'ef_search': args.graph_pir_ef_search,
                'max_iterations': args.graph_pir_max_iterations,
                'parallel': args.graph_pir_parallel,
                'max_neighbors_per_step': args.graph_pir_max_neighbors_per_step
            }
            results['graph_pir'] = runner.run_graph_pir_experiment(
                embeddings, documents, queries,
                graph_params=graph_params,
                top_k=args.top_k
            )
            print(f"✅ Graph-PIR completed with k_neighbors={args.graph_pir_k_neighbors}")
        except Exception as e:
            print(f"❌ Graph-PIR experiment failed: {e}")

        try:
            # Tiptoe with custom parameters
            tiptoe_params = {
                'k_clusters': runner.get_default_k_clusters(args.n_docs, args.tiptoe_k_clusters),
                'use_real_crypto': args.tiptoe_use_real_crypto,
                'poly_degree': args.tiptoe_poly_degree,
                'plain_modulus': args.tiptoe_plain_modulus
            }
            results['tiptoe'] = runner.run_tiptoe_experiment(
                embeddings, documents, queries,
                tiptoe_params=tiptoe_params,
                top_k=args.top_k
            )
            print(f"✅ Tiptoe completed with k_clusters={tiptoe_params['k_clusters']}, crypto={args.tiptoe_use_real_crypto}")
        except Exception as e:
            print(f"❌ Tiptoe experiment failed: {e}")

        runner.save_results(results, "single_experiment")

    if args.experiment in ["scalability", "all"]:
        print("Running scalability experiments...")

        # Create parameter dictionaries for each system
        pir_rag_params = {
            'k_clusters': args.pir_rag_k_clusters,
            'cluster_top_k': args.pir_rag_cluster_top_k
        }
        graph_pir_params = {
            'k_neighbors': args.graph_pir_k_neighbors,
            'ef_construction': args.graph_pir_ef_construction,
            'max_connections': args.graph_pir_max_connections,
            'ef_search': args.graph_pir_ef_search,
            'max_iterations': args.graph_pir_max_iterations,
            'parallel': args.graph_pir_parallel,
            'max_neighbors_per_step': args.graph_pir_max_neighbors_per_step
        }
        tiptoe_params = {
            'k_clusters': args.tiptoe_k_clusters,  # Will be overridden in scalability experiment
            'use_real_crypto': args.tiptoe_use_real_crypto,
            'poly_degree': args.tiptoe_poly_degree,
            'plain_modulus': args.tiptoe_plain_modulus
        }

        scalability_results = runner.run_scalability_experiment(
            n_queries=5,
            embed_dim=args.embed_dim,
            embeddings_path=args.embeddings_path,
            corpus_path=args.corpus_path,
            pir_rag_params=pir_rag_params,
            graph_pir_params=graph_pir_params,
            tiptoe_params=tiptoe_params
        )
        runner.save_results(scalability_results, "scalability")

    if args.experiment in ["sensitivity", "all"]:
        print("Running parameter sensitivity experiments...")

        # Use base parameters from arguments for sensitivity analysis
        base_pir_rag_params = {
            'k_clusters': args.pir_rag_k_clusters,
            'cluster_top_k': args.pir_rag_cluster_top_k
        }
        base_graph_pir_params = {
            'k_neighbors': args.graph_pir_k_neighbors,
            'ef_construction': args.graph_pir_ef_construction,
            'max_connections': args.graph_pir_max_connections,
            'ef_search': args.graph_pir_ef_search,
            'max_iterations': args.graph_pir_max_iterations,
            'parallel': args.graph_pir_parallel
        }

        sensitivity_results = runner.run_parameter_sensitivity_experiment(
            n_docs=args.n_docs,
            n_queries=5,
            embeddings_path=args.embeddings_path,
            corpus_path=args.corpus_path,
            base_pir_rag_params=base_pir_rag_params,
            base_graph_pir_params=base_graph_pir_params
        )
        runner.save_results(sensitivity_results, "parameter_sensitivity")

    if args.experiment in ["retrieval", "all"]:
        print("Running retrieval performance experiments...")

        # Prepare graph_params for Graph-PIR
        graph_params = {
            'k_neighbors': args.graph_pir_k_neighbors,
            'ef_construction': args.graph_pir_ef_construction,
            'max_connections': args.graph_pir_max_connections,
            'ef_search': args.graph_pir_ef_search,
            'max_iterations': args.graph_pir_max_iterations,
            'parallel': args.graph_pir_parallel,
            'max_neighbors_per_step': args.graph_pir_max_neighbors_per_step
        }

        retrieval_results = runner.run_retrieval_performance_experiment(
            n_docs=args.n_docs,
            n_queries=args.n_queries,  # Use exact number specified by user
            embeddings_path=args.embeddings_path,
            corpus_path=args.corpus_path,
            embed_dim=args.embed_dim,
            top_k=args.top_k,
            pir_rag_k_clusters=args.pir_rag_k_clusters,
            pir_rag_cluster_top_k=args.pir_rag_cluster_top_k,
            graph_params=graph_params,
            tiptoe_k_clusters=runner.get_default_k_clusters(args.n_docs, args.tiptoe_k_clusters)
        )
        if retrieval_results:
            runner.save_results(retrieval_results, "retrieval_performance")

    print("All experiments completed!")


if __name__ == "__main__":
    main()
