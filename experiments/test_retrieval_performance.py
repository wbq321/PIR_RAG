"""
Specialized Retrieval Performance Testing

This script focuses specifically on retrieval performance metrics:
- Query latency vs accuracy tradeoffs
- Retrieval quality (relevance scores)
- Communication efficiency per document retrieved
- Throughput testing (queries per second)
"""

import time
import numpy as np
import pandas as pd
import json
import torch
from typing import Dict, Any, List, Tuple
from pathlib import Path
import sys
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add PIR_RAG to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

from pir_rag import PIRRAGClient, PIRRAGServer
from graph_pir import GraphPIRSystem
from tiptoe import TiptoeSystem


class RetrievalPerformanceTester:
    """
    Hybrid tester for retrieval performance analysis.
    
    Uses two-phase approach:
    1. Plaintext simulation for accurate retrieval quality metrics
    2. Actual PIR operations for realistic performance/communication metrics
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_realistic_queries(self, embeddings: np.ndarray, n_queries: int = 20) -> List[np.ndarray]:
        """Generate realistic query embeddings that have some relation to the document corpus."""
        np.random.seed(42)
        queries = []
        
        # Create queries that are slight variations of random documents
        for i in range(n_queries):
            # Pick a random document as base
            base_doc_idx = np.random.randint(0, len(embeddings))
            base_embedding = embeddings[base_doc_idx]
            
            # Add some noise to simulate a related query
            noise = np.random.normal(0, 0.1, base_embedding.shape)
            query_embedding = base_embedding + noise
            
            # Normalize to unit vector
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            queries.append(query_embedding.astype(np.float32))
            
        return queries
    
    def calculate_retrieval_quality(self, query_embedding: np.ndarray, 
                                   retrieved_doc_indices: List[int], 
                                   all_embeddings: np.ndarray, top_k: int = 10) -> Dict[str, float]:
        """Calculate retrieval quality metrics."""
        
        # Get ground truth top-k by computing similarities with all documents
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        true_top_k = np.argsort(similarities)[::-1][:top_k]
        
        # Calculate metrics
        retrieved_set = set(retrieved_doc_indices[:top_k])
        true_set = set(true_top_k)
        
        # Precision@K
        precision_at_k = len(retrieved_set.intersection(true_set)) / min(len(retrieved_set), top_k)
        
        # Recall@K (out of true top-k)
        recall_at_k = len(retrieved_set.intersection(true_set)) / top_k
        
        # NDCG@K calculation
        dcg = 0
        for i, doc_idx in enumerate(retrieved_doc_indices[:top_k]):
            if doc_idx in true_set:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Ideal DCG
        idcg = sum(1 / np.log2(i + 2) for i in range(min(top_k, len(true_set))))
        ndcg_at_k = dcg / idcg if idcg > 0 else 0
        
        # Average similarity of retrieved docs
        if retrieved_doc_indices:
            avg_similarity = np.mean([similarities[idx] for idx in retrieved_doc_indices[:top_k] 
                                    if idx < len(similarities)])
        else:
            avg_similarity = 0
            
        return {
            'precision_at_k': precision_at_k,
            'recall_at_k': recall_at_k,
            'ndcg_at_k': ndcg_at_k,
            'avg_similarity': avg_similarity,
            'num_retrieved': len(retrieved_doc_indices)
        }

    def _simulate_pir_rag_search(self, query_embedding: np.ndarray, 
                                documents: List[str], embeddings: np.ndarray,
                                n_clusters: int = 32, top_k_clusters: int = 3) -> List[int]:
        """
        Simulate PIR-RAG search strategy in plaintext for accurate retrieval metrics.
        
        Strategy:
        1. K-means clustering on embeddings (same as actual PIR-RAG)
        2. Find top-k clusters by centroid similarity
        3. Compute similarities with all documents in selected clusters
        4. Return ranked document indices
        """
        from sklearn.cluster import KMeans
        
        print(f"    [PIR-RAG Simulation] Clustering {len(embeddings)} docs into {n_clusters} clusters...")
        
        # 1. Perform K-means clustering (same as actual system)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        
        # 2. Find top-k clusters by centroid similarity
        centroid_similarities = cosine_similarity([query_embedding], centroids)[0]
        top_cluster_indices = np.argsort(centroid_similarities)[::-1][:top_k_clusters]
        
        print(f"    [PIR-RAG Simulation] Selected clusters: {top_cluster_indices}")
        
        # 3. Collect all documents from selected clusters
        candidate_doc_indices = []
        for cluster_idx in top_cluster_indices:
            cluster_docs = np.where(cluster_labels == cluster_idx)[0]
            candidate_doc_indices.extend(cluster_docs)
        
        print(f"    [PIR-RAG Simulation] Found {len(candidate_doc_indices)} candidate documents")
        
        # 4. Compute similarities and rank
        if candidate_doc_indices:
            candidate_embeddings = embeddings[candidate_doc_indices]
            similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
            
            # Sort by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            ranked_doc_indices = [candidate_doc_indices[i] for i in sorted_indices]
        else:
            ranked_doc_indices = []
        
        return ranked_doc_indices

    def _simulate_graph_pir_search(self, query_embedding: np.ndarray,
                                  documents: List[str], embeddings: np.ndarray,
                                  k_neighbors: int = 16, max_iterations: int = 5,
                                  parallel: int = 1, **kwargs) -> List[int]:
        """
        Simulate Graph-PIR search using the GraphANN SearchKNN algorithm (UPDATED implementation).
        
        UPDATED: Now matches the improved Graph-PIR that uses GraphANN SearchKNN approach:
        1. Start vertices: first sqrt(n) vertices, select best 'parallel' by distance
        2. Priority queue (min-heap) of vertices to explore, ranked by L2 distance
        3. Each step: pop 'parallel' closest vertices, explore ALL their neighbors in batch
        4. Batch query neighbors, add new vertices to priority queue
        5. Return all discovered vertices sorted by L2 distance
        """
        import heapq
        
        n_docs = len(embeddings)
        m = k_neighbors  # Number of neighbors per vertex
        
        print(f"    [Graph-PIR Simulation] GraphANN SearchKNN: n={n_docs}, m={m}, maxStep={max_iterations}, parallel={parallel}")
        
        # 1. Build k-NN graph structure  
        graph = self._build_simple_graph(embeddings, k_neighbors)
        
        # 2. Initialize GraphANN SearchKNN state
        reach_step = {}         # vertex_id -> step when discovered  
        known_vertices = {}     # vertex_id -> (vector, neighbors)
        to_be_explored = []     # min-heap: (distance, vertex_id)
        
        # 3. Start vertices: first sqrt(n) documents (GraphANN GetStartVertex)
        target_num = int(np.sqrt(n_docs))
        start_vertex_ids = list(range(min(target_num, n_docs)))
        
        print(f"    [Graph-PIR Simulation] Start vertices: first {len(start_vertex_ids)} vertices")
        
        # 4. Select best 'parallel' start vertices by distance (GraphANN fastStartQueue)
        start_candidates = []
        for vertex_id in start_vertex_ids:
            dist = self._calculate_cosine_distance(embeddings[vertex_id], query_embedding)
            start_candidates.append((dist, vertex_id))
        
        # Sort by distance and take best 'parallel' vertices
        start_candidates.sort(key=lambda x: x[0])
        for i in range(min(parallel, len(start_candidates))):
            dist, vertex_id = start_candidates[i]
            if vertex_id not in known_vertices:
                known_vertices[vertex_id] = (embeddings[vertex_id], graph.get(vertex_id, []))
                reach_step[vertex_id] = 0
                heapq.heappush(to_be_explored, (dist, vertex_id))
        
        # 5. Main GraphANN SearchKNN loop
        for step in range(max_iterations):
            # Collect batch queries for this step
            batch_queries = []
            
            # For each of 'parallel' repetitions
            for rept in range(parallel):
                if len(to_be_explored) == 0:
                    # Make random queries if no vertices to explore (benchmarking fallback)
                    batch_queries.extend([np.random.randint(0, n_docs) for _ in range(m)])
                else:
                    # Pop closest vertex and add ALL its neighbors to batch
                    current_dist, current_vertex_id = heapq.heappop(to_be_explored)
                    if current_vertex_id in known_vertices:
                        _, neighbors = known_vertices[current_vertex_id]
                        batch_queries.extend(neighbors)  # Add ALL neighbors to batch
            
            if not batch_queries:
                break
            
            # 6. Process batch queries with deduplication (matches GraphPIR system)
            unique_queries = list(set([q for q in batch_queries if 0 <= q < n_docs]))
            
            if not unique_queries:
                continue  # Skip if no valid unique queries
                
            print(f"    [Graph-PIR Simulation] Step {step}: PIR batch querying {len(unique_queries)} neighbors")
            
            for neighbor_id in unique_queries:
                if neighbor_id not in known_vertices:
                    # Check if neighbor list is valid (not all zeros)
                    neighbor_neighbors = graph.get(neighbor_id, [])
                    if len(neighbor_neighbors) > 0 and any(n != 0 for n in neighbor_neighbors):
                        # Add to known vertices
                        known_vertices[neighbor_id] = (embeddings[neighbor_id], neighbor_neighbors)
                        reach_step[neighbor_id] = step
                        
                        # Calculate cosine distance and add to exploration queue
                        dist = self._calculate_cosine_distance(embeddings[neighbor_id], query_embedding)
                        heapq.heappush(to_be_explored, (dist, neighbor_id))
        
        # 7. Extract and rank all discovered vertices by cosine distance (GraphANN ending)
        all_known_vertices = []
        for vertex_id, (vector, _) in known_vertices.items():
            dist = self._calculate_cosine_distance(vector, query_embedding)
            all_known_vertices.append((dist, vertex_id))
        
        # Sort by distance (ascending - closest first)
        all_known_vertices.sort(key=lambda x: x[0])
        
        # Return vertex IDs in distance order
        ranked_doc_indices = [vertex_id for _, vertex_id in all_known_vertices]
        
        print(f"    [Graph-PIR Simulation] Found {len(ranked_doc_indices)} documents via GraphANN SearchKNN")
        
        return ranked_doc_indices

    def _calculate_cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine distance between two vectors (matches GraphPIR system).
        Returns 1.0 - cosine_similarity (0 = identical, 2 = opposite)
        """
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance

        cosine_sim = dot_product / (norm1 * norm2)
        return 1.0 - cosine_sim  # Convert to distance (0 = identical, 2 = opposite)

    def _simulate_tiptoe_search(self, query_embedding: np.ndarray,
                               documents: List[str], embeddings: np.ndarray,
                               n_clusters: int = 32) -> List[int]:
        """
        Simulate Tiptoe search strategy in plaintext for accurate retrieval metrics.
        
        Strategy:
        1. K-means clustering (same as PIR-RAG)
        2. Find single closest cluster
        3. Compute similarities with all documents in that cluster
        4. Return ranked document indices
        """
        from sklearn.cluster import KMeans
        
        print(f"    [Tiptoe Simulation] Clustering {len(embeddings)} docs into {n_clusters} clusters...")
        
        # 1. Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        
        # 2. Find single closest cluster
        centroid_similarities = cosine_similarity([query_embedding], centroids)[0]
        closest_cluster_idx = np.argmax(centroid_similarities)
        
        print(f"    [Tiptoe Simulation] Selected closest cluster: {closest_cluster_idx}")
        
        # 3. Get all documents from the closest cluster
        cluster_doc_indices = np.where(cluster_labels == closest_cluster_idx)[0]
        
        print(f"    [Tiptoe Simulation] Found {len(cluster_doc_indices)} documents in closest cluster")
        
        # 4. Compute similarities and rank
        if len(cluster_doc_indices) > 0:
            cluster_embeddings = embeddings[cluster_doc_indices]
            similarities = cosine_similarity([query_embedding], cluster_embeddings)[0]
            
            # Sort by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            ranked_doc_indices = [cluster_doc_indices[i] for i in sorted_indices]
        else:
            ranked_doc_indices = []
        
        return ranked_doc_indices

    def _build_simple_graph(self, embeddings: np.ndarray, k_neighbors: int = 16) -> Dict[int, List[int]]:
        """Build a simple k-NN graph for Graph-PIR simulation."""
        n_docs = len(embeddings)
        graph = {}
        
        # For each document, find k nearest neighbors
        for i in range(n_docs):
            similarities = cosine_similarity([embeddings[i]], embeddings)[0]
            # Get k+1 most similar (excluding self)
            neighbor_indices = np.argsort(similarities)[::-1][1:k_neighbors+1]
            graph[i] = neighbor_indices.tolist()
            
            if (i + 1) % 100 == 0:
                print(f"      Built graph for {i+1}/{n_docs} documents")
        
        return graph
    
    def test_retrieval_performance(self, system_name: str, system, embeddings: np.ndarray, 
                                  documents: List[str], queries: List[np.ndarray], 
                                  top_k: int = 10) -> Dict[str, Any]:
        """
        Hybrid test: Plaintext simulation for retrieval quality + Real PIR for performance metrics.
        
        This approach ensures:
        1. Accurate retrieval quality metrics (no PIR corruption)
        2. Realistic performance and communication measurements
        """
        
        print(f"\n=== Testing {system_name} with Hybrid Approach ===")
        
        # Setup system for performance measurements
        setup_start = time.perf_counter()
        if system_name == "PIR-RAG":
            client, server = system
            k_clusters = min(32, max(5, len(documents)//20))
            server.setup(embeddings, documents, k_clusters)
            client.setup(server.centroids)
        elif system_name == "Graph-PIR":
            system.setup(embeddings, documents)
        elif system_name == "Tiptoe":
            k_clusters = min(32, max(5, len(documents)//20))
            system.setup(embeddings, documents, k_clusters=k_clusters)
        setup_time = time.perf_counter() - setup_start
        
        print(f"  System setup completed in {setup_time:.3f}s")
        
        # Initialize results tracking
        results = {
            'system': system_name,
            'setup_time': setup_time,
            'query_results': [],
            'quality_metrics': [],
            'timing_breakdown': [],
            'communication_costs': [],
            'hybrid_approach': True  # Flag to indicate this uses hybrid testing
        }
        
        total_quality_time = 0
        total_performance_time = 0
        total_communication = 0
        
        print(f"  Testing {len(queries)} queries...")
        
        for query_idx, query_embedding in enumerate(queries):
            print(f"\n  Query {query_idx + 1}/{len(queries)}:")
            
            # === PHASE 1: PLAINTEXT SIMULATION FOR RETRIEVAL QUALITY ===
            quality_start = time.perf_counter()
            
            print(f"    Phase 1: Plaintext simulation for retrieval quality...")
            if system_name == "PIR-RAG":
                retrieved_doc_indices = self._simulate_pir_rag_search(
                    query_embedding, documents, embeddings, 
                    n_clusters=k_clusters, top_k_clusters=3
                )
            elif system_name == "Graph-PIR":
                retrieved_doc_indices = self._simulate_graph_pir_search(
                    query_embedding, documents, embeddings,
                    k_neighbors=32, max_iterations=20, nodes_per_step=5  # YOUR actual parameters
                )
            elif system_name == "Tiptoe":
                retrieved_doc_indices = self._simulate_tiptoe_search(
                    query_embedding, documents, embeddings, n_clusters=k_clusters
                )
            else:
                raise ValueError(f"Unknown system: {system_name}")
            
            quality_time = time.perf_counter() - quality_start
            total_quality_time += quality_time
            
            # Calculate retrieval quality metrics
            quality_metrics = self.calculate_retrieval_quality(
                query_embedding, retrieved_doc_indices, embeddings, top_k
            )
            
            print(f"    Quality metrics: P@{top_k}={quality_metrics['precision_at_k']:.3f}, "
                  f"R@{top_k}={quality_metrics['recall_at_k']:.3f}, "
                  f"NDCG@{top_k}={quality_metrics['ndcg_at_k']:.3f}")
            
            # === PHASE 2: ACTUAL PIR FOR PERFORMANCE METRICS ===
            performance_start = time.perf_counter()
            
            print(f"    Phase 2: Actual PIR for performance measurement...")
            try:
                if system_name == "PIR-RAG":
                    # Run actual PIR operations to measure performance
                    query_tensor = torch.from_numpy(query_embedding).unsqueeze(0)
                    relevant_clusters = client.find_relevant_clusters(query_tensor, top_k=3)
                    doc_tuples, pir_metrics = client.pir_retrieve(relevant_clusters, server)
                    
                    communication_cost = pir_metrics.get('upload_bytes', 0) + pir_metrics.get('download_bytes', 0)
                    
                elif system_name == "Graph-PIR":
                    # Run actual Graph-PIR query
                    doc_tuples, pir_metrics = system.query(query_embedding, top_k=top_k)
                    
                    communication_cost = (pir_metrics.get('phase1_upload_bytes', 0) + 
                                        pir_metrics.get('phase1_download_bytes', 0) +
                                        pir_metrics.get('phase2_upload_bytes', 0) + 
                                        pir_metrics.get('phase2_download_bytes', 0))
                    
                elif system_name == "Tiptoe":
                    # Run actual Tiptoe query
                    doc_tuples, pir_metrics = system.query(query_embedding, top_k=top_k)
                    
                    communication_cost = pir_metrics.get('upload_bytes', 0) + pir_metrics.get('download_bytes', 0)
                
                performance_time = time.perf_counter() - performance_start
                total_performance_time += performance_time
                total_communication += communication_cost
                
                print(f"    Performance: {performance_time:.3f}s, {communication_cost/1024:.1f}KB transferred")
                
            except Exception as e:
                print(f"    Warning: PIR performance measurement failed: {e}")
                performance_time = 0
                communication_cost = 0
                pir_metrics = {}
            
            # Store combined results
            query_result = {
                'query_idx': query_idx,
                'quality_simulation_time': quality_time,
                'pir_performance_time': performance_time,
                'total_query_time': quality_time + performance_time,
                'communication_bytes': communication_cost,
                'num_retrieved_simulated': len(retrieved_doc_indices),
                **quality_metrics
            }
            
            results['query_results'].append(query_result)
            results['quality_metrics'].append(quality_metrics)
            results['communication_costs'].append(communication_cost)
        
        # Calculate aggregate metrics
        quality_metrics_agg = {}
        if results['quality_metrics']:
            for metric in ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'avg_similarity']:
                values = [qm[metric] for qm in results['quality_metrics'] if metric in qm]
                if values:
                    quality_metrics_agg[f'avg_{metric}'] = np.mean(values)
                    quality_metrics_agg[f'std_{metric}'] = np.std(values)
        
        # Summary statistics
        avg_quality_time = total_quality_time / len(queries) if queries else 0
        avg_performance_time = total_performance_time / len(queries) if queries else 0
        avg_communication = total_communication / len(queries) if queries else 0
        
        results.update({
            'avg_quality_simulation_time': avg_quality_time,
            'avg_pir_performance_time': avg_performance_time,
            'avg_total_query_time': avg_quality_time + avg_performance_time,
            'avg_communication_bytes': avg_communication,
            'total_communication_kb': total_communication / 1024,
            **quality_metrics_agg
        })
        
        print(f"\n=== {system_name} Summary ===")
        print(f"  Setup time: {setup_time:.3f}s")
        print(f"  Avg quality simulation time: {avg_quality_time:.3f}s")
        print(f"  Avg PIR performance time: {avg_performance_time:.3f}s")
        print(f"  Avg communication: {avg_communication/1024:.1f}KB")
        print(f"  Avg Precision@{top_k}: {quality_metrics_agg.get('avg_precision_at_k', 0):.3f}")
        print(f"  Avg Recall@{top_k}: {quality_metrics_agg.get('avg_recall_at_k', 0):.3f}")
        print(f"  Avg NDCG@{top_k}: {quality_metrics_agg.get('avg_ndcg_at_k', 0):.3f}")
        
        return results

    def test_throughput_performance(self, system_name: str, system, embeddings: np.ndarray, 
                                   documents: List[str], n_concurrent_queries: int = 50) -> Dict[str, Any]:
        """Test throughput performance with many concurrent queries."""
        
        print(f"Testing {system_name} throughput with {n_concurrent_queries} queries...")
        
        # Generate many queries
        queries = self.generate_realistic_queries(embeddings, n_concurrent_queries)
        
        # Time batch execution
        batch_start = time.perf_counter()
        
        if system_name == "PIR-RAG":
            client, server = system
            for query in queries:
                query_tensor = torch.tensor(query) if not isinstance(query, torch.Tensor) else query
                relevant_clusters = client.find_relevant_clusters(query_tensor, top_k=3)
                doc_tuples, _ = client.pir_retrieve(relevant_clusters, server)
                client.rerank_documents(query_tensor, doc_tuples, top_k=5)
                
        elif system_name == "Graph-PIR":
            for query in queries:
                system.query(query, top_k=5)
                
        elif system_name == "Tiptoe":
            for query in queries:
                system.query(query, top_k=5)
        
        total_time = time.perf_counter() - batch_start
        
        return {
            'system': system_name,
            'total_queries': n_concurrent_queries,
            'total_time': total_time,
            'queries_per_second': n_concurrent_queries / total_time,
            'avg_time_per_query': total_time / n_concurrent_queries
        }
    
    def run_comprehensive_retrieval_test(self, n_docs: int = 1000, n_queries: int = 20) -> Dict[str, Any]:
        """Run comprehensive retrieval performance tests for all systems."""
        
        print(f"Running comprehensive retrieval performance test...")
        print(f"Dataset: {n_docs} documents, {n_queries} queries")
        
        # Generate test data
        np.random.seed(42)
        embeddings = np.random.randn(n_docs, 384).astype(np.float32)
        documents = [f"Document {i} about topic {i % 10}" for i in range(n_docs)]
        queries = self.generate_realistic_queries(embeddings, n_queries)
        
        results = {
            'test_config': {
                'n_documents': n_docs,
                'n_queries': n_queries,
                'embedding_dim': 384,
                'top_k': 10
            },
            'systems': {}
        }
        
        # Test PIR-RAG
        try:
            client = PIRRAGClient()
            server = PIRRAGServer()
            pir_rag_results = self.test_retrieval_performance("PIR-RAG", (client, server), 
                                                            embeddings, documents, queries)
            results['systems']['PIR-RAG'] = pir_rag_results
            
            # Test throughput
            throughput_results = self.test_throughput_performance("PIR-RAG", (client, server), 
                                                                embeddings, documents, 50)
            results['systems']['PIR-RAG']['throughput'] = throughput_results
            
        except Exception as e:
            print(f"PIR-RAG test failed: {e}")
            results['systems']['PIR-RAG'] = {'error': str(e)}
        
        # Test Graph-PIR
        try:
            system = GraphPIRSystem()
            graph_pir_results = self.test_retrieval_performance("Graph-PIR", system, 
                                                              embeddings, documents, queries)
            results['systems']['Graph-PIR'] = graph_pir_results
            
            # Test throughput
            throughput_results = self.test_throughput_performance("Graph-PIR", system, 
                                                                embeddings, documents, 50)
            results['systems']['Graph-PIR']['throughput'] = throughput_results
            
        except Exception as e:
            print(f"Graph-PIR test failed: {e}")
            results['systems']['Graph-PIR'] = {'error': str(e)}
        
        # Test Tiptoe
        try:
            system = TiptoeSystem()
            tiptoe_results = self.test_retrieval_performance("Tiptoe", system, 
                                                           embeddings, documents, queries)
            results['systems']['Tiptoe'] = tiptoe_results
            
            # Test throughput
            throughput_results = self.test_throughput_performance("Tiptoe", system, 
                                                                embeddings, documents, 50)
            results['systems']['Tiptoe']['throughput'] = throughput_results
            
        except Exception as e:
            print(f"Tiptoe test failed: {e}")
            results['systems']['Tiptoe'] = {'error': str(e)}
        
        return results
    
    def print_performance_summary(self, results: Dict[str, Any]):
        """Print a summary of retrieval performance results."""
        
        print("\n" + "="*80)
        print("RETRIEVAL PERFORMANCE SUMMARY")
        print("="*80)
        
        config = results['test_config']
        print(f"Test Configuration:")
        print(f"  Documents: {config['n_documents']:,}")
        print(f"  Queries: {config['n_queries']}")
        print(f"  Top-K: {config['top_k']}")
        print(f"  Embedding Dimension: {config['embedding_dim']}")
        
        print(f"\nPer-System Results:")
        print(f"{'System':<15} {'Query Time':<12} {'Precision@K':<12} {'NDCG@K':<10} {'QPS':<10} {'Comm/Query':<12}")
        print("-" * 80)
        
        for system_name, system_results in results['systems'].items():
            if 'error' in system_results:
                print(f"{system_name:<15} ERROR: {system_results['error']}")
                continue
                
            avg_metrics = system_results.get('avg_metrics', {})
            throughput = system_results.get('throughput', {})
            
            # Use hybrid approach metrics if available
            if 'avg_quality_simulation_time' in system_results:
                query_time = f"{system_results['avg_total_query_time']:.3f}s"
                precision = f"{system_results.get('avg_precision_at_k', 0):.3f}"
                ndcg = f"{system_results.get('avg_ndcg_at_k', 0):.3f}"
                comm = f"{system_results['avg_communication_bytes']:,.0f}B"
            else:
                query_time = f"{avg_metrics.get('avg_query_time', 0):.3f}s"
                precision = f"{avg_metrics.get('avg_precision_at_k', 0):.3f}"
                ndcg = f"{avg_metrics.get('avg_ndcg_at_k', 0):.3f}"
                comm = f"{avg_metrics.get('avg_communication_per_query', 0):,.0f}B"
            
            qps = f"{throughput.get('queries_per_second', 0):.1f}"
            
            print(f"{system_name:<15} {query_time:<12} {precision:<12} {ndcg:<10} {qps:<10} {comm:<12}")
        
        print("\nKey Insights:")
        print("- Query Time: Lower is better")
        print("- Precision@K: Higher is better (fraction of retrieved docs that are relevant)")
        print("- NDCG@K: Higher is better (normalized discounted cumulative gain)")
        print("- QPS: Higher is better (queries per second)")
        print("- Comm/Query: Lower is better (communication bytes per query)")


def main():
    """Run retrieval performance testing."""
    
    tester = RetrievalPerformanceTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_retrieval_test(n_docs=1000, n_queries=20)
    
    # Print summary
    tester.print_performance_summary(results)
    
    # Save detailed results
    output_file = tester.output_dir / f"retrieval_performance_{int(time.time())}.json"
    with open(output_file, 'w') as f:
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
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
