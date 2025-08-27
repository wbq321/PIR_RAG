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
    """Specialized tester for retrieval performance analysis."""
    
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
    
    def test_retrieval_performance(self, system_name: str, system, embeddings: np.ndarray, 
                                  documents: List[str], queries: List[np.ndarray], 
                                  top_k: int = 10) -> Dict[str, Any]:
        """Test retrieval performance for a single system."""
        
        print(f"Testing {system_name} retrieval performance...")
        
        # Setup system
        setup_start = time.perf_counter()
        if system_name == "PIR-RAG":
            client, server = system
            # Server does clustering first
            k_clusters = min(5, len(documents)//20)
            server.setup(embeddings, documents, k_clusters)
            # Client gets centroids from server
            client.setup(server.centroids)
        elif system_name == "Graph-PIR":
            system.setup(embeddings, documents)
        elif system_name == "Tiptoe":
            system.setup(embeddings, documents, k_clusters=min(5, len(documents)//20))
        setup_time = time.perf_counter() - setup_start
        
        # Test queries
        results = {
            'system': system_name,
            'setup_time': setup_time,
            'query_results': [],
            'quality_metrics': [],
            'timing_breakdown': [],
            'communication_costs': []
        }
        
        total_query_time = 0
        total_communication = 0
        
        for i, query_embedding in enumerate(queries):
            print(f"  Processing query {i+1}/{len(queries)}")
            
            # Execute query with timing
            query_start = time.perf_counter()
            
            if system_name == "PIR-RAG":
                # Step-by-step PIR-RAG execution
                cluster_start = time.perf_counter()
                query_tensor = torch.tensor(query_embedding) if not isinstance(query_embedding, torch.Tensor) else query_embedding
                relevant_clusters = client.find_relevant_clusters(query_tensor, top_k=3)
                cluster_time = time.perf_counter() - cluster_start
                
                pir_start = time.perf_counter()
                doc_tuples, pir_metrics = client.pir_retrieve(relevant_clusters, server)
                pir_time = time.perf_counter() - pir_start
                
                rerank_start = time.perf_counter()
                final_results = client.rerank_documents(query_tensor, doc_tuples, top_k=top_k)
                rerank_time = time.perf_counter() - rerank_start
                
                # Extract document indices from URLs (assuming format https://example.com/doc_{idx})
                retrieved_indices = []
                for url in final_results:
                    try:
                        idx = int(url.split('_')[-1])
                        retrieved_indices.append(idx)
                    except:
                        pass
                
                timing = {
                    'cluster_selection': cluster_time,
                    'pir_retrieval': pir_time,
                    'reranking': rerank_time,
                    'total': time.perf_counter() - query_start
                }
                
                comm_cost = pir_metrics.get('total_upload_bytes', 0) + pir_metrics.get('total_download_bytes', 0)
                
            elif system_name == "Graph-PIR":
                urls, metrics = system.query(query_embedding, top_k=top_k)
                
                # Extract document indices
                retrieved_indices = []
                for url in urls:
                    try:
                        idx = int(url.split('_')[-1])
                        retrieved_indices.append(idx)
                    except:
                        pass
                
                timing = {
                    'phase1': metrics.get('phase1_time', 0),
                    'phase2': metrics.get('phase2_time', 0),
                    'total': time.perf_counter() - query_start
                }
                
                comm_cost = (metrics.get('phase1_upload_bytes', 0) + metrics.get('phase1_download_bytes', 0) +
                           metrics.get('phase2_upload_bytes', 0) + metrics.get('phase2_download_bytes', 0))
                
            elif system_name == "Tiptoe":
                urls, metrics = system.query(query_embedding, top_k=top_k)
                
                # Extract document indices
                retrieved_indices = []
                for url in urls:
                    try:
                        idx = int(url.split('_')[-1])
                        retrieved_indices.append(idx)
                    except:
                        pass
                
                timing = {
                    'ranking': metrics.get('ranking_time', 0),
                    'retrieval': metrics.get('retrieval_time', 0),
                    'total': time.perf_counter() - query_start
                }
                
                comm_cost = metrics.get('total_upload_bytes', 0) + metrics.get('total_download_bytes', 0)
            
            query_time = time.perf_counter() - query_start
            total_query_time += query_time
            total_communication += comm_cost
            
            # Calculate retrieval quality
            quality = self.calculate_retrieval_quality(query_embedding, retrieved_indices, embeddings, top_k)
            
            results['query_results'].append({
                'query_id': i,
                'retrieved_indices': retrieved_indices,
                'query_time': query_time
            })
            results['quality_metrics'].append(quality)
            results['timing_breakdown'].append(timing)
            results['communication_costs'].append(comm_cost)
        
        # Aggregate metrics
        quality_metrics = results['quality_metrics']
        results['avg_metrics'] = {
            'avg_query_time': total_query_time / len(queries),
            'avg_precision_at_k': np.mean([q['precision_at_k'] for q in quality_metrics]),
            'avg_recall_at_k': np.mean([q['recall_at_k'] for q in quality_metrics]),
            'avg_ndcg_at_k': np.mean([q['ndcg_at_k'] for q in quality_metrics]),
            'avg_similarity': np.mean([q['avg_similarity'] for q in quality_metrics]),
            'avg_communication_per_query': total_communication / len(queries),
            'queries_per_second': len(queries) / total_query_time,
            'communication_efficiency': total_communication / (len(queries) * top_k)  # bytes per retrieved doc
        }
        
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
                
            metrics = system_results['avg_metrics']
            throughput = system_results.get('throughput', {})
            
            query_time = f"{metrics['avg_query_time']:.3f}s"
            precision = f"{metrics['avg_precision_at_k']:.3f}"
            ndcg = f"{metrics['avg_ndcg_at_k']:.3f}"
            qps = f"{throughput.get('queries_per_second', 0):.1f}"
            comm = f"{metrics['avg_communication_per_query']:,.0f}B"
            
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
