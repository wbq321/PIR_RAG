"""
Communication Efficiency Comparison: PIR-RAG vs Graph-PIR

This script provides a comprehensive comparison of communication costs
between the two private search approaches.
"""

import numpy as np
import pandas as pd
import time
import json
from typing import Dict, List, Any, Tuple
import argparse
import os
import sys
from pathlib import Path

# Add the PIR_RAG source to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from pir_rag import PIRRAGClient, PIRRAGServer
from graph_pir import GraphPIRSystem
from sentence_transformers import SentenceTransformer


class CommunicationComparison:
    """Compare communication efficiency between PIR-RAG and Graph-PIR systems"""
    
    def __init__(self, embeddings: np.ndarray, documents: List[str], model):
        self.embeddings = embeddings
        self.documents = documents
        self.model = model
        self.results = []
        
    def setup_systems(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Setup both PIR-RAG and Graph-PIR systems"""
        print("Setting up comparison systems...")
        
        # Setup PIR-RAG
        print("\n=== Setting up PIR-RAG ===")
        pir_rag_server = PIRRAGServer()
        pir_rag_setup = pir_rag_server.setup(
            embeddings=self.embeddings,
            documents_text=self.documents,
            n_clusters=config["pir_rag"]["n_clusters"]
        )
        
        pir_rag_client = PIRRAGClient()
        pir_rag_client_setup = pir_rag_client.setup(
            centroids=pir_rag_server.centroids,
            key_length=config["pir_rag"]["key_length"]
        )
        
        # Setup Graph-PIR
        print("\n=== Setting up Graph-PIR ===")
        graph_pir_system = GraphPIRSystem(private_mode=True)
        graph_pir_setup = graph_pir_system.setup(
            embeddings=self.embeddings,
            documents_text=self.documents,
            neighbor_count=config["graph_pir"]["neighbor_count"]
        )
        
        systems = {
            "pir_rag": {
                "server": pir_rag_server,
                "client": pir_rag_client,
                "setup_stats": {**pir_rag_setup, **pir_rag_client_setup}
            },
            "graph_pir": {
                "system": graph_pir_system,
                "setup_stats": graph_pir_setup
            }
        }
        
        return systems, {
            "pir_rag_setup": {**pir_rag_setup, **pir_rag_client_setup},
            "graph_pir_setup": graph_pir_setup
        }
    
    def run_pir_rag_query(self, systems: Dict, query_text: str, top_k: int) -> Dict[str, Any]:
        """Run a single query on PIR-RAG system"""
        server = systems["pir_rag"]["server"]
        client = systems["pir_rag"]["client"]
        
        # Encode query
        query_start = time.perf_counter()
        query_embedding = self.model.encode([query_text])
        
        # Find relevant clusters
        cluster_indices = client.find_relevant_clusters(
            query_embedding=query_embedding,
            top_k=min(5, len(server.centroids))  # Query top clusters
        )
        
        # Perform PIR retrieval
        retrieved_docs, metrics = client.pir_retrieve(server, cluster_indices)
        
        # Re-rank results
        if retrieved_docs:
            final_docs = client.rerank_documents(
                query_embedding=query_embedding,
                documents=retrieved_docs,
                model=self.model,
                top_k=top_k
            )
        else:
            final_docs = []
        
        query_time = time.perf_counter() - query_start
        
        return {
            "system": "pir_rag",
            "query_time": query_time,
            "results_count": len(final_docs),
            "clusters_queried": len(cluster_indices),
            "total_upload_bytes": metrics["total_upload_bytes"],
            "total_download_bytes": metrics["total_download_bytes"],
            "total_communication_bytes": metrics["total_upload_bytes"] + metrics["total_download_bytes"],
            "query_gen_time": metrics["total_query_gen_time"],
            "server_time": metrics["total_server_time"],
            "decode_time": metrics["total_decode_time"]
        }
    
    def run_graph_pir_query(self, systems: Dict, query_text: str, top_k: int, 
                           max_steps: int = 15, parallel_exploration: int = 2) -> Dict[str, Any]:
        """Run a single query on Graph-PIR system"""
        graph_system = systems["graph_pir"]["system"]
        
        # Encode query
        query_start = time.perf_counter()
        query_embedding = self.model.encode([query_text])
        
        # Perform graph search
        doc_indices, metrics = graph_system.search(
            query_embedding=query_embedding.flatten(),
            top_k=top_k,
            max_steps=max_steps,
            parallel_exploration=parallel_exploration
        )
        
        query_time = time.perf_counter() - query_start
        
        return {
            "system": "graph_pir",
            "query_time": query_time,
            "results_count": len(doc_indices),
            "vertex_accesses": metrics["vertex_accesses"],
            "steps_taken": metrics["steps_taken"],
            "total_communication_bytes": metrics["total_communication_bytes"],
            "search_time": metrics["search_time"],
            "max_steps": max_steps,
            "parallel_exploration": parallel_exploration
        }
    
    def run_comparison_experiment(self, queries: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive comparison experiment"""
        print(f"\n=== Starting Communication Comparison Experiment ===")
        print(f"Number of queries: {len(queries)}")
        print(f"Dataset size: {len(self.documents)} documents")
        
        # Setup systems
        systems, setup_stats = self.setup_systems(config)
        
        # Run queries on both systems
        pir_rag_results = []
        graph_pir_results = []
        
        print(f"\n=== Running Queries ===")
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}/{len(queries)}: {query[:50]}...")
            
            # PIR-RAG query
            print("  -> Running PIR-RAG query...")
            pir_rag_result = self.run_pir_rag_query(
                systems, query, config["top_k"]
            )
            pir_rag_results.append(pir_rag_result)
            
            # Graph-PIR query
            print("  -> Running Graph-PIR query...")
            graph_pir_result = self.run_graph_pir_query(
                systems, query, config["top_k"],
                config["graph_pir"]["max_steps"],
                config["graph_pir"]["parallel_exploration"]
            )
            graph_pir_results.append(graph_pir_result)
            
            # Print intermediate results
            print(f"     PIR-RAG: {pir_rag_result['total_communication_bytes']:,} bytes")
            print(f"     Graph-PIR: {graph_pir_result['total_communication_bytes']:,} bytes")
        
        # Aggregate results
        comparison_results = self._aggregate_results(
            pir_rag_results, graph_pir_results, setup_stats, config
        )
        
        return comparison_results
    
    def _aggregate_results(self, pir_rag_results: List[Dict], graph_pir_results: List[Dict],
                          setup_stats: Dict, config: Dict) -> Dict[str, Any]:
        """Aggregate and analyze comparison results"""
        
        # Calculate averages for PIR-RAG
        pir_rag_avg = {
            "avg_communication_bytes": np.mean([r["total_communication_bytes"] for r in pir_rag_results]),
            "avg_upload_bytes": np.mean([r["total_upload_bytes"] for r in pir_rag_results]),
            "avg_download_bytes": np.mean([r["total_download_bytes"] for r in pir_rag_results]),
            "avg_query_time": np.mean([r["query_time"] for r in pir_rag_results]),
            "avg_clusters_queried": np.mean([r["clusters_queried"] for r in pir_rag_results]),
            "total_communication_bytes": sum([r["total_communication_bytes"] for r in pir_rag_results])
        }
        
        # Calculate averages for Graph-PIR
        graph_pir_avg = {
            "avg_communication_bytes": np.mean([r["total_communication_bytes"] for r in graph_pir_results]),
            "avg_query_time": np.mean([r["query_time"] for r in graph_pir_results]),
            "avg_vertex_accesses": np.mean([r["vertex_accesses"] for r in graph_pir_results]),
            "avg_steps_taken": np.mean([r["steps_taken"] for r in graph_pir_results]),
            "total_communication_bytes": sum([r["total_communication_bytes"] for r in graph_pir_results])
        }
        
        # Communication efficiency comparison
        efficiency_ratio = pir_rag_avg["avg_communication_bytes"] / max(1, graph_pir_avg["avg_communication_bytes"])
        
        results = {
            "experiment_config": config,
            "setup_stats": setup_stats,
            "query_count": len(pir_rag_results),
            "dataset_size": len(self.documents),
            
            "pir_rag_results": {
                "individual_queries": pir_rag_results,
                "aggregated": pir_rag_avg
            },
            
            "graph_pir_results": {
                "individual_queries": graph_pir_results,
                "aggregated": graph_pir_avg
            },
            
            "comparison": {
                "communication_efficiency_ratio": efficiency_ratio,  # PIR-RAG / Graph-PIR
                "pir_rag_total_bytes": pir_rag_avg["total_communication_bytes"],
                "graph_pir_total_bytes": graph_pir_avg["total_communication_bytes"],
                "communication_savings": graph_pir_avg["total_communication_bytes"] - pir_rag_avg["total_communication_bytes"],
                "relative_savings": 1 - (graph_pir_avg["avg_communication_bytes"] / pir_rag_avg["avg_communication_bytes"]),
                
                "performance_comparison": {
                    "pir_rag_avg_query_time": pir_rag_avg["avg_query_time"],
                    "graph_pir_avg_query_time": graph_pir_avg["avg_query_time"],
                    "query_time_ratio": pir_rag_avg["avg_query_time"] / max(1e-6, graph_pir_avg["avg_query_time"])
                }
            }
        }
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of comparison results"""
        print(f"\n{'='*60}")
        print(f"COMMUNICATION EFFICIENCY COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nDataset: {results['dataset_size']} documents")
        print(f"Queries: {results['query_count']}")
        
        pir_rag = results["pir_rag_results"]["aggregated"]
        graph_pir = results["graph_pir_results"]["aggregated"]
        comparison = results["comparison"]
        
        print(f"\n--- PIR-RAG Results ---")
        print(f"Avg Communication: {pir_rag['avg_communication_bytes']:,.0f} bytes")
        print(f"Avg Upload: {pir_rag['avg_upload_bytes']:,.0f} bytes")
        print(f"Avg Download: {pir_rag['avg_download_bytes']:,.0f} bytes")
        print(f"Avg Query Time: {pir_rag['avg_query_time']:.3f}s")
        print(f"Avg Clusters Queried: {pir_rag['avg_clusters_queried']:.1f}")
        
        print(f"\n--- Graph-PIR Results ---")
        print(f"Avg Communication: {graph_pir['avg_communication_bytes']:,.0f} bytes")
        print(f"Avg Query Time: {graph_pir['avg_query_time']:.3f}s")
        print(f"Avg Vertex Accesses: {graph_pir['avg_vertex_accesses']:.1f}")
        print(f"Avg Steps: {graph_pir['avg_steps_taken']:.1f}")
        
        print(f"\n--- Comparison ---")
        print(f"Communication Efficiency Ratio: {comparison['communication_efficiency_ratio']:.2f}")
        print(f"Relative Communication Savings: {comparison['relative_savings']:.1%}")
        
        if comparison['relative_savings'] > 0:
            winner = "Graph-PIR"
            savings = comparison['relative_savings']
        else:
            winner = "PIR-RAG"
            savings = -comparison['relative_savings']
        
        print(f"Winner: {winner} (saves {savings:.1%} communication)")
        
        print(f"\nQuery Time Ratio: {comparison['performance_comparison']['query_time_ratio']:.2f}")
        if comparison['performance_comparison']['query_time_ratio'] > 1:
            print("PIR-RAG is slower")
        else:
            print("Graph-PIR is slower")


def main():
    parser = argparse.ArgumentParser(description="Compare communication efficiency of PIR-RAG vs Graph-PIR")
    parser.add_argument("--embeddings_path", required=True, help="Path to embeddings file")
    parser.add_argument("--corpus_path", required=True, help="Path to corpus CSV file")
    parser.add_argument("--model_path", required=True, help="Path to sentence transformer model")
    parser.add_argument("--output_path", default="communication_comparison_results.json", help="Output file path")
    parser.add_argument("--n_queries", type=int, default=20, help="Number of test queries")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to retrieve")
    parser.add_argument("--max_docs", type=int, default=None, help="Maximum number of documents to use (for testing with smaller datasets)")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    embeddings = np.load(args.embeddings_path)
    corpus_df = pd.read_csv(args.corpus_path)
    documents = corpus_df['text'].tolist()
    
    # Limit dataset size if specified
    if args.max_docs is not None:
        max_docs = min(args.max_docs, len(documents), len(embeddings))
        embeddings = embeddings[:max_docs]
        documents = documents[:max_docs]
        print(f"Limited dataset to {max_docs} documents for testing")
    
    print(f"Using {len(embeddings)} embeddings and {len(documents)} documents")
    
    # Load model
    print("Loading model...")
    model = SentenceTransformer(args.model_path)
    
    # Configuration
    config = {
        "top_k": args.top_k,
        "pir_rag": {
            "n_clusters": min(1000, len(documents) // 10),
            "key_length": 2048
        },
        "graph_pir": {
            "neighbor_count": 32,
            "max_steps": 15,
            "parallel_exploration": 2
        }
    }
    
    print(f"Configuration: {config}")
    
    # Generate test queries
    print(f"Generating {args.n_queries} test queries...")
    random_indices = np.random.choice(len(documents), args.n_queries, replace=False)
    test_queries = [documents[i][:100] + "..." for i in random_indices]  # Use document prefixes as queries
    
    # Run comparison
    comparison = CommunicationComparison(embeddings[:len(documents)], documents, model)
    results = comparison.run_comparison_experiment(test_queries, config)
    
    # Print summary
    comparison.print_summary(results)
    
    # Save results
    print(f"\nSaving results to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Comparison complete!")


if __name__ == "__main__":
    main()
