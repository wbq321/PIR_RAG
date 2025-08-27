#!/usr/bin/env python3
"""
PIR-RAG Experiment Runner

A modular and configurable experiment script for evaluating the PIR-RAG system.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pir_rag import PIRRAGClient, PIRRAGServer
from pir_rag.utils import prepare_docs_by_size, prepare_evaluation_data


class PIRRAGExperimentRunner:
    """Main experiment runner for PIR-RAG system."""
    
    def __init__(self, embeddings_path: str, corpus_path: str, model_path: str = "../shared_models/bge_model"):
        """
        Initialize the experiment runner.
        
        Args:
            embeddings_path: Path to the embeddings .npy file
            corpus_path: Path to the corpus .csv file
            model_path: Path to the local sentence transformer model
        """
        self.embeddings_path = embeddings_path
        self.corpus_path = corpus_path
        self.model_path = model_path
        
        # Load data and model
        self._load_data()
        self._load_model()
    
    def _load_data(self):
        """Load embeddings and corpus data."""
        print("--- Loading Data ---")
        self.full_embeddings = np.load(self.embeddings_path)
        self.full_corpus_df = pd.read_csv(self.corpus_path)
        self.full_corpus_texts = self.full_corpus_df['text'].dropna().tolist()

        if len(self.full_embeddings) != len(self.full_corpus_texts):
            raise ValueError("Embedding and corpus file length mismatch.")
        
        print(f"Loaded {len(self.full_corpus_texts)} documents with embeddings")
    
    def _load_model(self):
        """Load the sentence transformer model."""
        print("--- Loading Model ---")
        self.model = SentenceTransformer(self.model_path, device='cpu')
        print(f"Model loaded from {self.model_path}")
    
    def run_single_experiment(self, config: dict) -> dict:
        """
        Run a single experiment configuration.
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\\n--- Running Experiment: {config.get('name', 'Unnamed')} ---")
        
        # Prepare data subset if needed
        if 'doc_size_filter' in config:
            corpus_texts = prepare_docs_by_size(
                self.full_corpus_texts,
                config['doc_size_filter']['target_bytes'],
                config['doc_size_filter']['tolerance'],
                config['doc_size_filter']['n_docs']
            )
            # Find indices and filter embeddings
            indices = [self.full_corpus_texts.index(doc) for doc in corpus_texts]
            embeddings = self.full_embeddings[indices]
        else:
            corpus_texts = self.full_corpus_texts
            embeddings = self.full_embeddings
        
        # Prepare evaluation data
        eval_queries, eval_ground_truth = prepare_evaluation_data(
            corpus_texts, embeddings, config.get('n_eval_queries', 100)
        )
        
        # Set up server
        server = PIRRAGServer()
        server_stats = server.setup(
            embeddings, corpus_texts, config['n_clusters']
        )
        
        # Set up client
        client = PIRRAGClient()
        client_stats = client.setup(
            server.centroids, config.get('key_length', 2048)
        )
        
        # Run queries
        latencies = []
        all_metrics = []
        hits_at_10 = 0
        
        print(f"  -> Running {len(eval_queries)} queries...")
        for qid, query_data in tqdm(eval_queries.items(), desc="  Querying", leave=False):
            query_embedding = query_data["embedding"]
            
            query_start_time = time.perf_counter()
            
            # 1. Find relevant clusters
            cluster_indices = client.find_relevant_clusters(
                query_embedding, config.get('top_n_clusters', 3)
            )
            
            # 2. PIR retrieval (now returns document tuples with embeddings)
            retrieved_docs, pir_metrics = client.pir_retrieve(cluster_indices, server)
            
            # 3. Re-ranking (using embeddings from PIR, no additional server request)
            final_docs = client.rerank_documents(
                query_embedding, retrieved_docs, top_k=10
            )
            
            query_latency = time.perf_counter() - query_start_time
            
            # Check accuracy
            if eval_ground_truth[qid] in set(final_docs):
                hits_at_10 += 1
            
            latencies.append(query_latency)
            all_metrics.append(pir_metrics)
        
        # Aggregate metrics
        avg_metrics = {
            "query_latency_s": np.mean(latencies),
            "query_gen_time_s": np.mean([m["total_query_gen_time"] for m in all_metrics]),
            "server_time_s": np.mean([m["total_server_time"] for m in all_metrics]),
            "decode_time_s": np.mean([m["total_decode_time"] for m in all_metrics]),
            "upload_kb": np.mean([m["total_upload_bytes"] for m in all_metrics]) / 1024,
            "download_kb": np.mean([m["total_download_bytes"] for m in all_metrics]) / 1024,
        }
        
        # Compile final results
        result = {
            **config,
            "n_docs": len(corpus_texts),
            "recall_at_10": hits_at_10 / len(eval_queries),
            "server_setup_s": server_stats["setup_time"],
            "client_setup_s": client_stats["setup_time"],
            **avg_metrics
        }
        
        return result
    
    def run_document_size_experiments(self) -> list:
        """Run experiments varying document sizes."""
        print("\\n" + "="*50)
        print("Document Size Impact Experiments")
        print("="*50)
        
        results = []
        document_sizes = [1500, 2500, 3500, 4500]
        
        for doc_size in document_sizes:
            config = {
                "name": f"DocSize_{doc_size}",
                "experiment_type": "document_size",
                "doc_size_filter": {
                    "target_bytes": doc_size,
                    "tolerance": 0.5,
                    "n_docs": 100
                },
                "n_clusters": 10,
                "key_length": 1024,
                "top_n_clusters": 1,
                "n_eval_queries": 10,
                "target_doc_size": doc_size
            }
            
            try:
                result = self.run_single_experiment(config)
                results.append(result)
            except ValueError as e:
                print(f"  -> Skipping doc size {doc_size}: {e}")
                continue
        
        return results
    
    def run_cluster_experiments(self) -> list:
        """Run experiments varying number of clusters."""
        print("\\n" + "="*50)
        print("Cluster Count Impact Experiments")
        print("="*50)
        
        results = []
        cluster_counts = [100, 500, 1000, 5000]
        
        for n_clusters in cluster_counts:
            config = {
                "name": f"Clusters_{n_clusters}",
                "experiment_type": "cluster_count",
                "n_clusters": n_clusters,
                "key_length": 2048,
                "top_n_clusters": 3,
                "n_eval_queries": 50
            }
            
            result = self.run_single_experiment(config)
            results.append(result)
        
        return results
    
    def run_top_k_experiments(self) -> list:
        """Run experiments varying top-k clusters retrieved."""
        print("\\n" + "="*50)
        print("Top-K Clusters Impact Experiments")
        print("="*50)
        
        results = []
        top_k_values = [1, 2, 3, 5, 10]
        
        for top_k in top_k_values:
            config = {
                "name": f"TopK_{top_k}",
                "experiment_type": "top_k_clusters",
                "n_clusters": 1000,
                "key_length": 2048,
                "top_n_clusters": top_k,
                "n_eval_queries": 50
            }
            
            result = self.run_single_experiment(config)
            results.append(result)
        
        return results
    
    def run_key_length_experiments(self) -> list:
        """Run experiments varying encryption key length."""
        print("\\n" + "="*50)
        print("Key Length Impact Experiments")
        print("="*50)
        
        results = []
        key_lengths = [1024, 2048, 3072]
        
        for key_length in key_lengths:
            config = {
                "name": f"KeyLen_{key_length}",
                "experiment_type": "key_length",
                "n_clusters": 1000,
                "key_length": key_length,
                "top_n_clusters": 3,
                "n_eval_queries": 50
            }
            
            result = self.run_single_experiment(config)
            results.append(result)
        
        return results
    
    def run_all_experiments(self) -> pd.DataFrame:
        """Run all experiments and return results."""
        all_results = []
        
        # Run different experiment types
        all_results.extend(self.run_document_size_experiments())
        # Uncomment the following lines to run additional experiments
        # all_results.extend(self.run_cluster_experiments())
        # all_results.extend(self.run_top_k_experiments())
        # all_results.extend(self.run_key_length_experiments())
        
        return pd.DataFrame(all_results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run PIR-RAG experiments")
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to embeddings .npy file")
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="Path to corpus .csv file")
    parser.add_argument("--model_path", type=str, default="../shared_models/bge_model",
                        help="Path to local sentence transformer model")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    runner = PIRRAGExperimentRunner(
        args.embeddings_path,
        args.corpus_path,
        args.model_path
    )
    
    results_df = runner.run_all_experiments()
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"pir_rag_results_{timestamp}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    
    print("\\n\\n--- EXPERIMENT RESULTS ---")
    print(results_df.to_string())
    print(f"\\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
