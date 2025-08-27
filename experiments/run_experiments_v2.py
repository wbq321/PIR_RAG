#!/usr/bin/env python3
"""
PIR-RAG Experiment Runner (Updated Version)

A modular and configurable experiment script for evaluating the PIR-RAG system
with support for YAML configuration and pre-processed size groups.
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pir_rag import PIRRAGClient, PIRRAGServer
from pir_rag.utils import prepare_evaluation_data


class PIRRAGExperimentRunner:
    """Main experiment runner for PIR-RAG system with YAML configuration support."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the experiment runner.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or "config.yaml"
        self.config = self._load_config()
        
        # Load model
        self._load_model()
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded configuration from {self.config_path}")
        return config
    
    def _load_model(self):
        """Load the sentence transformer model."""
        print("--- Loading Model ---")
        model_path = self.config['data']['model_path']
        self.model = SentenceTransformer(model_path, device='cpu')
        print(f"Model loaded from {model_path}")
    
    def _load_size_group_data(self, group_name: str, n_docs: int = None) -> tuple:
        """
        Load data from a specific size group.
        
        Args:
            group_name: Name of the size group (small, medium, etc.)
            n_docs: Number of documents to sample (None for all)
            
        Returns:
            Tuple of (corpus_texts, embeddings)
        """
        size_groups_config = self.config['data']['size_groups']
        
        if group_name not in size_groups_config:
            raise ValueError(f"Size group '{group_name}' not found in configuration")
        
        # Load corpus data
        corpus_path = size_groups_config[group_name]
        corpus_df = pd.read_csv(corpus_path)
        corpus_texts = corpus_df['text'].dropna().tolist()
        
        # Load embeddings
        embedding_path = corpus_path.replace('.csv', '_embeddings.npy')
        embeddings = np.load(embedding_path)
        
        if len(corpus_texts) != len(embeddings):
            raise ValueError(f"Text and embedding length mismatch for group {group_name}")
        
        # Sample if requested
        if n_docs is not None and n_docs < len(corpus_texts):
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(corpus_texts), n_docs, replace=False)
            corpus_texts = [corpus_texts[i] for i in indices]
            embeddings = embeddings[indices]
        
        print(f"Loaded {len(corpus_texts)} documents from size group '{group_name}'")
        return corpus_texts, embeddings
    
    def _load_full_dataset(self) -> tuple:
        """Load the full dataset (for experiments not using size groups)."""
        print("--- Loading Full Dataset ---")
        embeddings_path = self.config['data']['embeddings_path']
        corpus_path = self.config['data']['corpus_path']
        
        embeddings = np.load(embeddings_path)
        corpus_df = pd.read_csv(corpus_path)
        corpus_texts = corpus_df['text'].dropna().tolist()

        if len(embeddings) != len(corpus_texts):
            raise ValueError("Embedding and corpus file length mismatch.")
        
        print(f"Loaded {len(corpus_texts)} documents with embeddings")
        return corpus_texts, embeddings
    
    def run_single_experiment(self, config: dict, corpus_texts: list, embeddings: np.ndarray) -> dict:
        """
        Run a single experiment configuration.
        
        Args:
            config: Experiment configuration dictionary
            corpus_texts: List of document texts
            embeddings: Document embeddings array
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n--- Running: {config.get('name', 'Unnamed Experiment')} ---")
        print(f"Documents: {len(corpus_texts)}, Clusters: {config['n_clusters']}")
        
        # Prepare evaluation data
        eval_queries, eval_ground_truth = prepare_evaluation_data(
            corpus_texts, embeddings, config.get('n_eval_queries', 10)
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
        
        # Calculate average document size
        avg_doc_size = np.mean([len(text.encode('utf-8')) for text in corpus_texts])
        
        # Compile final results
        result = {
            **config,
            "n_docs": len(corpus_texts),
            "avg_doc_size_bytes": avg_doc_size,
            "recall_at_10": hits_at_10 / len(eval_queries),
            "server_setup_s": server_stats["setup_time"],
            "client_setup_s": client_stats["setup_time"],
            **avg_metrics
        }
        
        return result
    
    def run_document_size_experiments(self) -> list:
        """Run experiments varying document sizes using pre-processed groups."""
        print("\n" + "="*60)
        print("Document Size Impact Experiments (Using Size Groups)")
        print("="*60)
        
        exp_config = self.config['experiments']['document_size']
        if not exp_config.get('enabled', False):
            print("Document size experiments are disabled in config.")
            return []
        
        results = []
        size_groups = exp_config['size_groups']
        n_docs_per_group = exp_config['n_docs_per_group']
        
        for group_name in size_groups:
            try:
                # Load data for this size group
                corpus_texts, embeddings = self._load_size_group_data(group_name, n_docs_per_group)
                
                # Configure experiment
                config = {
                    "name": f"DocSize_{group_name}",
                    "experiment_type": "document_size",
                    "size_group": group_name,
                    "n_eval_queries": exp_config['n_eval_queries'],
                    **exp_config['fixed_params']
                }
                
                result = self.run_single_experiment(config, corpus_texts, embeddings)
                results.append(result)
                
            except Exception as e:
                print(f"  -> Error with size group '{group_name}': {e}")
                continue
        
        return results
    
    def run_document_count_experiments(self) -> list:
        """Run experiments varying document counts within same size group."""
        print("\n" + "="*60)
        print("Document Count Impact Experiments")
        print("="*60)
        
        exp_config = self.config['experiments']['document_count']
        if not exp_config.get('enabled', False):
            print("Document count experiments are disabled in config.")
            return []
        
        results = []
        target_group = exp_config['target_size_group']
        doc_counts = exp_config['doc_counts']
        
        # Load full data for the target size group
        corpus_texts_full, embeddings_full = self._load_size_group_data(target_group, n_docs=None)
        
        print(f"Using size group '{target_group}' with {len(corpus_texts_full)} total documents")
        
        for doc_count in doc_counts:
            if doc_count > len(corpus_texts_full):
                print(f"  -> Skipping doc_count {doc_count}: exceeds available documents ({len(corpus_texts_full)})")
                continue
            
            try:
                # Sample documents for this count
                np.random.seed(42)  # For reproducibility
                indices = np.random.choice(len(corpus_texts_full), doc_count, replace=False)
                corpus_texts = [corpus_texts_full[i] for i in indices]
                embeddings = embeddings_full[indices]
                
                # Configure experiment
                config = {
                    "name": f"DocCount_{doc_count}",
                    "experiment_type": "document_count",
                    "size_group": target_group,
                    "doc_count": doc_count,
                    "n_eval_queries": exp_config['n_eval_queries'],
                    **exp_config['fixed_params']
                }
                
                # Adjust cluster count based on document count (optional)
                if 'n_clusters' in config and config['n_clusters'] > doc_count:
                    config['n_clusters'] = max(doc_count // 10, 5)  # Heuristic: ~10 docs per cluster
                
                result = self.run_single_experiment(config, corpus_texts, embeddings)
                results.append(result)
                
            except Exception as e:
                print(f"  -> Error with doc_count {doc_count}: {e}")
                continue
        
        return results
    
    def run_cluster_experiments(self) -> list:
        """Run experiments varying number of clusters."""
        print("\n" + "="*50)
        print("Cluster Count Impact Experiments")
        print("="*50)
        
        exp_config = self.config['experiments']['cluster_count']
        if not exp_config.get('enabled', False):
            print("Cluster count experiments are disabled in config.")
            return []
        
        # Load full dataset
        #corpus_texts, embeddings = self._load_full_dataset()
        
        if 'target_size_group' in exp_config:
            # Use specific size group
            target_group = exp_config['target_size_group']
            n_docs = exp_config.get('n_docs', None)
            corpus_texts, embeddings = self._load_size_group_data(target_group, n_docs)
            print(f"Using size group '{target_group}' with {len(corpus_texts)} documents")
        else:
            # Use full dataset (original behavior)
            corpus_texts, embeddings = self._load_full_dataset()
        
        results = []
        cluster_counts = exp_config['cluster_counts']
        
        for n_clusters in cluster_counts:
            if n_clusters > len(corpus_texts):
                print(f"  -> Skipping {n_clusters} clusters: exceeds document count ({len(corpus_texts)})")
                continue
            config = {
                "name": f"Clusters_{n_clusters}",
                "experiment_type": "cluster_count",
                "n_clusters": n_clusters,
                "n_eval_queries": exp_config['n_eval_queries'],
                **exp_config['fixed_params']
            }
            if 'target_size_group' in exp_config:
                config['size_group'] = exp_config['target_size_group']
                config['n_docs_used'] = len(corpus_texts)
            result = self.run_single_experiment(config, corpus_texts, embeddings)
            results.append(result)
        
        return results
    
    def run_top_k_experiments(self) -> list:
        """Run experiments varying top-k clusters retrieved."""
        print("\n" + "="*50)
        print("Top-K Clusters Impact Experiments")
        print("="*50)
        
        exp_config = self.config['experiments']['top_k_clusters']
        if not exp_config.get('enabled', False):
            print("Top-K cluster experiments are disabled in config.")
            return []
        
        # Load full dataset
        corpus_texts, embeddings = self._load_full_dataset()
        
        results = []
        top_k_values = exp_config['top_k_values']
        
        for top_k in top_k_values:
            config = {
                "name": f"TopK_{top_k}",
                "experiment_type": "top_k_clusters",
                "top_n_clusters": top_k,
                "n_eval_queries": exp_config['n_eval_queries'],
                **exp_config['fixed_params']
            }
            
            result = self.run_single_experiment(config, corpus_texts, embeddings)
            results.append(result)
        
        return results
    
    def run_key_length_experiments(self) -> list:
        """Run experiments varying encryption key length."""
        print("\n" + "="*50)
        print("Key Length Impact Experiments")
        print("="*50)
        
        exp_config = self.config['experiments']['key_length']
        if not exp_config.get('enabled', False):
            print("Key length experiments are disabled in config.")
            return []
        
        # Load full dataset
        corpus_texts, embeddings = self._load_full_dataset()
        
        results = []
        key_lengths = exp_config['key_lengths']
        
        for key_length in key_lengths:
            config = {
                "name": f"KeyLen_{key_length}",
                "experiment_type": "key_length",
                "key_length": key_length,
                "n_eval_queries": exp_config['n_eval_queries'],
                **exp_config['fixed_params']
            }
            
            result = self.run_single_experiment(config, corpus_texts, embeddings)
            results.append(result)
        
        return results
    
    def run_all_experiments(self) -> pd.DataFrame:
        """Run all enabled experiments and return results."""
        all_results = []
        
        # Run different experiment types based on configuration
        all_results.extend(self.run_document_size_experiments())
        all_results.extend(self.run_document_count_experiments())
        all_results.extend(self.run_cluster_experiments())
        all_results.extend(self.run_top_k_experiments())
        all_results.extend(self.run_key_length_experiments())
        
        return pd.DataFrame(all_results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run PIR-RAG experiments")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    
    # Run experiments
    runner = PIRRAGExperimentRunner(args.config)
    
    # Determine output directory
    output_dir = args.output_dir or runner.config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    results_df = runner.run_all_experiments()
    
    if results_df.empty:
        print("\nNo experiments were run. Check your configuration file.")
        return
    
    # Save results
    timestamp = time.strftime(runner.config['output']['timestamp_format'])
    output_filename = f"pir_rag_results_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    
    print("\n\n--- EXPERIMENT RESULTS ---")
    print(results_df.to_string())
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
