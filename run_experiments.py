#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_final_experiments.py

A rigorous, end-to-end experimental script for the PIR-RAG system.
This version implements the complete client-side re-ranking pipeline and
calculates effectiveness (Recall@10) on the large-scale dataset.

Usage:
  python run_final_experiments.py \
    --embeddings_path /path/to/your/embeddings_1000000.npy \
    --corpus_path /path/to/your/corpus_1000000.csv
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

from phe import paillier
from sklearn.cluster import KMeans
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import random
# ===================================================================
# Global Settings & Helpers
# ===================================================================
CHUNK_SIZE = 64
MAX_CHUNKS_HOLDER = [0] 

def encode_text_to_chunks(text_blob):
    byte_data = text_blob.encode('utf-8')
    chunks = []
    for i in range(0, len(byte_data), CHUNK_SIZE):
        chunks.append(int.from_bytes(byte_data[i:i+CHUNK_SIZE], 'big'))
    return chunks

def decode_chunks_to_text(int_chunks):
    byte_data = b''.join(
        c.to_bytes((c.bit_length() + 7) // 8, 'big') for c in int_chunks if c != 0
    )
    return byte_data.decode('utf-8', errors='ignore')

def prepare_docs_by_size(all_docs, target_byte_size, tolerance, n_docs):
    """
    从所有文档中筛选出指定平均字节大小的子集。
    :param all_docs: 包含所有文档文本的列表。
    :param target_byte_size: 目标平均字节大小, e.g., 500, 1000, 2000 (bytes)。
    :param tolerance: 允许的平均大小浮动范围, e.g., 0.2 (30%)。
    :param n_docs: 需要筛选出的文档数量, e.g., 1000。
    :return: 筛选出的文档列表。
    """
    print(f"  -> Screening for {n_docs} docs with avg. byte size around {target_byte_size} bytes...")
    selected_docs = []
    # 为了可复现性，固定随机种子
    import random
    random.seed(42)
    # 打乱文档顺序以获得多样性
    shuffled_docs = random.sample(all_docs, len(all_docs))

    # 计算目标字节范围
    lower_bound = (1 - tolerance) * target_byte_size
    upper_bound = (1 + tolerance) * target_byte_size

    for doc in shuffled_docs:
        if len(selected_docs) >= n_docs:
            break
        # 计算文档的UTF-8编码字节长度
        doc_byte_size = len(doc.encode('utf-8'))
        
        # 检查单个文档的字节大小是否在目标范围内
        if lower_bound <= doc_byte_size <= upper_bound:
            selected_docs.append(doc)
    
    if len(selected_docs) < n_docs:
        # 如果筛选不出足够的文档，可以放宽条件或报错
        print(f"Warning: Only found {len(selected_docs)} docs for target byte size {target_byte_size}. Using what we have.")
        if not selected_docs: raise ValueError(f"Could not find any documents for the specified byte size range [{lower_bound:.0f}, {upper_bound:.0f}].")

    # 如果找到了超过所需数量的文档，则截取
    return selected_docs[:n_docs]
# ===================================================================
# Core PIR-RAG Classes
# ===================================================================

class ExperimentServer:
    def setup(self, embeddings, documents_text, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', verbose=0)
        labels = kmeans.fit_predict(embeddings)
        self.centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        clusters_text = [[] for _ in range(n_clusters)]
        self.doc_to_cluster_map = {i: labels[i] for i in range(len(documents_text))}

        for i, text in enumerate(documents_text):
            clusters_text[labels[i]].append(text)

        chunked_clusters = [encode_text_to_chunks("|||".join(c)) for c in clusters_text]
        
        MAX_CHUNKS_HOLDER[0] = max(len(c) for c in chunked_clusters if c) if any(chunked_clusters) else 0

        self.pir_db_by_chunk = [[] for _ in range(MAX_CHUNKS_HOLDER[0])]
        for chunk_idx in range(MAX_CHUNKS_HOLDER[0]):
            for cluster_chunks in chunked_clusters:
                self.pir_db_by_chunk[chunk_idx].append(cluster_chunks[chunk_idx] if chunk_idx < len(cluster_chunks) else 0)

    def handle_pir_query(self, encrypted_query, public_key):
        print(f"    [Query] HE operations PIR ...")
        print(f"    [Query]  Max chunks {MAX_CHUNKS_HOLDER[0]}...")
        return [
            sum(
                [chunk_db[i] * encrypted_query[i] for i in range(len(chunk_db))], 
                public_key.encrypt(0)
            )
            for chunk_db in self.pir_db_by_chunk
        ]

class ExperimentClient:
    def setup(self, centroids, key_length):
        self.centroids = centroids
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        
    def generate_pir_query(self, cluster_idx, num_clusters):
        """生成PIR查询向量和上传大小"""
        query_vec = [self.public_key.encrypt(1) if i == cluster_idx else self.public_key.encrypt(0) for i in range(num_clusters)]
        upload_bytes = sum(sys.getsizeof(c.ciphertext()) for c in query_vec)
        return query_vec, upload_bytes
        
    def decode_pir_response(self, encrypted_chunks):
        """解码PIR响应并返回文档"""
        retrieved_chunks = [self.private_key.decrypt(c) for c in encrypted_chunks]
        retrieved_text = decode_chunks_to_text(retrieved_chunks)
        docs = list(filter(None, retrieved_text.split("|||")))
        download_bytes = sum(sys.getsizeof(c.ciphertext()) for c in encrypted_chunks)
        return docs, download_bytes
        
    def pir_retrieve(self, server, cluster_indices):
        num_clusters = len(self.centroids)
        total_query_gen_time = 0
        total_server_time = 0
        total_decode_time = 0
        candidate_docs_text = []
        total_upload_bytes = 0
        total_download_bytes = 0

        for cluster_idx in cluster_indices:
            #generate query
            
            print(f"    [Query] Preparing PIR for cluster_idx: {cluster_idx}...")
            start_time = time.perf_counter()
            query_vec, upload_bytes = self.generate_pir_query(cluster_idx, num_clusters)
            total_query_gen_time += (time.perf_counter() - start_time)
            total_upload_bytes += upload_bytes
            print(f"    [Query] Finish preparing PIR for cluster_idx: {cluster_idx}...")
            
            #server compute
            server_query_start_time = time.perf_counter()
            print(f"    [Query] Server start query for cluster_idx: {cluster_idx}...")
            encrypted_chunks = server.handle_pir_query(query_vec, self.public_key)
            total_server_time = time.perf_counter() - server_query_start_time
            print(f"    [Query] Server finish query for cluster_idx: {cluster_idx}...")
            print(f"    [Query] Total server query time for this query: {total_server_time:.2f} seconds.")

            #Cliente decode
            start_time = time.perf_counter()
            docs, download_bytes = self.decode_pir_response(encrypted_chunks)
            total_decode_time += (time.perf_counter() - start_time)
            total_download_bytes += download_bytes
            
            candidate_docs_text.extend(docs)

        return candidate_docs_text, total_upload_bytes, total_download_bytes, \
               total_query_gen_time, total_server_time, total_decode_time

# ===================================================================
# Experiment Execution Logic
# ===================================================================

def prepare_evaluation_data(corpus_texts, corpus_embeddings, n_eval_queries=100):
    """Creates a self-contained evaluation set from the corpus."""
    print(f"Preparing {n_eval_queries} evaluation queries...")
    eval_queries = {}
    eval_ground_truth = {}
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    sample_indices = np.random.choice(len(corpus_texts), n_eval_queries, replace=False)
    
    for i, doc_idx in enumerate(sample_indices):
        qid = f"q_{i}"
        # Use the first 30 words as a query, a common heuristic
        query_text = " ".join(corpus_texts[doc_idx].split()[:30])
        
        eval_queries[qid] = {
            "text": query_text,
            "embedding": torch.tensor(corpus_embeddings[doc_idx], dtype=torch.float32).unsqueeze(0)
        }
        # The ground truth is the original document itself
        eval_ground_truth[qid] = corpus_texts[doc_idx]
        
    return eval_queries, eval_ground_truth

def run_single_experiment(
    n_clusters, key_length, top_n_clusters, 
    embeddings, corpus_texts, eval_queries, eval_ground_truth, model
):
    """Runs one full experiment configuration."""
    print("  -> Starting server setup (K-means)...")
    server_setup_start = time.perf_counter()
    server = ExperimentServer()
    server.setup(embeddings, corpus_texts, n_clusters)
    server_setup_time = time.perf_counter() - server_setup_start
    print(f"  -> server setup finished in {server_setup_time:.2f} seconds.")
    
    print("  -> Starting client setup (generating Paillier keys)...")
    client_setup_start = time.perf_counter()
    client = ExperimentClient()
    client.setup(server.centroids, key_length)
    client_setup_time = time.perf_counter() - client_setup_start
    print(f"  -> Client setup finished in {client_setup_time:.2f} seconds.")
    
    latencies, uploads, downloads = [], [], []
    query_gen_times, server_times, decode_times = [], [], []
    hits_at_10 = 0
    
    print("  -> Starting query loop...")
    for qid, query_data in tqdm(eval_queries.items(), desc="  Querying", leave=False):
        query_embedding = query_data["embedding"]
        
        query_start_time = time.perf_counter()
        
        # 1. Client-side ANN vs centroids
        similarities = util.cos_sim(query_embedding, client.centroids)[0]
        best_cluster_indices = torch.topk(similarities, k=top_n_clusters).indices.tolist()

        # 2. PIR retrieval (now returns docs)
        query_start_time = time.perf_counter()
        retrieved_docs, upload_bytes, download_bytes, \
        query_gen_time, server_time, decode_time = client.pir_retrieve(server, best_cluster_indices)
        
        # 3. Client-side final re-ranking
        if retrieved_docs:
            retrieved_embeddings = model.encode(
                retrieved_docs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False
            )
            final_sims = util.cos_sim(query_embedding, retrieved_embeddings)[0]
            num_retrieved = len(retrieved_docs)
            top_k_value = min(10, num_retrieved)
            if top_k_value > 0: # 确保至少有一个文档可供选择
                top_k_indices = torch.topk(final_sims, k=top_k_value).indices
                final_top_10_docs = {retrieved_docs[j] for j in top_k_indices}
            else: # 如果 retrieved_docs 为空（虽然上面的 if 已经处理了，但这样更健壮）
                final_top_10_docs = set()
        else:
            final_top_10_docs = set()
            
        query_latency = time.perf_counter() - query_start_time

        latencies.append(query_latency)
        uploads.append(upload_bytes)
        downloads.append(download_bytes)
        query_gen_times.append(query_gen_time)
        server_times.append(server_time)
        decode_times.append(decode_time)
        
        if eval_ground_truth[qid] in final_top_10_docs:
            hits_at_10 += 1
            
    return {
        "n_docs": len(corpus_texts),
        "n_clusters": n_clusters,
        "key_length": key_length,
        "top_n_clusters": top_n_clusters,
        "server_setup_s": server_setup_time,
        "client_setup_s": client_setup_time,
        "avg_query_latency_s": np.mean(latencies),
        "avg_upload_kb": np.mean(uploads) / 1024,
        "avg_download_kb": np.mean(downloads) / 1024,
        "recall_at_10": hits_at_10 / len(eval_queries),
        "avg_query_gen_s": np.mean(query_gen_times),
        "avg_server_comm_s": np.mean(server_times),
        "avg_decode_s": np.mean(decode_times),
    }

def main(args):
    print("--- Loading Data and Models ---")
    full_embeddings = np.load(args.embeddings_path)
    full_corpus_df = pd.read_csv(args.corpus_path)
    full_corpus_texts = full_corpus_df['text'].dropna().tolist()

    if len(full_embeddings) != len(full_corpus_texts):
        raise ValueError("Embedding and corpus file length mismatch.")

    # Load model once
    local_model_path = './local_bge_model'
    model = SentenceTransformer(local_model_path, device='cpu')
    full_eval_queries, full_eval_ground_truth = prepare_evaluation_data(full_corpus_texts, full_embeddings, n_eval_queries=100)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    all_results = []


    #Experiment for different document size
    print("\n" + "="*20, "Experiment: Impact of Document Size", "="*20)
    document_sizes_to_test = [1500, 2500, 3500, 4500]
    for doc_size in document_sizes_to_test:
        print(f"\n--- Running for documents with average size: ~{doc_size} tokens ---")

        # 1. 准备特定大小的数据集 (1000个文档)
        try:
            # 筛选出500个文档，其平均token长度在目标值上下10%的范围内
            corpus_texts = prepare_docs_by_size(full_corpus_texts, doc_size, 0.5, 100)

            # 找到这些文档在原始列表中的索引
            indices = [full_corpus_texts.index(doc) for doc in corpus_texts]
            # 根据索引筛选embeddings
            embeddings = full_embeddings[indices]

            # 计算并记录实际的平均大小
            actual_avg_size = np.mean([len(t.encode('utf-8')) for t in corpus_texts])
            print(f"  -> Prepared {len(corpus_texts)} docs with actual avg. size: {actual_avg_size:.2f} .")

        except ValueError as e:
            print(f"  -> {e}. Skipping size {doc_size}.")
            continue

        eval_queries, eval_ground_truth = prepare_evaluation_data(corpus_texts, embeddings, n_eval_queries=10)
        print("  -> Evaluation data prepared.")
        params = {
            'n_clusters': 10,    # 聚类数减少，因为总文档数少了
            'key_length': 1024,  # 密钥长度减小以加速
            'top_n_clusters': 1, # 只检索一个簇，简化分析
            'embeddings': embeddings,
            'corpus_texts': corpus_texts,
            'eval_queries': eval_queries,
            'eval_ground_truth': eval_ground_truth,
            'model': model,
        }
        result = run_single_experiment(**params)
        all_results.append(result)

    '''
    base_params = {
        'embeddings': full_embeddings,
        'corpus_texts': full_corpus_texts,
        'eval_queries': full_eval_queries,
        'eval_ground_truth': full_eval_ground_truth,
        'model': model
    }

    # --- Run Experiments ---
    print("\n" + "="*20, "Experiment 1: Impact of n_clusters", "="*20)
    for n_c in [100, 500, 1000, 5000, 10000]:
        print(f"Running for n_clusters = {n_c}")
        result = run_single_experiment(n_clusters=n_c, key_length=2048, top_n_clusters=3, **base_params)
        all_results.append(result)

    print("\n" + "="*20, "Experiment 2: Impact of top_n_clusters", "="*20)
    for k in [1, 2, 3, 5, 10]:
        print(f"Running for top_n_clusters = {k}")
        result = run_single_experiment(n_clusters=1000, key_length=2048, top_n_clusters=k, **base_params)
        all_results.append(result)

    print("\n" + "="*20, "Experiment 3: Impact of key_length", "="*20)
    for kl in [1024, 2048, 3072]:
        print(f"Running for key_length = {kl}")
        result = run_single_experiment(n_clusters=1000, key_length=kl, top_n_clusters=3, **base_params)
        all_results.append(result)
    '''
    # --- Save Results ---
    results_df = pd.DataFrame(all_results)
    print("\n\n--- FINAL EXPERIMENT RESULTS ---")
    print(results_df.to_string())

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"pir_rag_final_results_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    
    parser = argparse.ArgumentParser(description="Run final PIR-RAG experiments.")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to .npy embeddings file.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to .csv corpus file.")
    args = parser.parse_args()
    
    main(args)