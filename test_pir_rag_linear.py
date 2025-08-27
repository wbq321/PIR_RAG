#!/usr/bin/env python3
"""
Test the modified PIR-RAG system with SimpleLinearHomomorphicScheme.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import time
import torch
from pir_rag import PIRRAGClient, PIRRAGServer

def test_pir_rag_linear():
    """Test PIR-RAG with the fast linear homomorphic scheme."""
    print("="*60)
    print("TESTING PIR-RAG WITH LINEAR HOMOMORPHIC SCHEME")
    print("="*60)

    # Create test data
    print("Creating test data...")
    n_docs = 20
    embedding_dim = 384
    n_clusters = 5

    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
    documents = [
        f"Test document {i}: This contains information about topic {i % 3}. "
        f"Document {i} has unique content for testing retrieval accuracy."
        for i in range(n_docs)
    ]

    print(f"Created {n_docs} documents, {n_clusters} clusters")

    # Setup server
    print("\n--- Server Setup ---")
    server = PIRRAGServer()
    server_start = time.perf_counter()
    server_stats = server.setup(embeddings, documents, n_clusters)
    server_time = time.perf_counter() - server_start
    print(f"Server setup: {server_time:.3f}s")
    print(f"Setup stats: {server_stats}")

    # Setup client
    print("\n--- Client Setup ---")
    client = PIRRAGClient()
    client_start = time.perf_counter()
    client_stats = client.setup(server.centroids)
    client_time = time.perf_counter() - client_start
    print(f"Client setup: {client_time:.3f}s")
    print(f"Setup stats: {client_stats}")

    # Test query
    print("\n--- Query Test ---")
    query_embedding = torch.tensor(np.random.randn(embedding_dim).astype(np.float32))
    
    # Find relevant clusters
    print("Finding relevant clusters...")
    cluster_start = time.perf_counter()
    cluster_indices = client.find_relevant_clusters(query_embedding, top_k=2)
    cluster_time = time.perf_counter() - cluster_start
    print(f"Cluster selection: {cluster_time:.3f}s, selected clusters: {cluster_indices}")

    # PIR retrieval
    print("PIR retrieval...")
    pir_start = time.perf_counter()
    retrieved_urls, pir_metrics = client.pir_retrieve(server, cluster_indices)
    pir_time = time.perf_counter() - pir_start
    print(f"PIR retrieval: {pir_time:.3f}s")
    print(f"Retrieved {len(retrieved_urls)} URLs")
    print(f"PIR metrics: {pir_metrics}")
    
    # Re-ranking
    print("Re-ranking documents...")
    rerank_start = time.perf_counter()
    final_urls = client.rerank_documents(query_embedding, retrieved_urls, server, top_k=5)
    rerank_time = time.perf_counter() - rerank_start
    print(f"Re-ranking: {rerank_time:.3f}s")
    
    print(f"\nFinal results: {len(final_urls)} URLs")
    for i, url in enumerate(final_urls[:3]):
        print(f"  {i+1}. {url}")

    # Total performance
    total_time = cluster_time + pir_time + rerank_time
    print(f"\nTotal query time: {total_time:.3f}s")
    print(f"  - Cluster selection: {cluster_time:.3f}s")
    print(f"  - PIR retrieval: {pir_time:.3f}s") 
    print(f"  - Re-ranking: {rerank_time:.3f}s")

    print("\n✅ PIR-RAG with LinearHomomorphicScheme works!")
    return True

if __name__ == "__main__":
    try:
        test_pir_rag_linear()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
