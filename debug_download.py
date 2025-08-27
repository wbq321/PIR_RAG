#!/usr/bin/env python3
"""
Debug script to analyze what PIR-RAG server sends to client
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from pir_rag.client import PIRRAGClient
from pir_rag.server import PIRRAGServer

def debug_download_size():
    print("🔍 Debugging PIR-RAG Download Size")
    print("=" * 50)
    
    # Create small test case
    n_docs = 5
    embed_dim = 10  # Small embedding for testing
    
    # Generate synthetic data
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
    documents = [f"Document {i}" for i in range(n_docs)]
    
    # Setup system
    server = PIRRAGServer()
    client = PIRRAGClient()
    
    server.setup(embeddings, documents, n_clusters=2)
    client.setup(server.centroids)
    
    print(f"📊 Test setup: {n_docs} docs, {embed_dim}D embeddings, 2 clusters")
    
    # Generate a PIR query for cluster 0
    query_vec, upload_bytes = client.generate_pir_query(0, 2)
    print(f"📤 Upload size: {upload_bytes} bytes")
    
    # Debug what server sends
    print("\n🔍 Analyzing server response...")
    
    # Check cluster contents
    cluster_0_urls = server.clusters_urls[0]
    print(f"Cluster 0 URLs: {cluster_0_urls}")
    
    # Simulate what server creates for cluster 0
    cluster_data_parts = []
    total_chars = 0
    
    for url in cluster_0_urls:
        doc_idx = int(url.split('_')[-1])
        embedding = server.document_embeddings[doc_idx]
        embedding_str = ','.join(map(str, embedding))
        combined_data = f"{url}|||{embedding_str}"
        cluster_data_parts.append(combined_data)
        
        print(f"  URL: {url}")
        print(f"  Embedding length: {len(embedding_str)} chars")
        print(f"  Combined data length: {len(combined_data)} chars")
        total_chars += len(combined_data)
    
    cluster_combined_string = "###".join(cluster_data_parts)
    print(f"\n📊 Total cluster string length: {len(cluster_combined_string)} chars")
    print(f"📊 Total cluster string bytes: {len(cluster_combined_string.encode('utf-8'))} bytes")
    
    # Simulate actual server response
    encrypted_chunks = server.handle_pir_query(query_vec, client.crypto_scheme)
    print(f"📊 Encrypted chunks: {len(encrypted_chunks)} chunks")
    
    # Estimate download size
    download_bytes = len(str(encrypted_chunks).encode('utf-8'))
    print(f"📥 Estimated download size: {download_bytes:,} bytes")
    
    print(f"\n🎯 Problem Analysis:")
    print(f"  - Each embedding becomes a {len(embedding_str)} character string")
    print(f"  - With 384D embeddings, this becomes ~3,000+ chars per embedding")
    print(f"  - Multiple documents per cluster multiply this")
    print(f"  - Encryption overhead adds more data")
    
    print(f"\n💡 Solution: Use binary encoding instead of string conversion")

if __name__ == "__main__":
    debug_download_size()
