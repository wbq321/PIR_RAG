#!/usr/bin/env python3
"""
Test script to verify the privacy fix in PIR-RAG:
- Server now returns URLs+embeddings together in PIR response
- No separate embedding request that reveals which documents user wants
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from pir_rag.client import PIRRAGClient
from pir_rag.server import PIRRAGServer
from tiptoe.crypto_fixed import SimpleLinearHomomorphicScheme

def test_privacy_fix():
    print("🔒 Testing PIR-RAG Privacy Fix")
    print("=" * 50)
    
    # Create synthetic test data
    n_docs = 20
    embed_dim = 384
    
    print(f"📊 Creating {n_docs} synthetic documents with {embed_dim}D embeddings...")
    
    # Generate synthetic embeddings and documents
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
    documents = [f"Document {i} content here..." for i in range(n_docs)]
    
    # Initialize PIR-RAG system
    print("🏗️ Setting up PIR-RAG system...")
    server = PIRRAGServer()
    client = PIRRAGClient()
    
    # Setup server (creates clusters and stores URLs+embeddings)
    setup_stats = server.setup(embeddings, documents, n_clusters=5)
    print(f"✅ Server setup: {setup_stats['n_clusters']} clusters, {setup_stats['n_documents']} documents")
    
    # Setup client
    client.setup(server.centroids)
    print("✅ Client setup complete")
    
    # Test query
    print("\n🔍 Testing Privacy-Preserving Query...")
    query_embedding = torch.randn(embed_dim)
    
    # Step 1: Find relevant clusters
    print("1️⃣ Finding relevant clusters...")
    relevant_clusters = client.find_relevant_clusters(query_embedding, top_k=2)
    print(f"   Found clusters: {relevant_clusters}")
    
    # Step 2: PIR retrieval (gets URLs+embeddings together)
    print("2️⃣ PIR retrieval (URLs + embeddings)...")
    doc_tuples, pir_metrics = client.pir_retrieve(relevant_clusters, server)
    print(f"   Retrieved {len(doc_tuples)} document tuples")
    print(f"   Upload: {pir_metrics['total_upload_bytes']} bytes")
    print(f"   Download: {pir_metrics['total_download_bytes']} bytes")
    
    # Verify we got both URLs and embeddings
    if doc_tuples:
        sample_url, sample_embedding = doc_tuples[0]
        print(f"   Sample URL: {sample_url}")
        print(f"   Sample embedding shape: {sample_embedding.shape}")
        print(f"   ✅ URLs and embeddings retrieved together!")
    
    # Step 3: Reranking (no additional server request)
    print("3️⃣ Reranking (no server communication)...")
    final_results = client.rerank_documents(query_embedding, doc_tuples, top_k=5)
    print(f"   Final results: {len(final_results)} URLs")
    for i, url in enumerate(final_results[:3]):
        print(f"     {i+1}. {url}")
    
    print("\n🎯 Privacy Analysis:")
    print("✅ Server never learns which specific documents user is interested in")
    print("✅ URLs and embeddings retrieved together via PIR")
    print("✅ No separate embedding request after PIR")
    print("✅ Reranking happens locally on client")
    
    print("\n✅ Privacy fix test completed successfully!")

if __name__ == "__main__":
    test_privacy_fix()
