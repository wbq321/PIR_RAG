#!/usr/bin/env python3
"""
Debug script to trace the complete PIR-RAG workflow step by step.
This will help identify where the empty results are coming from.
"""

import os
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pir_rag.server import PIRRAGServer
from pir_rag.client import PIRRAGClient
from tiptoe.crypto_fixed import SimpleLinearHomomorphicScheme

def create_sample_data():
    """Create minimal sample data for testing"""
    print("=" * 60)
    print("STEP 1: Creating Sample Data")
    print("=" * 60)

    # Create simple test documents
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing analyzes human language",
        "Computer vision processes and analyzes visual data",
        "Reinforcement learning learns through trial and error"
    ]

    print(f"Created {len(documents)} test documents")
    for i, doc in enumerate(documents):
        print(f"  Doc {i}: {doc[:50]}...")

    return documents

def create_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    """Create embeddings for documents"""
    print("\n" + "=" * 60)
    print("STEP 2: Creating Embeddings")
    print("=" * 60)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, convert_to_tensor=False)

    print(f"Created embeddings with shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    return embeddings, model

def test_server_setup(documents, embeddings):
    """Test server setup and clustering"""
    print("\n" + "=" * 60)
    print("STEP 3: Server Setup and Clustering")
    print("=" * 60)

    server = PIRRAGServer()
    n_clusters = 3  # Use 3 clusters for 5 documents

    setup_stats = server.setup(embeddings, documents, n_clusters=n_clusters)
    print(f"Server setup completed:")
    for key, value in setup_stats.items():
        print(f"  {key}: {value}")

    # Check cluster assignments
    print(f"\nCluster information:")
    print(f"Number of centroids: {len(server.centroids)}")
    print(f"Centroids shape: {server.centroids.shape}")

    # Print cluster contents
    for cluster_idx, cluster_urls in enumerate(server.clusters_urls):
        print(f"  Cluster {cluster_idx}: {len(cluster_urls)} documents")
        for url in cluster_urls:
            print(f"    {url}")

    return server

def test_client_setup(server):
    """Test client setup"""
    print("\n" + "=" * 60)
    print("STEP 4: Client Setup")
    print("=" * 60)

    client = PIRRAGClient()
    setup_stats = client.setup(server.centroids)

    print(f"Client setup completed:")
    for key, value in setup_stats.items():
        print(f"  {key}: {value}")

    return client

def test_cluster_selection(client, model, query_text):
    """Test cluster selection for a query"""
    print("\n" + "=" * 60)
    print("STEP 5: Cluster Selection")
    print("=" * 60)

    # Create query embedding
    query_embedding = model.encode([query_text], convert_to_tensor=True)[0]
    print(f"Query: {query_text}")
    print(f"Query embedding shape: {query_embedding.shape}")

    # Find relevant clusters
    top_k_clusters = 2
    relevant_clusters = client.find_relevant_clusters(query_embedding, top_k_clusters)

    print(f"Selected {len(relevant_clusters)} clusters: {relevant_clusters}")

    return query_embedding, relevant_clusters

def test_pir_query_generation(client, relevant_clusters, server):
    """Test PIR query generation"""
    print("\n" + "=" * 60)
    print("STEP 6: PIR Query Generation")
    print("=" * 60)

    num_clusters = len(server.centroids)
    print(f"Total clusters: {num_clusters}")

    pir_queries = []
    for cluster_idx in relevant_clusters:
        print(f"\nGenerating PIR query for cluster {cluster_idx}")
        query_vec, upload_size = client.generate_pir_query(cluster_idx, num_clusters)

        print(f"  Query vector length: {len(query_vec)}")
        print(f"  Upload size: {upload_size} bytes")

        # Check the query vector structure
        non_zero_positions = []
        for i, encrypted_val in enumerate(query_vec):
            # Decrypt to check the value (for debugging only)
            decrypted_val = client.crypto_scheme.decrypt(encrypted_val)
            if decrypted_val != 0:
                non_zero_positions.append(i)

        print(f"  Non-zero positions: {non_zero_positions}")
        pir_queries.append((cluster_idx, query_vec))

    return pir_queries

def test_server_pir_processing(server, pir_queries, client):
    """Test server-side PIR processing"""
    print("\n" + "=" * 60)
    print("STEP 7: Server PIR Processing")
    print("=" * 60)

    all_encrypted_results = []

    for cluster_idx, query_vec in pir_queries:
        print(f"\nProcessing PIR query for cluster {cluster_idx}")

        # Process the query
        encrypted_results = server.handle_pir_query(query_vec, client.crypto_scheme)

        print(f"  Received {len(encrypted_results)} encrypted chunks")

        # Try to decrypt a few chunks to see what we get (debugging only)
        sample_decrypted = []
        for i, encrypted_chunk in enumerate(encrypted_results[:5]):  # First 5 chunks
            try:
                decrypted_val = client.crypto_scheme.decrypt(encrypted_chunk)
                sample_decrypted.append(decrypted_val)
            except Exception as e:
                sample_decrypted.append(f"ERROR: {e}")

        print(f"  Sample decrypted values: {sample_decrypted}")
        all_encrypted_results.extend(encrypted_results)

    return all_encrypted_results

def test_client_decryption(client, encrypted_results):
    """Test client-side decryption and URL extraction"""
    print("\n" + "=" * 60)
    print("STEP 8: Client Decryption and URL Extraction")
    print("=" * 60)

    print(f"Decrypting {len(encrypted_results)} encrypted chunks")

    # Decrypt all chunks
    decrypted_chunks = []
    for i, encrypted_chunk in enumerate(encrypted_results):
        try:
            decrypted_val = client.crypto_scheme.decrypt(encrypted_chunk)
            decrypted_chunks.append(decrypted_val)
            if i < 10:  # Show first 10 for debugging
                print(f"  Chunk {i}: {decrypted_val}")
        except Exception as e:
            print(f"  Chunk {i}: DECRYPTION ERROR: {e}")
            decrypted_chunks.append(0)

    # Try to extract URLs and embeddings
    urls, embeddings = client._decrypt_url_chunks(encrypted_results)

    print(f"\nExtracted results:")
    print(f"  URLs: {len(urls)}")
    print(f"  Embeddings: {len(embeddings)}")

    for i, url in enumerate(urls):
        print(f"    {i}: {url}")
        if i < len(embeddings):
            print(f"       Embedding shape: {embeddings[i].shape}")

    return urls, embeddings

def test_complete_workflow():
    """Run the complete PIR-RAG workflow"""
    print("PIR-RAG WORKFLOW DEBUGGING")
    print("=" * 60)

    try:
        # Step 1: Create data
        documents = create_sample_data()

        # Step 2: Create embeddings
        embeddings, model = create_embeddings(documents)

        # Step 3: Setup server
        server = test_server_setup(documents, embeddings)

        # Step 4: Setup client
        client = test_client_setup(server)

        # Step 5: Test with a query
        query_text = "What is machine learning?"
        query_embedding, relevant_clusters = test_cluster_selection(client, model, query_text)

        # Step 6: Generate PIR queries
        pir_queries = test_pir_query_generation(client, relevant_clusters, server)

        # Step 7: Process PIR queries on server
        encrypted_results = test_server_pir_processing(server, pir_queries, client)

        # Step 8: Decrypt and extract results on client
        urls, embeddings = test_client_decryption(client, encrypted_results)

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Successfully retrieved {len(urls)} URLs and {len(embeddings)} embeddings")

        if urls:
            print("SUCCESS: PIR-RAG workflow completed successfully!")
        else:
            print("FAILURE: PIR-RAG workflow returned empty results")

    except Exception as e:
        print(f"\nERROR in workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_workflow()
