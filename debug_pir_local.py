#!/usr/bin/env python3
"""
Local PIR-RAG workflow debugging script - no external model downloads required.
Uses random embeddings to test the PIR workflow without network dependencies.
"""

import os
import sys
import numpy as np
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pir_rag.server import PIRRAGServer
from pir_rag.client import PIRRAGClient
from tiptoe.crypto_fixed import SimpleLinearHomomorphicScheme

def create_local_embeddings(documents, embedding_dim=384):
    """Create deterministic local embeddings without external models"""
    print("\n" + "=" * 60)
    print("STEP 2: Creating Local Embeddings (No Network Required)")
    print("=" * 60)

    # Use deterministic random embeddings based on document content
    embeddings = []
    for i, doc in enumerate(documents):
        # Create pseudo-embeddings based on document hash
        np.random.seed(hash(doc) % 2**32)  # Deterministic seed
        embedding = np.random.normal(0, 1, embedding_dim).astype(np.float32)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    print(f"Created local embeddings with shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    return embeddings

def create_local_query_embedding(query_text, embedding_dim=384):
    """Create local query embedding"""
    np.random.seed(hash(query_text) % 2**32)  # Deterministic seed
    embedding = np.random.normal(0, 1, embedding_dim).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

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
        print(f"  Doc {i}: {doc}")

    return documents

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

    # Show document-to-cluster mapping
    print(f"\nDocument-to-cluster mapping:")
    for doc_idx, cluster_idx in server.doc_to_cluster_map.items():
        print(f"  Doc {doc_idx} -> Cluster {cluster_idx}")

    return server

def test_client_setup(server):
    """Test client setup"""
    print("\n" + "=" * 60)
    print("STEP 4: Client Setup")
    print("=" * 60)

    client = PIRRAGClient()

    # Convert numpy centroids to torch tensor if needed
    import torch
    if isinstance(server.centroids, np.ndarray):
        centroids_tensor = torch.from_numpy(server.centroids)
    else:
        centroids_tensor = server.centroids

    setup_stats = client.setup(centroids_tensor)

    print(f"Client setup completed:")
    for key, value in setup_stats.items():
        print(f"  {key}: {value}")

    return client

def test_cluster_selection(client, query_embedding):
    """Test cluster selection for a query"""
    print("\n" + "=" * 60)
    print("STEP 5: Cluster Selection")
    print("=" * 60)

    import torch

    # Convert to torch tensor if needed
    if isinstance(query_embedding, np.ndarray):
        query_tensor = torch.from_numpy(query_embedding)
    else:
        query_tensor = query_embedding

    print(f"Query embedding shape: {query_tensor.shape}")

    # Find relevant clusters
    top_k_clusters = 2
    relevant_clusters = client.find_relevant_clusters(query_tensor, top_k_clusters)

    print(f"Selected {len(relevant_clusters)} clusters: {relevant_clusters}")

    return query_tensor, relevant_clusters

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
        print(f"  Query vector length: {len(query_vec)}")

        # Check what documents are in this cluster
        cluster_urls = server.clusters_urls[cluster_idx]
        print(f"  Cluster {cluster_idx} contains {len(cluster_urls)} URLs:")
        for url in cluster_urls:
            print(f"    {url}")

        # Process the query
        encrypted_results = server.handle_pir_query(query_vec, client.crypto_scheme)

        print(f"  Received {len(encrypted_results)} encrypted chunks")

        # Try to decrypt a few chunks to see what we get (debugging only)
        sample_decrypted = []
        for i, encrypted_chunk in enumerate(encrypted_results[:10]):  # First 10 chunks
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
            if i < 20:  # Show first 20 for debugging
                print(f"  Chunk {i}: {decrypted_val}")
        except Exception as e:
            print(f"  Chunk {i}: DECRYPTION ERROR: {e}")
            decrypted_chunks.append(0)

    print(f"\nTotal decrypted chunks: {len(decrypted_chunks)}")
    non_zero_chunks = [x for x in decrypted_chunks if x != 0]
    print(f"Non-zero chunks: {len(non_zero_chunks)}")

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
    print("PIR-RAG WORKFLOW DEBUGGING (LOCAL MODE)")
    print("=" * 80)

    try:
        # Step 1: Create data
        documents = create_sample_data()

        # Step 2: Create local embeddings (no network required)
        embeddings = create_local_embeddings(documents)

        # Step 3: Setup server
        server = test_server_setup(documents, embeddings)

        # Step 4: Setup client
        client = test_client_setup(server)

        # Step 5: Test with a query
        query_text = "What is machine learning?"
        print(f"\nQuery: {query_text}")
        query_embedding = create_local_query_embedding(query_text)
        query_tensor, relevant_clusters = test_cluster_selection(client, query_embedding)

        # Step 6: Generate PIR queries
        pir_queries = test_pir_query_generation(client, relevant_clusters, server)

        # Step 7: Process PIR queries on server
        encrypted_results = test_server_pir_processing(server, pir_queries, client)

        # Step 8: Decrypt and extract results on client
        urls, embeddings_result = test_client_decryption(client, encrypted_results)

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Successfully retrieved {len(urls)} URLs and {len(embeddings_result)} embeddings")

        if urls:
            print("SUCCESS: PIR-RAG workflow completed successfully!")
            print("\nRetrieved URLs:")
            for i, url in enumerate(urls):
                print(f"  {i+1}: {url}")
        else:
            print("FAILURE: PIR-RAG workflow returned empty results")
            print("This indicates an issue in the PIR encryption/decryption process")

    except Exception as e:
        print(f"\nERROR in workflow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_workflow()
