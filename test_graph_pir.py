"""
Test Graph-PIR system implementation.
"""

import numpy as np
import sys
import os

# Add PIR_RAG to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from graph_pir import GraphPIRSystem


def test_graph_pir_basic():
    """Test basic Graph-PIR functionality."""
    print("=== Testing Graph-PIR System ===")

    # Create test data
    n_docs = 100
    embed_dim = 384
    np.random.seed(42)

    embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
    documents = [f"This is test document {i} with some content for testing." for i in range(n_docs)]

    print(f"Test data: {n_docs} documents, {embed_dim}D embeddings")

    # Set up Graph-PIR system
    graph_pir = GraphPIRSystem()
    setup_metrics = graph_pir.setup(embeddings, documents)

    print(f"Setup complete in {setup_metrics['total_setup_time']:.2f}s")
    print(f"  - Graph setup: {setup_metrics['graph_setup_time']:.2f}s")
    print(f"  - Vector PIR setup: {setup_metrics['vector_pir_setup_time']:.2f}s")
    print(f"  - Document PIR setup: {setup_metrics['doc_pir_setup_time']:.2f}s")

    # Test query
    query_embedding = np.random.randn(embed_dim).astype(np.float32)
    top_k = 5

    print(f"\nTesting query for top-{top_k} documents...")
    retrieved_docs, query_metrics = graph_pir.query(query_embedding, top_k)

    print(f"Query results:")
    print(f"  - Retrieved {len(retrieved_docs)} documents")
    print(f"  - Phase 1 time: {query_metrics['phase1_time']:.3f}s")
    print(f"  - Phase 2 time: {query_metrics['phase2_time']:.3f}s")
    print(f"  - Total candidates: {query_metrics['total_candidates']}")
    print(f"  - Upload bytes: {query_metrics['phase1_upload_bytes'] + query_metrics['phase2_upload_bytes']}")
    print(f"  - Download bytes: {query_metrics['phase1_download_bytes'] + query_metrics['phase2_download_bytes']}")

    # Show first few retrieved documents
    print(f"\nFirst few retrieved documents:")
    for i, doc in enumerate(retrieved_docs[:3]):
        print(f"  {i+1}. {doc[:80]}...")

    # System info
    info = graph_pir.get_system_info()
    print(f"\nSystem info: {info}")

    print("=== Graph-PIR Test Complete ===")
    return True


if __name__ == "__main__":
    test_graph_pir_basic()
