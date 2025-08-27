#!/usr/bin/env python3
"""
Test script to verify Graph-PIR new arguments work correctly.
"""

import numpy as np
import sys
import os

# Add PIR_RAG to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from graph_pir import GraphPIRSystem

def test_graph_pir_arguments():
    """Test Graph-PIR with custom max_iterations and nodes_per_step parameters."""
    print("Testing Graph-PIR with custom traversal arguments...")

    # Create small test data
    n_docs = 20
    embed_dim = 16
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
    documents = [f"Document {i}: Test content for document {i}" for i in range(n_docs)]

    # Test with custom parameters
    custom_params = {
        'k_neighbors': 8,
        'ef_construction': 100,
        'max_connections': 8,
        'max_iterations': 15,  # Custom: more iterations
        'nodes_per_step': 3    # Custom: fewer nodes per step
    }

    print(f"Testing with parameters: {custom_params}")

    try:
        # Create system and setup
        system = GraphPIRSystem()
        setup_metrics = system.setup(embeddings, documents, graph_params=custom_params)

        # Verify parameters are stored correctly
        assert hasattr(system, 'max_iterations'), "max_iterations not stored"
        assert hasattr(system, 'nodes_per_step'), "nodes_per_step not stored"
        assert system.max_iterations == 15, f"Expected max_iterations=15, got {system.max_iterations}"
        assert system.nodes_per_step == 3, f"Expected nodes_per_step=3, got {system.nodes_per_step}"

        print(f"✅ Parameters stored correctly:")
        print(f"  max_iterations: {system.max_iterations}")
        print(f"  nodes_per_step: {system.nodes_per_step}")

        # Test a query
        query_embedding = np.random.randn(embed_dim).astype(np.float32)
        urls, query_metrics = system.query(query_embedding, top_k=5)

        print(f"✅ Query completed successfully")
        print(f"  Returned {len(urls)} URLs")
        print(f"  Graph traversal steps: {query_metrics.get('graph_traversal_steps', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_default_parameters():
    """Test Graph-PIR with default parameters."""
    print("\nTesting Graph-PIR with default parameters...")

    # Create small test data
    n_docs = 10
    embed_dim = 16
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
    documents = [f"Document {i}: Test content for document {i}" for i in range(n_docs)]

    try:
        # Create system with no custom parameters (should use defaults)
        system = GraphPIRSystem()
        setup_metrics = system.setup(embeddings, documents)  # No graph_params provided

        # Verify default parameters
        assert system.max_iterations == 10, f"Expected default max_iterations=10, got {system.max_iterations}"
        assert system.nodes_per_step == 5, f"Expected default nodes_per_step=5, got {system.nodes_per_step}"

        print(f"✅ Default parameters correct:")
        print(f"  max_iterations: {system.max_iterations}")
        print(f"  nodes_per_step: {system.nodes_per_step}")

        return True

    except Exception as e:
        print(f"❌ Default test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Graph-PIR New Traversal Arguments")
    print("=" * 60)

    success1 = test_graph_pir_arguments()
    success2 = test_default_parameters()

    if success1 and success2:
        print("\n🎉 All tests passed! Graph-PIR traversal arguments are working correctly.")
    else:
        print("\n💥 Some tests failed. Check the output above.")
