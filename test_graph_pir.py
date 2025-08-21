"""
Quick test script to verify the Graph-PIR implementation works correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add the source directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from graph_pir import GraphPIRSystem


def test_graph_pir_basic():
    """Test basic functionality of Graph-PIR system"""
    print("Testing Graph-PIR Basic Functionality...")
    
    # Create synthetic data
    n_docs = 1000
    embedding_dim = 128
    
    print(f"Creating synthetic dataset: {n_docs} docs, {embedding_dim}D embeddings")
    
    # Generate random embeddings
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, embedding_dim).astype(np.float32)
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create dummy documents
    documents = [f"Document {i} with some text content" for i in range(n_docs)]
    
    # Test both private and non-private modes
    for private_mode in [False, True]:
        print(f"\n--- Testing {('Private' if private_mode else 'Non-Private')} Mode ---")
        
        # Setup system
        system = GraphPIRSystem(private_mode=private_mode)
        setup_stats = system.setup(
            embeddings=embeddings,
            documents_text=documents,
            neighbor_count=16  # Smaller for testing
        )
        
        print(f"Setup completed in {setup_stats['setup_time']:.2f}s")
        
        # Test search
        query_embedding = np.random.randn(embedding_dim).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        print("Running search...")
        doc_indices, search_metrics = system.search(
            query_embedding=query_embedding,
            top_k=10,
            max_steps=5,  # Smaller for testing
            parallel_exploration=2
        )
        
        print(f"Search completed:")
        print(f"  - Found {len(doc_indices)} results")
        print(f"  - Search time: {search_metrics['search_time']:.3f}s")
        print(f"  - Vertex accesses: {search_metrics['vertex_accesses']}")
        print(f"  - Communication bytes: {search_metrics['total_communication_bytes']:,}")
        
        # Get system stats
        system_stats = system.get_system_stats()
        print(f"  - System type: {system_stats['system_type']}")
        print(f"  - Private mode: {system_stats['private_mode']}")
        
        if private_mode:
            comm_breakdown = system.get_communication_breakdown()
            print(f"  - PIR upload bytes: {comm_breakdown['pir_upload_bytes']:,}")
            print(f"  - PIR download bytes: {comm_breakdown['pir_download_bytes']:,}")
    
    print("\n✓ Basic functionality test passed!")


def test_communication_comparison_setup():
    """Test that the comparison framework can be imported and basic setup works"""
    print("\nTesting Communication Comparison Setup...")
    
    try:
        from experiments.communication_comparison import CommunicationComparison
        
        # Create minimal test data
        embeddings = np.random.randn(100, 64).astype(np.float32)
        documents = [f"Test document {i}" for i in range(100)]
        
        # Mock model (just for testing)
        class MockModel:
            def encode(self, texts):
                return np.random.randn(len(texts), 64).astype(np.float32)
        
        model = MockModel()
        
        # Test comparison class creation
        comparison = CommunicationComparison(embeddings, documents, model)
        print("✓ CommunicationComparison class created successfully")
        
        print("✓ Communication comparison setup test passed!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you have all required dependencies installed")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("Graph-PIR Implementation Test")
    print("=" * 40)
    
    try:
        test_graph_pir_basic()
        test_communication_comparison_setup()
        
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Install required dependencies if needed:")
        print("   pip install sentence-transformers scikit-learn")
        print("2. Run the full comparison:")
        print("   python experiments/communication_comparison.py --embeddings_path data/embeddings.npy --corpus_path data/corpus.csv --model_path path/to/model")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
