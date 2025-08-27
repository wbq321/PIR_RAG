#!/usr/bin/env python3
"""
Quick test to verify tiptoe implementation is using real crypto.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from tiptoe import TiptoeSystem

def test_tiptoe_crypto_usage():
    """Test if tiptoe is using real cryptography."""
    print("=" * 60)
    print("Testing Tiptoe Real Crypto Usage")
    print("=" * 60)
    
    # Create test data
    documents = [
        "This is document 1 about machine learning and AI.",
        "Document 2 discusses natural language processing.",
        "The third document covers computer vision topics.",
        "Document 4 is about deep learning algorithms.",
        "This fifth document explores reinforcement learning."
    ]
    
    embeddings = np.random.rand(len(documents), 384).astype(np.float32)
    
    # Initialize tiptoe system
    print(f"Setting up Tiptoe with {len(documents)} documents...")
    tiptoe = TiptoeSystem(target_dim=32, n_clusters=2, security_param=128)
    setup_metrics = tiptoe.setup(embeddings, documents)
    
    print(f"✓ Setup complete in {setup_metrics['total_setup_time']:.3f}s")
    
    # Check what crypto is being used
    print("\nCrypto Implementation Check:")
    print(f"  Homomorphic Ranking: {'REAL Pyfhel' if tiptoe.homomorphic_ranking else 'SIMULATED'}")
    print(f"  PIR System: {'REAL Paillier' if hasattr(tiptoe.pir_system, 'key_length') else 'SIMULATED'}")
    
    # Test a query
    query_embedding = np.random.rand(384).astype(np.float32)
    print(f"\nTesting query...")
    
    retrieved_docs, metrics = tiptoe.query(query_embedding, top_k=3)
    
    print(f"✓ Query completed in {metrics['total_query_time']:.3f}s")
    print(f"  Retrieved {len(retrieved_docs)} documents")
    print(f"  Selected cluster: {metrics['selected_cluster']}")
    
    # Check communication costs
    print(f"\nCommunication Analysis:")
    print(f"  Phase 1 Upload: {metrics.get('phase1_upload_bytes', 0):,} bytes")
    print(f"  Phase 1 Download: {metrics.get('phase1_download_bytes', 0):,} bytes")
    print(f"  Phase 2 Upload: {metrics.get('phase2_upload_bytes', 0):,} bytes")
    print(f"  Phase 2 Download: {metrics.get('phase2_download_bytes', 0):,} bytes")
    print(f"  Total Upload: {metrics.get('upload_bytes', 0):,} bytes")
    print(f"  Total Download: {metrics.get('download_bytes', 0):,} bytes")
    
    # Analyze results
    total_comm = metrics.get('upload_bytes', 0) + metrics.get('download_bytes', 0)
    
    print(f"\n" + "=" * 60)
    print("Analysis:")
    
    if tiptoe.homomorphic_ranking and hasattr(tiptoe.pir_system, 'key_length'):
        crypto_status = "FULL REAL CRYPTO"
        expected_min_comm = 1000  # Should be at least 1KB for real crypto
    elif tiptoe.homomorphic_ranking:
        crypto_status = "HYBRID (Real Ranking + Simulated PIR)"
        expected_min_comm = 500  # Should still have some communication
    else:
        crypto_status = "FULLY SIMULATED"
        expected_min_comm = 0
    
    print(f"  Crypto Status: {crypto_status}")
    print(f"  Total Communication: {total_comm:,} bytes")
    
    if total_comm >= expected_min_comm:
        print(f"  ✓ Communication costs look realistic for {crypto_status}")
    else:
        print(f"  ⚠ Communication costs seem too low for {crypto_status}")
        print(f"    Expected at least {expected_min_comm} bytes")
    
    print("=" * 60)
    
    return {
        'has_real_ranking': tiptoe.homomorphic_ranking is not None,
        'has_real_pir': hasattr(tiptoe.pir_system, 'key_length'),
        'total_communication': total_comm,
        'query_time': metrics['total_query_time']
    }

if __name__ == "__main__":
    results = test_tiptoe_crypto_usage()
    
    print(f"\nSummary:")
    print(f"  Real Homomorphic Ranking: {'✓' if results['has_real_ranking'] else '✗'}")
    print(f"  Real PIR: {'✓' if results['has_real_pir'] else '✗'}")
    print(f"  Query Time: {results['query_time']:.3f}s")
    print(f"  Communication: {results['total_communication']:,} bytes")
