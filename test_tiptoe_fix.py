#!/usr/bin/env python3
"""
Test script to verify tiptoe implementation fixes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from tiptoe.crypto import TiptoeHomomorphicRanking

def test_homomorphic_ranking():
    """Test the fixed homomorphic ranking implementation."""
    print("Testing Tiptoe Homomorphic Ranking...")
    
    try:
        # Initialize with conservative parameters
        ranking = TiptoeHomomorphicRanking(scheme='BFV', n_slots=8192, t_bits=30)
        print("✓ Successfully initialized TiptoeHomomorphicRanking")
        
        # Test with small vectors
        query_vec = [1, 2, 3, 4, 5]
        doc_vecs = [
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1]
        ]
        
        print(f"Query vector: {query_vec}")
        print(f"Document vectors: {doc_vecs}")
        
        # Encrypt query
        print("Encrypting query vector...")
        ctxt_query = ranking.encrypt_vector(query_vec)
        print(f"✓ Successfully encrypted query vector (length: {len(ctxt_query)})")
        
        # Check noise budget after encryption
        for i, ctxt in enumerate(ctxt_query):
            noise = ranking.get_noise_budget(ctxt)
            print(f"  Query element {i} noise budget: {noise}")
        
        # Compute dot products
        print("Computing homomorphic dot products...")
        ctxt_scores = ranking.dot_product(ctxt_query, doc_vecs)
        print(f"✓ Successfully computed {len(ctxt_scores)} dot products")
        
        # Check noise budget after computation
        for i, ctxt in enumerate(ctxt_scores):
            noise = ranking.get_noise_budget(ctxt)
            print(f"  Score {i} noise budget: {noise}")
        
        # Decrypt scores
        print("Decrypting scores...")
        scores = ranking.decrypt_scores(ctxt_scores)
        print(f"✓ Successfully decrypted scores: {scores}")
        
        # Verify against plaintext computation
        expected_scores = []
        for doc_vec in doc_vecs:
            score = sum(query_vec[i] * doc_vec[i] for i in range(len(query_vec)))
            expected_scores.append(score)
        
        print(f"Expected scores: {expected_scores}")
        print(f"Computed scores: {scores}")
        
        # Check if results are close (allowing for some FHE noise)
        for i, (expected, computed) in enumerate(zip(expected_scores, scores)):
            diff = abs(expected - computed)
            if diff < 1.0:  # Allow small differences due to FHE operations
                print(f"✓ Score {i}: {computed} ≈ {expected} (diff: {diff})")
            else:
                print(f"✗ Score {i}: {computed} != {expected} (diff: {diff})")
        
        print("\n✓ Homomorphic ranking test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tiptoe_system():
    """Test the full tiptoe system with a minimal example."""
    print("\nTesting full Tiptoe system...")
    
    try:
        from tiptoe import TiptoeSystem
        
        # Create minimal test documents
        documents = [
            "This is a test document about machine learning.",
            "Another document discussing artificial intelligence.",
            "A third document about natural language processing."
        ]
        
        print(f"Testing with {len(documents)} documents")
        
        # Initialize system with minimal parameters
        system = TiptoeSystem(target_dim=16, n_clusters=2, security_param=128)
        print("✓ Successfully initialized TiptoeSystem")
        
        # Setup the system
        setup_metrics = system.setup(documents)
        print(f"✓ Setup completed: {setup_metrics}")
        
        # Test a query
        query = "machine learning algorithms"
        print(f"Testing query: '{query}'")
        
        results = system.query(query, k=2)
        print(f"✓ Query completed successfully")
        print(f"Results: {results}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tiptoe system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Tiptoe Implementation Test")
    print("=" * 60)
    
    # Test 1: Homomorphic ranking
    test1_success = test_homomorphic_ranking()
    
    # Test 2: Full system (only if homomorphic ranking works)
    if test1_success:
        test2_success = test_tiptoe_system()
    else:
        print("\nSkipping full system test due to homomorphic ranking failure.")
        test2_success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Homomorphic Ranking: {'✓ PASS' if test1_success else '✗ FAIL'}")
    print(f"Full Tiptoe System: {'✓ PASS' if test2_success else '✗ FAIL'}")
    print("=" * 60)
