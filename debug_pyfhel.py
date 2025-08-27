#!/usr/bin/env python3
"""
Minimal test for debugging the Pyfhel BFV transparent ciphertext issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

def test_basic_pyfhel():
    """Test basic Pyfhel operations to identify the issue."""
    print("Testing basic Pyfhel BFV operations...")
    
    try:
        from Pyfhel import Pyfhel
        
        # Initialize with conservative parameters for Pyfhel 3.4.3
        HE = Pyfhel()
        HE.contextGen(scheme='BFV', n=4096, t_bits=20, sec=128)
        HE.keyGen()
        HE.relinKeyGen()  # Generate relinearization keys
        
        print("✓ Pyfhel context and keys generated")
        
        # Test basic encryption/decryption with single integers
        value = 5
        ctxt = HE.encryptInt(value)  # Encrypt single integer
        decrypted = HE.decryptInt(ctxt)
        print(f"Basic encrypt/decrypt: {value} -> {decrypted}")
        
        # Test multiplication by plaintext
        multiplier = 3
        ctxt_mult = ctxt * multiplier
        decrypted_mult = HE.decryptInt(ctxt_mult)
        print(f"Multiply by plaintext: {value} * {multiplier} = {decrypted_mult}")
        
        # Test addition
        ctxt2 = HE.encryptInt(2)
        ctxt_sum = ctxt + ctxt2
        decrypted_sum = HE.decryptInt(ctxt_sum)
        print(f"Addition: {value} + 2 = {decrypted_sum}")
        
        # Test the problematic operation: dot product
        query = [1, 2, 3]
        doc = [4, 5, 6]
        
        # Encrypt query (each element separately)
        ctxt_query = [HE.encryptInt(x) for x in query]
        
        # Compute dot product
        products = []
        for i in range(len(query)):
            prod = ctxt_query[i] * doc[i]
            products.append(prod)
        
        # Sum the products
        result = products[0].copy()
        for p in products[1:]:
            result += p
        
        # Decrypt result
        dot_product = HE.decryptInt(result)
        expected = sum(query[i] * doc[i] for i in range(len(query)))
        
        print(f"Dot product: {query} · {doc} = {dot_product} (expected: {expected})")
        
        if abs(dot_product - expected) < 1:
            print("✓ Dot product test passed!")
            return True
        else:
            print("✗ Dot product test failed!")
            return False
        
    except Exception as e:
        print(f"✗ Basic Pyfhel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tiptoe_crypto_direct():
    """Test the TiptoeHomomorphicRanking class directly."""
    print("\nTesting TiptoeHomomorphicRanking directly...")
    
    try:
        from tiptoe.crypto import TiptoeHomomorphicRanking
        
        ranking = TiptoeHomomorphicRanking()
        print("✓ TiptoeHomomorphicRanking initialized")
        
        # Simple test vectors
        query = [1, 2]
        docs = [[3, 4], [1, 1]]
        
        print(f"Query: {query}")
        print(f"Documents: {docs}")
        
        # Encrypt query
        ctxt_query = ranking.encrypt_vector(query)
        print("✓ Query encrypted")
        
        # Compute dot products
        ctxt_scores = ranking.dot_product(ctxt_query, docs)
        print("✓ Dot products computed")
        
        # Decrypt
        scores = ranking.decrypt_scores(ctxt_scores)
        print(f"Scores: {scores}")
        
        # Check against expected
        expected = [sum(query[i] * doc[i] for i in range(len(query))) for doc in docs]
        print(f"Expected: {expected}")
        
        return scores == expected or all(abs(s - e) < 1 for s, e in zip(scores, expected))
        
    except Exception as e:
        print(f"✗ TiptoeHomomorphicRanking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Minimal Pyfhel BFV Debug Test")
    print("=" * 50)
    
    # Test basic Pyfhel first
    basic_success = test_basic_pyfhel()
    
    # Test our wrapper if basic test passes
    if basic_success:
        wrapper_success = test_tiptoe_crypto_direct()
    else:
        wrapper_success = False
    
    print("\n" + "=" * 50)
    print("Results:")
    print(f"Basic Pyfhel: {'✓ PASS' if basic_success else '✗ FAIL'}")
    print(f"Tiptoe Crypto: {'✓ PASS' if wrapper_success else '✗ FAIL'}")
    print("=" * 50)
