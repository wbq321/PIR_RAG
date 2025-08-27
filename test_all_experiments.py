#!/usr/bin/env python3
"""
Test script to verify all experiment files work with the updated PIR-RAG implementation
"""

import sys
import os
import numpy as np

# Add the experiments directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

def test_comprehensive_experiment():
    """Test comprehensive_experiment.py"""
    print("🧪 Testing comprehensive_experiment.py...")
    try:
        from comprehensive_experiment import PIRExperimentRunner
        print("  ✅ Import successful")
        
        runner = PIRExperimentRunner()
        print("  ✅ PIRExperimentRunner created")
        
        # Test data generation
        embeddings, documents = runner.generate_test_data(n_docs=5, embed_dim=32, seed=42)
        print(f"  ✅ Test data generated: {len(documents)} docs, {embeddings.shape} embeddings")
        
        # Test PIR-RAG experiment with very small parameters
        queries = np.random.randn(1, 32).astype(np.float32)
        result = runner.run_pir_rag_experiment(embeddings, documents, queries, k=2)
        print("  ✅ PIR-RAG experiment completed")
        print(f"  📊 Result keys: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in comprehensive_experiment: {e}")
        return False

def test_retrieval_performance():
    """Test test_retrieval_performance.py"""
    print("\n🧪 Testing test_retrieval_performance.py...")
    try:
        from test_retrieval_performance import RetrievalPerformanceTester
        print("  ✅ Import successful")
        
        tester = RetrievalPerformanceTester()
        print("  ✅ RetrievalPerformanceTester created")
        
        # Test with very small parameters
        result = tester.run_comprehensive_retrieval_test(n_docs=5, n_queries=2)
        print("  ✅ Comprehensive retrieval test completed")
        print(f"  📊 Result keys: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in test_retrieval_performance: {e}")
        return False

def test_run_pir_comparison():
    """Test run_pir_comparison.py (the working reference)"""
    print("\n🧪 Testing run_pir_comparison.py (reference)...")
    try:
        # Just check if it imports without errors
        from run_pir_comparison import main, generate_test_queries
        print("  ✅ Import successful")
        print("  ✅ generate_test_queries function available")
        
        # Test the generate_test_queries function
        queries = generate_test_queries(n_queries=2, dim=32)
        print(f"  ✅ Generated {len(queries)} test queries")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in run_pir_comparison: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing all experiment files for PIR-RAG compatibility...\n")
    
    results = []
    results.append(("comprehensive_experiment.py", test_comprehensive_experiment()))
    results.append(("test_retrieval_performance.py", test_retrieval_performance()))
    results.append(("run_pir_comparison.py", test_run_pir_comparison()))
    
    print("\n" + "="*60)
    print("📋 TEST SUMMARY:")
    print("="*60)
    
    all_passed = True
    for filename, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {filename}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! All experiment files are ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    print("="*60)
