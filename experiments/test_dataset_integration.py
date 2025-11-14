#!/usr/bin/env python3
"""
Test script for dataset integration with PIR-RAG experiments
"""

import os
import sys

def test_dataset_integration():
    """Test the dataset integration functionality."""
    
    print("🧪 Testing Dataset Integration")
    print("=" * 50)
    
    # Test 1: Dataset loader functionality
    print("\n1️⃣ Testing dataset loader...")
    try:
        from dataset_loader import DatasetLoader
        
        # Test with smaller limits for memory efficiency
        loader = DatasetLoader()
        
        # Get basic info first
        info = loader.get_dataset_info()
        
        for dataset_name, details in info.items():
            if details.get('sample_loaded'):
                print(f"  ✅ {dataset_name}: {details['eval_metric']}")
                print(f"     Queries: {details['queries_shape']}, Database: {details['database_shape']}")
            else:
                print(f"  ❌ {dataset_name}: {details.get('error', 'Failed')}")
                
        # Test SIFT with smaller limits to avoid memory issues
        print(f"\n  🔧 Testing SIFT with smaller limits...")
        try:
            queries, database, ground_truth = loader.load_dataset("SIFT", max_queries=3, max_docs=50)
            print(f"  ✅ SIFT (small): Recall@10")
            print(f"     Queries: {queries.shape}, Database: {database.shape}")
        except Exception as e:
            print(f"  ❌ SIFT (small): {e}")
            
    except Exception as e:
        print(f"  ❌ Dataset loader test failed: {e}")
        return
    
    # Test 2: Retrieval performance tester with dataset metrics
    print("\n2️⃣ Testing dataset metrics calculation...")
    try:
        from test_retrieval_performance import RetrievalPerformanceTester
        tester = RetrievalPerformanceTester()
        
        # Test MRR calculation
        retrieved = [5, 2, 8, 1, 10]
        ground_truth_laion = [2]  # Correct answer is doc 2
        mrr_result = tester.calculate_dataset_metrics(retrieved, ground_truth_laion, "LAION", 0)
        print(f"  ✅ LAION MRR@100 test: {mrr_result}")
        
        # Test Recall calculation 
        import numpy as np
        ground_truth_sift = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  # True neighbors
        recall_result = tester.calculate_dataset_metrics(retrieved, ground_truth_sift, "SIFT", 0)
        print(f"  ✅ SIFT Recall@10 test: {recall_result}")
        
    except Exception as e:
        print(f"  ❌ Metrics calculation test failed: {e}")
        return
    
    # Test 3: Dataset loading for each dataset
    print("\n3️⃣ Testing individual dataset loading...")
    try:
        loader = DatasetLoader()
        
        # Test with memory-efficient limits
        test_configs = [
            ("LAION", 2, 10),
            ("MS_MARCO", 2, 10), 
            ("SIFT", 2, 50)  # Smaller limit for SIFT to avoid memory issues
        ]
        
        for dataset_name, max_queries, max_docs in test_configs:
            try:
                queries, database, ground_truth = loader.load_dataset(dataset_name, max_queries=max_queries, max_docs=max_docs)
                print(f"  ✅ {dataset_name}: {queries.shape} queries, {database.shape} database")
                print(f"     Ground truth type: {type(ground_truth)}")
            except Exception as e:
                print(f"  ⚠️ {dataset_name}: {e}")
                
    except Exception as e:
        print(f"  ❌ Individual dataset loading failed: {e}")
        return
    
    # Test 4: Import comprehensive experiment
    print("\n4️⃣ Testing comprehensive experiment integration...")
    try:
        from comprehensive_experiment import PIRExperimentRunner
        runner = PIRExperimentRunner()
        print("  ✅ PIRExperimentRunner imported successfully")
        print("  ✅ Dataset parameters should be available in run_retrieval_performance_experiment")
        
        # Check if the method signature is correct
        import inspect
        sig = inspect.signature(runner.run_retrieval_performance_experiment)
        if 'dataset_name' in sig.parameters:
            print("  ✅ dataset_name parameter found in method signature")
        else:
            print("  ❌ dataset_name parameter missing from method signature")
            
    except Exception as e:
        print(f"  ❌ Comprehensive experiment integration test failed: {e}")
        return
    
    print("\n🎉 All tests completed!")
    print("\n📋 Usage examples:")
    print("1. Test with LAION dataset:")
    print("   python comprehensive_experiment.py --experiment retrieval --dataset LAION --max-dataset-docs 100 --max-dataset-queries 5")
    print("\n2. Test with MS_MARCO dataset:")
    print("   python comprehensive_experiment.py --experiment retrieval --dataset MS_MARCO --max-dataset-docs 100 --max-dataset-queries 5") 
    print("\n3. Test with SIFT dataset (use smaller limits for memory):")
    print("   python comprehensive_experiment.py --experiment retrieval --dataset SIFT --max-dataset-docs 50 --max-dataset-queries 3")
    print("\n💡 注意: SIFT 数据集较大，建议使用较小的 max-dataset-docs 参数以避免内存不足")

if __name__ == "__main__":
    test_dataset_integration()