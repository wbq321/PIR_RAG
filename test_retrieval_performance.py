#!/usr/bin/env python3
"""
Simple Retrieval Performance Test Runner

Tests the retrieval quality and speed of all three PIR systems.
Measures precision, recall, NDCG, and throughput.
"""

import sys
import os
import time

# Add PIR_RAG to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def main():
    print("🔍 PIR Systems Retrieval Performance Test")
    print("=" * 50)
    print("This test measures:")
    print("✓ Query latency (time per query)")
    print("✓ Retrieval quality (precision, recall, NDCG)")
    print("✓ Communication efficiency (bytes per query)")
    print("✓ Throughput (queries per second)")
    print("✓ Scalability across dataset sizes")
    print()
    
    # Test different configurations
    test_configs = [
        {"n_docs": 500, "n_queries": 10, "name": "Small Test"},
        {"n_docs": 1000, "n_queries": 20, "name": "Medium Test"},
        {"n_docs": 2000, "n_queries": 20, "name": "Large Test"}
    ]
    
    for config in test_configs:
        print(f"🚀 Running {config['name']}: {config['n_docs']} docs, {config['n_queries']} queries")
        
        try:
            # Import here to avoid path issues
            from experiments.test_retrieval_performance import RetrievalPerformanceTester
            
            tester = RetrievalPerformanceTester()
            results = tester.run_comprehensive_retrieval_test(
                n_docs=config['n_docs'], 
                n_queries=config['n_queries']
            )
            
            tester.print_performance_summary(results)
            print("\n" + "-" * 80 + "\n")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("✅ Retrieval performance testing complete!")
    print("📊 Check the results/ directory for detailed JSON outputs")
    print("📈 You can also run the comprehensive experiment framework:")
    print("   python experiments/comprehensive_experiment.py --experiment all")

if __name__ == "__main__":
    main()
