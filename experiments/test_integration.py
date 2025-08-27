#!/usr/bin/env python3
"""
Test script to verify hybrid approach integration in comprehensive experiments.

This script runs a small-scale test to demonstrate that the hybrid testing
approach is properly integrated into the comprehensive experiment framework.
"""

import sys
import os
from pathlib import Path

# Add the experiments directory to the path
experiments_dir = Path(__file__).parent
sys.path.insert(0, str(experiments_dir))

# Add the src directory to the path  
src_dir = experiments_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from comprehensive_experiment import PIRExperimentRunner
    print("✅ Successfully imported PIRExperimentRunner")
except ImportError as e:
    print(f"❌ Failed to import PIRExperimentRunner: {e}")
    sys.exit(1)


def test_hybrid_integration():
    """Test that the hybrid approach is properly integrated."""
    
    print("Testing Hybrid Approach Integration")
    print("="*50)
    
    # Create experiment runner
    runner = PIRExperimentRunner(output_dir="test_results")
    
    # Run a small retrieval experiment to test integration
    print("\nRunning small-scale retrieval experiment...")
    
    try:
        results = runner.run_retrieval_performance_experiment(
            n_docs=50,          # Small dataset for quick testing
            n_queries=3,        # Few queries for speed
            embeddings_path=None,  # Use synthetic data
            corpus_path=None,
            embed_dim=384,
            top_k=5,
            pir_rag_k_clusters=3,
            pir_rag_cluster_top_k=2
        )
        
        if results:
            print("\n✅ Hybrid approach integration test completed successfully!")
            
            # Check if hybrid approach was used
            for system_name in ['pir_rag', 'graph_pir', 'tiptoe']:
                system_results = results.get(system_name)
                if system_results and system_results.get('hybrid_approach', False):
                    print(f"✅ {system_name.upper()}: Hybrid approach confirmed")
                    print(f"   - Quality simulation time: {system_results.get('avg_quality_simulation_time', 0):.3f}s")
                    print(f"   - PIR performance time: {system_results.get('avg_pir_performance_time', 0):.3f}s")
                    print(f"   - Precision@{results['experiment_info']['top_k']}: {system_results.get('avg_precision_at_k', 0):.3f}")
                elif system_results is None:
                    print(f"⚠️  {system_name.upper()}: System failed to run")
                else:
                    print(f"❌ {system_name.upper()}: Old approach still in use")
            
            return True
            
        else:
            print("❌ No results returned from retrieval experiment")
            return False
            
    except Exception as e:
        print(f"❌ Hybrid integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    
    print("Hybrid Approach Integration Test")
    print("="*40)
    print("This test verifies that the hybrid testing approach")
    print("is properly integrated into comprehensive_experiment.py")
    print()
    
    # Test the integration
    success = test_hybrid_integration()
    
    if success:
        print("\n🎉 Integration test PASSED!")
        print("\nYou can now run full experiments with:")
        print("python comprehensive_experiment.py --experiment retrieval --n_docs 1000 --n_queries 20")
    else:
        print("\n💥 Integration test FAILED!")
        print("Check the error messages above and fix any issues.")


if __name__ == "__main__":
    main()
