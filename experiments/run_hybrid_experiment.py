#!/usr/bin/env python3
"""
Run retrieval performance experiment with the new hybrid approach.

This script demonstrates running the comprehensive retrieval experiment
with the hybrid testing methodology integrated.
"""

import sys
import os
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "experiments"))

def run_hybrid_retrieval_experiment():
    """Run the hybrid retrieval experiment."""
    
    print("Running Hybrid Retrieval Performance Experiment")
    print("=" * 55)
    print()
    print("This experiment uses the hybrid approach:")
    print("• Phase 1: Plaintext simulation for accurate retrieval quality")
    print("• Phase 2: Real PIR operations for realistic performance metrics")
    print()
    
    # Import the runner
    try:
        from comprehensive_experiment import PIRExperimentRunner
    except ImportError as e:
        print(f"❌ Failed to import experiment runner: {e}")
        print("Make sure you're running from the PIR_RAG directory")
        return False
    
    # Create the runner
    runner = PIRExperimentRunner(output_dir="results")
    
    # Configuration
    config = {
        'n_docs': 500,              # Moderate dataset size
        'n_queries': 10,            # Reasonable number of queries  
        'embeddings_path': None,    # Use synthetic data
        'corpus_path': None,
        'embed_dim': 384,
        'top_k': 10,
        'pir_rag_k_clusters': None, # Use default (n_docs/20)
        'pir_rag_cluster_top_k': 3
    }
    
    print(f"Experiment Configuration:")
    print(f"  Documents: {config['n_docs']:,}")
    print(f"  Queries: {config['n_queries']}")
    print(f"  Top-K: {config['top_k']}")
    print(f"  Embedding Dimension: {config['embed_dim']}")
    print(f"  Data Type: {'Synthetic' if not config['embeddings_path'] else 'Real'}")
    print()
    
    try:
        # Run the experiment
        print("Starting hybrid retrieval experiment...")
        results = runner.run_retrieval_performance_experiment(**config)
        
        if results:
            print("\n🎉 Experiment completed successfully!")
            
            # Save results with timestamp
            runner.save_results(results, "hybrid_retrieval_test")
            
            # Print summary
            print(f"\nExperiment Summary:")
            experiment_info = results.get('experiment_info', {})
            print(f"  Timestamp: {experiment_info.get('timestamp', 'N/A')}")
            print(f"  Total Systems Tested: {len([k for k in results.keys() if k != 'experiment_info' and results[k] is not None])}")
            
            # Show which systems used hybrid approach
            hybrid_systems = []
            for system_name in ['pir_rag', 'graph_pir', 'tiptoe']:
                system_results = results.get(system_name)
                if system_results and system_results.get('hybrid_approach', False):
                    hybrid_systems.append(system_name.upper())
            
            if hybrid_systems:
                print(f"  Hybrid Approach Used: {', '.join(hybrid_systems)}")
            else:
                print(f"  ⚠️  No systems used hybrid approach")
            
            print(f"\nResults saved to: results/hybrid_retrieval_test_{experiment_info.get('timestamp', 'unknown')}.json")
            
            return True
            
        else:
            print("❌ Experiment returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    
    print("Hybrid PIR Retrieval Performance Testing")
    print("=" * 45)
    print()
    
    success = run_hybrid_retrieval_experiment()
    
    if success:
        print("\n✅ Hybrid retrieval experiment completed successfully!")
        print("\nNext steps:")
        print("1. Check the results/ directory for detailed output")
        print("2. Compare quality metrics (should be much higher than 2%/14%/42%)")
        print("3. Analyze performance metrics for realistic timing/communication costs")
        print("4. Run with larger datasets for comprehensive evaluation")
    else:
        print("\n❌ Experiment failed. Check error messages above.")


if __name__ == "__main__":
    main()
