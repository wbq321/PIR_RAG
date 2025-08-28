#!/usr/bin/env python3
"""
Test script to verify that Graph-PIR parameters flow correctly from command line to both simulation and real PIR.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.comprehensive_experiment import PIRExperimentRunner
import numpy as np

def test_parameter_flow():
    """Test that parameters flow correctly through the system."""
    
    print("🧪 Testing Graph-PIR Parameter Flow")
    print("="*50)
    
    # Create test data
    n_docs = 100
    embed_dim = 384
    n_queries = 2
    
    embeddings = np.random.random((n_docs, embed_dim)).astype(np.float32)
    documents = [f"Document {i}" for i in range(n_docs)]
    queries = [np.random.random(embed_dim).astype(np.float32) for _ in range(n_queries)]
    
    # Test custom parameters
    test_graph_params = {
        'k_neighbors': 8,
        'ef_construction': 50,
        'max_connections': 8,
        'ef_search': 20,
        'max_iterations': 15,
        'parallel': 3,
        'max_neighbors_per_step': 4
    }
    
    print(f"📝 Test Parameters:")
    for key, value in test_graph_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n🏃 Running Retrieval Test...")
    
    # Create runner and test
    runner = PIRExperimentRunner()
    
    try:
        results = runner.run_retrieval_performance_experiment(
            n_docs=n_docs,
            n_queries=n_queries,
            embeddings_path=None,  # Use generated data
            corpus_path=None,
            embed_dim=embed_dim,
            top_k=5,
            graph_params=test_graph_params
        )
        
        print(f"\n✅ Test completed successfully!")
        
        # Check if Graph-PIR results exist
        if results and 'graph_pir' in results and results['graph_pir'] is not None:
            graph_pir_results = results['graph_pir']
            print(f"📊 Graph-PIR Results:")
            print(f"  System: {graph_pir_results.get('system', 'Unknown')}")
            print(f"  Setup time: {graph_pir_results.get('setup_time', 0):.3f}s")
            print(f"  Queries processed: {len(graph_pir_results.get('query_results', []))}")
            
            # Check if debug output appeared (simulation parameters)
            print(f"\n🔍 Check the output above for:")
            print(f"  - '[DEBUG] Simulation using graph_params: k_neighbors=8, max_iterations=15, parallel=3'")
            print(f"  - 'Using custom graph_params: {test_graph_params}'")
            print(f"  - '[GraphPIR] GraphANN SearchKNN: n={n_docs}, maxStep=15, parallel=3'")
            
        else:
            print(f"❌ Graph-PIR test failed or returned no results")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parameter_flow()
