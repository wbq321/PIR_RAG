#!/usr/bin/env python3
"""
Quick test to see if comprehensive_experiment.py imports and runs correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

try:
    from comprehensive_experiment import PIRExperimentRunner
    print("✅ Successfully imported PIRExperimentRunner")
    
    # Test with very small parameters
    runner = PIRExperimentRunner()
    print("✅ Created PIRExperimentRunner instance")
    
    # Try loading synthetic data
    embeddings, documents = runner.load_or_generate_data(n_docs=10, embed_dim=32, seed=42)
    print(f"✅ Generated test data: {len(documents)} docs, {embeddings.shape} embeddings")
    
    # Try running basic experiment with minimal parameters
    result = runner.run_basic_experiments(n_docs=10, n_queries=2, embed_dim=32)
    print("✅ Basic experiment completed successfully!")
    print("Results keys:", list(result.keys()))
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
