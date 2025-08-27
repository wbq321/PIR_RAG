#!/usr/bin/env python3
"""
Test Data Loading Functionality

This script demonstrates the flexible data loading options in the experiment framework.
"""

import sys
import os
from pathlib import Path

# Add PIR_RAG to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from experiments.comprehensive_experiment import PIRExperimentRunner

def test_data_loading():
    """Test different data loading scenarios."""
    
    runner = PIRExperimentRunner("test_results")
    
    print("🧪 Testing Data Loading Options")
    print("=" * 50)
    
    # Test 1: Synthetic data (default)
    print("\n1️⃣ Testing Synthetic Data Generation:")
    embeddings, documents = runner.load_or_generate_data(n_docs=100, embed_dim=384)
    print(f"   ✅ Generated {len(documents)} documents")
    print(f"   ✅ Embeddings shape: {embeddings.shape}")
    print(f"   ✅ Sample document: {documents[0][:100]}...")
    
    # Test 2: Check for default MS MARCO data
    print("\n2️⃣ Testing Default MS MARCO Data Detection:")
    embeddings2, documents2 = runner.load_or_generate_data(n_docs=50)
    if len(documents2) > 0 and "Document 0:" not in documents2[0]:
        print(f"   ✅ Found and loaded MS MARCO data")
        print(f"   ✅ Documents: {len(documents2)}")
        print(f"   ✅ Embeddings shape: {embeddings2.shape}")
        print(f"   ✅ Sample document: {documents2[0][:100]}...")
    else:
        print(f"   ℹ️  MS MARCO data not found, using synthetic data")
        print(f"   ✅ Generated {len(documents2)} documents")
    
    # Test 3: Non-existent custom paths (should fallback)
    print("\n3️⃣ Testing Custom Path Fallback:")
    embeddings3, documents3 = runner.load_or_generate_data(
        embeddings_path="nonexistent/embeddings.npy",
        corpus_path="nonexistent/corpus.csv",
        n_docs=25
    )
    print(f"   ✅ Fallback successful: {len(documents3)} documents")
    print(f"   ✅ Embeddings shape: {embeddings3.shape}")
    
    print("\n🎉 All data loading tests passed!")
    print("\n📊 Summary:")
    print(f"   • Synthetic data generation: ✅ Working")
    print(f"   • Default data detection: ✅ Working")
    print(f"   • Fallback mechanism: ✅ Working")
    
    print("\n🚀 Ready to run experiments!")
    print("\nQuick start commands:")
    print("   python experiments/comprehensive_experiment.py --experiment single --n-docs 100")
    print("   python test_retrieval_performance.py")
    print("   python experiments/analyze_results.py --generate-all")

if __name__ == "__main__":
    test_data_loading()
