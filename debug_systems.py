#!/usr/bin/env python3
"""
Quick test to check if Graph-PIR and Tiptoe systems can be imported and instantiated
"""

import sys
import os
import numpy as np

# Add PIR_RAG to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_imports():
    """Test if all systems can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from pir_rag import PIRRAGClient, PIRRAGServer
        print("  ✅ PIR-RAG imported successfully")
    except Exception as e:
        print(f"  ❌ PIR-RAG import failed: {e}")
    
    try:
        from graph_pir import GraphPIRSystem
        print("  ✅ Graph-PIR imported successfully")
    except Exception as e:
        print(f"  ❌ Graph-PIR import failed: {e}")
        return False
    
    try:
        from tiptoe import TiptoeSystem
        print("  ✅ Tiptoe imported successfully")
    except Exception as e:
        print(f"  ❌ Tiptoe import failed: {e}")
        return False
    
    return True

def test_instantiation():
    """Test if systems can be instantiated."""
    print("\n🧪 Testing instantiation...")
    
    try:
        from graph_pir import GraphPIRSystem
        system = GraphPIRSystem()
        print("  ✅ GraphPIRSystem instantiated successfully")
    except Exception as e:
        print(f"  ❌ GraphPIRSystem instantiation failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        from tiptoe import TiptoeSystem
        system = TiptoeSystem()
        print("  ✅ TiptoeSystem instantiated successfully")
    except Exception as e:
        print(f"  ❌ TiptoeSystem instantiation failed: {e}")
        import traceback
        traceback.print_exc()

def test_minimal_setup():
    """Test minimal setup with tiny dataset."""
    print("\n🧪 Testing minimal setup...")
    
    # Create minimal test data
    embeddings = np.random.randn(10, 16).astype(np.float32)
    documents = [f"Test document {i}" for i in range(10)]
    
    try:
        from graph_pir import GraphPIRSystem
        system = GraphPIRSystem()
        graph_params = {'k_neighbors': 4, 'ef_construction': 50, 'max_connections': 4}
        result = system.setup(embeddings, documents, graph_params=graph_params)
        print(f"  ✅ GraphPIR setup completed: {result}")
    except Exception as e:
        print(f"  ❌ GraphPIR setup failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        from tiptoe import TiptoeSystem
        system = TiptoeSystem()
        result = system.setup(embeddings, documents, k_clusters=2, use_real_crypto=False)
        print(f"  ✅ Tiptoe setup completed: {result}")
    except Exception as e:
        print(f"  ❌ Tiptoe setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Debugging Graph-PIR and Tiptoe Issues\n")
    
    if test_imports():
        test_instantiation()
        test_minimal_setup()
    
    print("\n✅ Debug test completed!")
