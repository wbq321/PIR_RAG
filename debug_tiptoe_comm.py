#!/usr/bin/env python3
"""
Debug script to check what communication costs Tiptoe is returning.
"""

import numpy as np
import sys
import os

# Add PIR_RAG to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from tiptoe import TiptoeSystem

def debug_tiptoe_communication():
    """Test what communication metrics Tiptoe returns."""
    print("Debugging Tiptoe communication costs...")
    
    # Create small test data
    n_docs = 50
    embed_dim = 16
    np.random.seed(42)
    embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
    documents = [f"Document {i}: Test content for document {i}" for i in range(n_docs)]
    
    # Test with Tiptoe
    tiptoe_params = {
        'k_clusters': 5,
        'use_real_crypto': False  # Use simulated for debugging
    }
    
    print(f"Testing with parameters: {tiptoe_params}")
    print(f"Documents: {n_docs}, Embedding dim: {embed_dim}")
    
    try:
        # Create system and setup
        system = TiptoeSystem()
        setup_metrics = system.setup(embeddings, documents, **tiptoe_params)
        
        print(f"✅ Setup completed")
        print(f"Setup metrics: {setup_metrics}")
        
        # Test a query
        query_embedding = np.random.randn(embed_dim).astype(np.float32)
        urls, query_metrics = system.query(query_embedding, top_k=5)
        
        print(f"✅ Query completed")
        print(f"Returned {len(urls)} URLs")
        print("\n📊 Communication Metrics:")
        print(f"  upload_bytes: {query_metrics.get('upload_bytes', 'NOT FOUND')}")
        print(f"  download_bytes: {query_metrics.get('download_bytes', 'NOT FOUND')}")
        print(f"  total_upload_bytes: {query_metrics.get('total_upload_bytes', 'NOT FOUND')}")
        print(f"  total_download_bytes: {query_metrics.get('total_download_bytes', 'NOT FOUND')}")
        print(f"  total_communication: {query_metrics.get('total_communication', 'NOT FOUND')}")
        print(f"  pir_communication: {query_metrics.get('pir_communication', 'NOT FOUND')}")
        
        print("\n⏱️  Timing Metrics:")
        print(f"  ranking_time: {query_metrics.get('ranking_time', 'NOT FOUND')}")
        print(f"  retrieval_time: {query_metrics.get('retrieval_time', 'NOT FOUND')}")
        print(f"  cluster_selection_time: {query_metrics.get('cluster_selection_time', 'NOT FOUND')}")
        print(f"  total_ranking_time: {query_metrics.get('total_ranking_time', 'NOT FOUND')}")
        
        print("\n🔍 Debug Metrics:")
        debug_ranking = query_metrics.get('debug_ranking_metrics', {})
        debug_retrieval = query_metrics.get('debug_retrieval_metrics', {})
        print(f"  debug_ranking_metrics: {debug_ranking}")
        print(f"  debug_retrieval_metrics: {debug_retrieval}")
        
        print("\n📋 All Available Metrics:")
        for key, value in query_metrics.items():
            if isinstance(value, (int, float)) and 'time' not in key.lower():
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Debugging Tiptoe Communication Costs")
    print("=" * 60)
    
    success = debug_tiptoe_communication()
    
    if success:
        print("\n🎉 Debug completed successfully!")
    else:
        print("\n💥 Debug failed. Check the output above.")
