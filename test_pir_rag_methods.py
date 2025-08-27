#!/usr/bin/env python3
"""
Test PIR-RAG Method Calls

This script tests that all PIR-RAG method calls work correctly with proper tensor types.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add PIR_RAG to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

try:
    from pir_rag import PIRRAGClient, PIRRAGServer
    
    def test_pir_rag_methods():
        """Test all PIR-RAG methods with correct parameter types."""
        print("🧪 Testing PIR-RAG Method Calls")
        print("=" * 50)
        
        # Create test data
        n_docs = 20
        embed_dim = 384
        np.random.seed(42)
        embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
        documents = [f"Document {i} about topic {i % 3}" for i in range(n_docs)]
        
        # Setup PIR-RAG
        client = PIRRAGClient()
        server = PIRRAGServer()
        
        # Server setup
        k_clusters = min(3, n_docs // 5)
        server.setup(embeddings, documents, k_clusters)
        print(f"✅ Server setup: {k_clusters} clusters")
        
        # Client setup
        client.setup(server.centroids)
        print(f"✅ Client setup: {len(server.centroids)} centroids")
        
        # Test query flow
        query_np = np.random.randn(embed_dim).astype(np.float32)
        query_tensor = torch.tensor(query_np)
        
        print(f"\n🔍 Testing query methods:")
        
        # Test find_relevant_clusters
        try:
            relevant_clusters = client.find_relevant_clusters(query_tensor, top_k=2)
            print(f"✅ find_relevant_clusters: {len(relevant_clusters)} clusters")
        except Exception as e:
            print(f"❌ find_relevant_clusters failed: {e}")
            return False
        
        # Test pir_retrieve
        try:
            urls, pir_metrics = client.pir_retrieve(relevant_clusters, server)
            print(f"✅ pir_retrieve: {len(urls)} URLs retrieved")
        except Exception as e:
            print(f"❌ pir_retrieve failed: {e}")
            return False
        
        # Test rerank_documents
        try:
            final_results = client.rerank_documents(query_tensor, urls, top_k=3)
            print(f"✅ rerank_documents: {len(final_results)} final results")
        except Exception as e:
            print(f"❌ rerank_documents failed: {e}")
            return False
        
        print(f"\n🎉 All PIR-RAG method tests PASSED!")
        return True
    
    if __name__ == "__main__":
        success = test_pir_rag_methods()
        if success:
            print("\n✅ PIR-RAG methods are working correctly!")
            print("   The experiments should now run without tensor type errors.")
        else:
            print("\n❌ Some PIR-RAG methods failed!")
            sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
