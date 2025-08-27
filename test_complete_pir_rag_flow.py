#!/usr/bin/env python3
"""
Test PIR-RAG Complete Flow

This script tests the complete PIR-RAG flow with correct method signatures.
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
    
    def test_complete_pir_rag_flow():
        """Test the complete PIR-RAG flow with correct method signatures."""
        print("🧪 Testing Complete PIR-RAG Flow")
        print("=" * 50)
        
        # Create test data
        n_docs = 30
        embed_dim = 384
        np.random.seed(42)
        embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
        documents = [f"Document {i} about topic {i % 5}: {'content ' * (i % 10 + 1)}" for i in range(n_docs)]
        
        print(f"✅ Generated test data: {n_docs} docs, {embed_dim}D embeddings")
        
        # Setup PIR-RAG
        client = PIRRAGClient()
        server = PIRRAGServer()
        
        # Server setup (clustering)
        k_clusters = min(5, n_docs // 5)
        server_result = server.setup(embeddings, documents, k_clusters)
        print(f"✅ Server setup: {k_clusters} clusters, {server_result['setup_time']:.3f}s")
        
        # Client setup (centroids)
        client_result = client.setup(server.centroids)
        print(f"✅ Client setup: {len(server.centroids)} centroids, {client_result['setup_time']:.3f}s")
        
        # Test complete query flow
        query_np = np.random.randn(embed_dim).astype(np.float32)
        query_tensor = torch.tensor(query_np)
        
        print(f"\n🔍 Testing complete query flow:")
        
        # 1. Find relevant clusters
        try:
            relevant_clusters = client.find_relevant_clusters(query_tensor, top_k=3)
            print(f"✅ find_relevant_clusters: {relevant_clusters}")
        except Exception as e:
            print(f"❌ find_relevant_clusters failed: {e}")
            return False
        
        # 2. PIR retrieve
        try:
            urls, pir_metrics = client.pir_retrieve(relevant_clusters, server)
            print(f"✅ pir_retrieve: {len(urls)} URLs, {pir_metrics['total_server_time']:.3f}s server time")
        except Exception as e:
            print(f"❌ pir_retrieve failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. Rerank documents
        try:
            if urls:  # Only if we got some URLs
                final_results = client.rerank_documents(query_tensor, urls, server, top_k=5)
                print(f"✅ rerank_documents: {len(final_results)} final results")
            else:
                print("⚠️  No URLs to rerank")
        except Exception as e:
            print(f"❌ rerank_documents failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n🎉 Complete PIR-RAG flow test PASSED!")
        print(f"   Total query flow: find clusters → PIR retrieve → rerank")
        print(f"   All method signatures are correct!")
        return True
    
    if __name__ == "__main__":
        success = test_complete_pir_rag_flow()
        if success:
            print("\n✅ PIR-RAG complete flow working correctly!")
            print("   All method parameter orders are fixed.")
            print("   The experiments should now run without parameter errors.")
        else:
            print("\n❌ PIR-RAG flow test failed!")
            sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
