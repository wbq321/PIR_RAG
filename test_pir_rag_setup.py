#!/usr/bin/env python3
"""
Test PIR-RAG Setup Fix

This script tests that the PIR-RAG setup works correctly with the new flow.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add PIR_RAG to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

try:
    from pir_rag import PIRRAGClient, PIRRAGServer
    
    def test_pir_rag_setup():
        """Test the corrected PIR-RAG setup flow."""
        print("🧪 Testing PIR-RAG Setup Flow")
        print("=" * 40)
        
        # Create test data
        n_docs = 50
        embed_dim = 384
        np.random.seed(42)
        embeddings = np.random.randn(n_docs, embed_dim).astype(np.float32)
        documents = [f"Document {i} about topic {i % 5}" for i in range(n_docs)]
        
        print(f"✅ Generated test data: {n_docs} docs, {embed_dim}D embeddings")
        
        # Test the setup flow
        try:
            # Step 1: Create client and server
            client = PIRRAGClient()
            server = PIRRAGServer()
            print("✅ Created PIR-RAG client and server")
            
            # Step 2: Server setup (does clustering)
            k_clusters = min(5, n_docs // 10)
            server_result = server.setup(embeddings, documents, k_clusters)
            print(f"✅ Server setup completed: {k_clusters} clusters")
            print(f"   Setup time: {server_result['setup_time']:.4f}s")
            
            # Step 3: Client setup (gets centroids from server)
            client_result = client.setup(server.centroids)
            print(f"✅ Client setup completed")
            print(f"   Setup time: {client_result['setup_time']:.4f}s")
            
            # Step 4: Test a simple query
            query = np.random.randn(embed_dim).astype(np.float32)
            relevant_clusters = client.find_relevant_clusters(query, top_clusters=2)
            print(f"✅ Query test successful: found {len(relevant_clusters)} relevant clusters")
            
            print("\n🎉 PIR-RAG setup test PASSED!")
            return True
            
        except Exception as e:
            print(f"\n❌ PIR-RAG setup test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if __name__ == "__main__":
        success = test_pir_rag_setup()
        if not success:
            sys.exit(1)
        
        print("\n✅ PIR-RAG setup is working correctly!")
        print("   The comprehensive experiment should now work properly.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the PIR_RAG directory and src/ is accessible")
    sys.exit(1)
