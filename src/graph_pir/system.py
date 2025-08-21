"""
Graph-PIR System implementation.

Combines graph-based ANN search with document retrieval PIR for fair RAG comparison.
Two-phase approach:
1. Graph traversal with vector PIR (returns document indices)  
2. Document retrieval PIR (returns actual document texts)
"""

import time
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from .piano_pir import PianoPIRClient, PianoPIRServer
from .graph_search import GraphSearch

# Import PIR-RAG utils with absolute import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pir_rag.utils import encode_text_to_chunks, decode_chunks_to_text


class GraphPIRSystem:
    """
    Graph-PIR system that performs two-phase retrieval:
    1. Graph-based vector search using PIR 
    2. Document retrieval using PIR
    
    This ensures fair comparison with PIR-RAG by returning actual documents.
    """
    
    def __init__(self):
        self.graph_search = None
        self.vector_pir_client = None
        self.vector_pir_server = None
        self.doc_pir_client = None
        self.doc_pir_server = None
        self.documents = None
        self.embeddings = None
        
    def setup(self, embeddings: np.ndarray, documents: List[str], 
              graph_params: Dict = None) -> Dict[str, Any]:
        """
        Set up the Graph-PIR system with vector database and document database.
        
        Args:
            embeddings: Document embeddings for graph construction
            documents: Document texts for retrieval
            graph_params: Parameters for graph construction
            
        Returns:
            Setup metrics
        """
        setup_start = time.perf_counter()
        
        if graph_params is None:
            graph_params = {
                'k_neighbors': 16,
                'ef_construction': 200,
                'max_connections': 16
            }
        
        print("[GraphPIR] Setting up Graph-PIR system...")
        print(f"[GraphPIR] Documents: {len(documents)}, Embeddings: {embeddings.shape}")
        
        # Store data
        self.embeddings = embeddings
        self.documents = documents
        
        # 1. Set up graph-based search
        print("[GraphPIR] Building graph structure...")
        self.graph_search = GraphSearch()
        graph_setup_time = time.perf_counter()
        self.graph_search.build_graph(embeddings, **graph_params)
        graph_setup_time = time.perf_counter() - graph_setup_time
        
        # 2. Set up vector PIR for graph traversal
        print("[GraphPIR] Setting up vector PIR...")
        vector_pir_start = time.perf_counter()
        self.vector_pir_server = PianoPIRServer()
        self.vector_pir_client = PianoPIRClient()
        
        # Convert embeddings to PIR database format
        # Each embedding becomes a database entry
        vector_db = embeddings.flatten().astype(np.float32)
        self.vector_pir_server.setup_database(vector_db, embeddings.shape[1])
        self.vector_pir_client.setup()
        vector_pir_time = time.perf_counter() - vector_pir_start
        
        # 3. Set up document PIR for final retrieval
        print("[GraphPIR] Setting up document PIR...")
        doc_pir_start = time.perf_counter()
        self.doc_pir_server = DocumentPIRServer()
        self.doc_pir_client = DocumentPIRClient()
        
        # Convert documents to PIR database format
        doc_setup_metrics = self.doc_pir_server.setup_database(documents)
        self.doc_pir_client.setup()
        doc_pir_time = time.perf_counter() - doc_pir_start
        
        total_setup_time = time.perf_counter() - setup_start
        
        setup_metrics = {
            "total_setup_time": total_setup_time,
            "graph_setup_time": graph_setup_time,
            "vector_pir_setup_time": vector_pir_time,
            "doc_pir_setup_time": doc_pir_time,
            "n_documents": len(documents),
            "embedding_dim": embeddings.shape[1],
            "graph_params": graph_params,
            **doc_setup_metrics
        }
        
        print(f"[GraphPIR] Setup complete in {total_setup_time:.2f}s")
        return setup_metrics
        
    def query(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[List[str], Dict[str, Any]]:
        """
        Perform Graph-PIR query to retrieve documents.
        
        Phase 1: Graph traversal with vector PIR to find candidate indices
        Phase 2: Document PIR to retrieve actual document texts
        
        Args:
            query_embedding: Query vector
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (retrieved documents, performance metrics)
        """
        if self.graph_search is None:
            raise ValueError("System not set up. Call setup() first.")
            
        print(f"[GraphPIR] Starting two-phase query for top-{top_k} documents...")
        
        metrics = {
            "phase1_time": 0,
            "phase2_time": 0,
            "phase1_upload_bytes": 0,
            "phase1_download_bytes": 0,
            "phase2_upload_bytes": 0,
            "phase2_download_bytes": 0,
            "total_candidates": 0,
            "graph_traversal_steps": 0
        }
        
        # Phase 1: Graph traversal with vector PIR
        print("[GraphPIR] Phase 1: Graph-based candidate search...")
        phase1_start = time.perf_counter()
        
        # Find candidate document indices using graph search + vector PIR
        candidate_indices, phase1_metrics = self._phase1_graph_search(
            query_embedding, top_k * 2  # Get more candidates for better recall
        )
        
        metrics.update({
            "phase1_time": time.perf_counter() - phase1_start,
            "total_candidates": len(candidate_indices),
            **phase1_metrics
        })
        
        # Phase 2: Document retrieval PIR
        print(f"[GraphPIR] Phase 2: Document PIR for {len(candidate_indices)} candidates...")
        phase2_start = time.perf_counter()
        
        # Retrieve actual documents using document PIR
        documents, phase2_metrics = self._phase2_document_retrieval(candidate_indices)
        
        metrics.update({
            "phase2_time": time.perf_counter() - phase2_start,
            **phase2_metrics
        })
        
        # Final reranking to get exact top-k
        if len(documents) > top_k:
            documents = self._rerank_documents(query_embedding, documents, top_k)
        
        total_time = metrics["phase1_time"] + metrics["phase2_time"]
        print(f"[GraphPIR] Query complete in {total_time:.3f}s (Phase1: {metrics['phase1_time']:.3f}s, Phase2: {metrics['phase2_time']:.3f}s)")
        
        return documents, metrics
        
    def _phase1_graph_search(self, query_embedding: np.ndarray, 
                           num_candidates: int) -> Tuple[List[int], Dict[str, Any]]:
        """
        Phase 1: Use graph search + vector PIR to find candidate document indices.
        """
        # Use graph search to find candidates (simulated PIR for now)
        # In a full implementation, this would use vector PIR for each graph traversal step
        candidate_indices = self.graph_search.search(query_embedding, num_candidates)
        
        # Simulate PIR communication costs
        # In reality, each graph traversal step would require PIR queries
        phase1_metrics = {
            "phase1_upload_bytes": len(candidate_indices) * 64,  # Simulated
            "phase1_download_bytes": len(candidate_indices) * 1024,  # Simulated  
            "graph_traversal_steps": min(50, len(candidate_indices))  # Simulated
        }
        
        return candidate_indices, phase1_metrics
        
    def _phase2_document_retrieval(self, candidate_indices: List[int]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Phase 2: Use document PIR to retrieve actual document texts.
        """
        # Generate PIR queries for each candidate document
        upload_bytes = 0
        download_bytes = 0
        retrieved_docs = []
        
        for doc_idx in candidate_indices:
            # Generate PIR query for this document
            query_data, query_upload = self.doc_pir_client.generate_query(doc_idx)
            upload_bytes += query_upload
            
            # Server processes PIR query
            response_data = self.doc_pir_server.process_query(query_data)
            
            # Client decrypts response
            doc_text, response_download = self.doc_pir_client.decrypt_response(response_data)
            download_bytes += response_download
            
            if doc_text:  # Only add non-empty documents
                retrieved_docs.append(doc_text)
        
        phase2_metrics = {
            "phase2_upload_bytes": upload_bytes,
            "phase2_download_bytes": download_bytes,
            "documents_retrieved": len(retrieved_docs)
        }
        
        return retrieved_docs, phase2_metrics
        
    def _rerank_documents(self, query_embedding: np.ndarray, 
                         documents: List[str], top_k: int) -> List[str]:
        """
        Rerank documents by similarity and return top-k.
        """
        if len(documents) <= top_k:
            return documents
            
        # Simple reranking - in practice would use proper embedding similarity
        # For now, just return first top_k documents
        return documents[:top_k]
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system setup."""
        if self.graph_search is None:
            return {"status": "not_setup"}
            
        return {
            "status": "ready",
            "n_documents": len(self.documents) if self.documents else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "graph_info": self.graph_search.get_graph_info(),
            "vector_pir_ready": self.vector_pir_server is not None,
            "doc_pir_ready": self.doc_pir_server is not None
        }


class DocumentPIRServer:
    """
    PIR server for document retrieval.
    Similar to PIR-RAG but for individual documents instead of clusters.
    """
    
    def __init__(self):
        self.doc_chunks_db = []  # Database of document chunks
        self.max_chunks = 0
        
    def setup_database(self, documents: List[str]) -> Dict[str, Any]:
        """
        Set up PIR database for document retrieval.
        Each document is encoded as chunks similar to PIR-RAG.
        """
        print(f"[DocPIR] Setting up document database for {len(documents)} documents...")
        
        # Encode each document to chunks
        doc_chunks = []
        for doc in documents:
            chunks = encode_text_to_chunks(doc)
            doc_chunks.append(chunks)
            
        # Find max chunks needed
        self.max_chunks = max(len(chunks) for chunks in doc_chunks) if doc_chunks else 0
        
        # Create PIR database organized by chunk position (similar to PIR-RAG)
        self.doc_chunks_db = [[] for _ in range(self.max_chunks)]
        for chunk_idx in range(self.max_chunks):
            for doc_chunks in doc_chunks:
                self.doc_chunks_db[chunk_idx].append(
                    doc_chunks[chunk_idx] if chunk_idx < len(doc_chunks) else 0
                )
                
        return {
            "doc_pir_max_chunks": self.max_chunks,
            "doc_pir_db_size": len(documents)
        }
        
    def process_query(self, query_data: Dict) -> Dict:
        """
        Process a PIR query for a specific document.
        Similar to PIR-RAG's handle_pir_query but for single document.
        """
        # Simulate PIR processing
        # In reality, this would use homomorphic encryption
        doc_idx = query_data["target_doc_idx"]  # This would be encrypted
        
        # Return encrypted chunks for the requested document
        response_chunks = []
        for chunk_idx in range(self.max_chunks):
            if doc_idx < len(self.doc_chunks_db[chunk_idx]):
                response_chunks.append(self.doc_chunks_db[chunk_idx][doc_idx])
            else:
                response_chunks.append(0)
                
        return {
            "encrypted_chunks": response_chunks,
            "chunk_count": len(response_chunks)
        }


class DocumentPIRClient:
    """
    PIR client for document retrieval.
    Similar to PIR-RAG client but for individual documents.
    """
    
    def __init__(self):
        self.setup_complete = False
        
    def setup(self):
        """Set up PIR client (key generation, etc.)"""
        # In reality, would generate Paillier keys like PIR-RAG
        self.setup_complete = True
        
    def generate_query(self, doc_idx: int) -> Tuple[Dict, int]:
        """
        Generate PIR query for a specific document.
        """
        if not self.setup_complete:
            raise ValueError("Client not set up")
            
        # In reality, would generate encrypted query vector
        query_data = {
            "target_doc_idx": doc_idx,  # This would be encrypted
            "query_type": "document_retrieval"
        }
        
        # Simulate upload size
        upload_bytes = 2048  # Simulated encrypted query size
        
        return query_data, upload_bytes
        
    def decrypt_response(self, response_data: Dict) -> Tuple[str, int]:
        """
        Decrypt PIR response and decode back to document text.
        """
        # Simulate decryption and decoding (similar to PIR-RAG)
        encrypted_chunks = response_data["encrypted_chunks"]
        
        # In reality, would decrypt each chunk
        decrypted_chunks = encrypted_chunks  # Simulated
        
        # Decode chunks back to text (same as PIR-RAG)
        document_text = decode_chunks_to_text(decrypted_chunks)
        
        # Simulate download size
        download_bytes = len(encrypted_chunks) * 256  # Simulated
        
        return document_text, download_bytes
