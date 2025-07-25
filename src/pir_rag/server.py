"""
PIR-RAG Server implementation.
"""

import time
from typing import List, Dict, Any
import torch
import numpy as np
from sklearn.cluster import KMeans

from .utils import encode_text_to_chunks, MAX_CHUNKS_HOLDER


class PIRRAGServer:
    """
    Server-side implementation of the PIR-RAG system.
    
    Handles document clustering, PIR database setup, and encrypted query processing.
    """
    
    def __init__(self):
        self.centroids = None
        self.pir_db_by_chunk = []
        self.doc_to_cluster_map = {}
    
    def setup(self, embeddings: np.ndarray, documents_text: List[str], n_clusters: int) -> Dict[str, Any]:
        """
        Set up the server with document clustering and PIR database.
        
        Args:
            embeddings: Document embeddings array
            documents_text: List of document texts
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with setup statistics
        """
        setup_start = time.perf_counter()
        
        print(f"  -> Starting server setup with {len(documents_text)} docs, {n_clusters} clusters...")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto', verbose=0)
        labels = kmeans.fit_predict(embeddings)
        self.centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        # Group documents by cluster
        clusters_text = [[] for _ in range(n_clusters)]
        self.doc_to_cluster_map = {i: labels[i] for i in range(len(documents_text))}

        for i, text in enumerate(documents_text):
            clusters_text[labels[i]].append(text)

        # Encode clusters for PIR
        chunked_clusters = [encode_text_to_chunks("|||".join(c)) for c in clusters_text]
        
        MAX_CHUNKS_HOLDER[0] = max(len(c) for c in chunked_clusters if c) if any(chunked_clusters) else 0

        # Create PIR database organized by chunk position
        self.pir_db_by_chunk = [[] for _ in range(MAX_CHUNKS_HOLDER[0])]
        for chunk_idx in range(MAX_CHUNKS_HOLDER[0]):
            for cluster_chunks in chunked_clusters:
                self.pir_db_by_chunk[chunk_idx].append(
                    cluster_chunks[chunk_idx] if chunk_idx < len(cluster_chunks) else 0
                )
        
        setup_time = time.perf_counter() - setup_start
        
        return {
            "setup_time": setup_time,
            "n_clusters": n_clusters,
            "n_documents": len(documents_text),
            "max_chunks": MAX_CHUNKS_HOLDER[0]
        }
    
    def handle_pir_query(self, encrypted_query: List, public_key) -> List:
        """
        Process an encrypted PIR query and return encrypted response.
        
        Args:
            encrypted_query: List of encrypted query values
            public_key: Paillier public key for homomorphic operations
            
        Returns:
            List of encrypted chunk responses
        """
        print(f"    [Server] Processing PIR query with {len(encrypted_query)} clusters, {MAX_CHUNKS_HOLDER[0]} chunks...")
        
        # Perform homomorphic computation for each chunk position
        encrypted_response = []
        for chunk_db in self.pir_db_by_chunk:
            # Compute sum of (chunk_value * encrypted_query_bit) for all clusters
            chunk_result = public_key.encrypt(0)  # Start with encrypted zero
            for i, chunk_value in enumerate(chunk_db):
                if i < len(encrypted_query):
                    chunk_result += chunk_value * encrypted_query[i]
            encrypted_response.append(chunk_result)
        
        return encrypted_response
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the clustering setup."""
        return {
            "n_clusters": len(self.centroids) if self.centroids is not None else 0,
            "centroids_shape": self.centroids.shape if self.centroids is not None else None,
            "max_chunks": MAX_CHUNKS_HOLDER[0],
            "db_size": len(self.pir_db_by_chunk)
        }
