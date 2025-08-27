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

        # Store original embeddings and URLs for semantic ranking
        self.document_embeddings = embeddings  # Keep original embeddings
        self.document_urls = [f"https://example.com/doc_{i}" for i in range(len(documents_text))]
        
        # Group documents by cluster but store URLs instead of document text
        clusters_urls = [[] for _ in range(n_clusters)]
        self.doc_to_cluster_map = {i: labels[i] for i in range(len(documents_text))}

        # Generate synthetic URLs for documents (realistic web search scenario)
        for i, text in enumerate(documents_text):
            synthetic_url = f"https://example.com/doc_{i}"
            clusters_urls[labels[i]].append(synthetic_url)

        # FIXED: Use simple cluster-based URL storage (like Tiptoe)
        # This avoids text encoding issues completely
        self.clusters_urls = clusters_urls
        
        # Store document indices for each cluster for easy URL generation  
        self.cluster_doc_indices = [[] for _ in range(n_clusters)]
        for doc_idx, cluster_idx in self.doc_to_cluster_map.items():
            self.cluster_doc_indices[cluster_idx].append(doc_idx)
        
        # No chunk encoding needed - we'll use direct URL PIR
        MAX_CHUNKS_HOLDER[0] = 1
        
        setup_time = time.perf_counter() - setup_start
        
        return {
            "setup_time": setup_time,
            "n_clusters": n_clusters,
            "n_documents": len(documents_text),
            "max_chunks": MAX_CHUNKS_HOLDER[0]
        }
    
    def handle_pir_query(self, encrypted_query: List, crypto_scheme) -> List:
        """
        Process PIR query and return encrypted URLs using homomorphic operations.
        
        FIXED: Return encrypted URLs that client must decrypt, preserving privacy.
        
        Args:
            encrypted_query: List of encrypted query values (one per cluster)
            crypto_scheme: SimpleLinearHomomorphicScheme instance
            
        Returns:
            List of encrypted URL data that client must decrypt
        """
        encrypted_results = []
        
        # For each cluster, compute encrypted result using homomorphic operations
        for cluster_idx in range(len(self.clusters_urls)):
            if cluster_idx < len(encrypted_query):
                cluster_urls = self.clusters_urls[cluster_idx]
                
                if cluster_urls:
                    # Encode the URLs as a simple concatenated string for this cluster
                    cluster_url_string = "|||".join(cluster_urls)
                    
                    # Convert to bytes and then to integers (using smaller chunks)
                    url_bytes = cluster_url_string.encode('utf-8')
                    
                    # Use 4-byte chunks to avoid overflow in linear scheme
                    chunk_results = []
                    for i in range(0, len(url_bytes), 4):
                        chunk = url_bytes[i:i+4]
                        # Pad to 4 bytes
                        while len(chunk) < 4:
                            chunk += b'\x00'
                        
                        chunk_int = int.from_bytes(chunk, 'big')
                        
                        # Homomorphic multiplication: encrypted_query[cluster_idx] * chunk_value
                        if chunk_int > 0:
                            encrypted_chunk = crypto_scheme.scalar_multiply(
                                encrypted_query[cluster_idx], chunk_int
                            )
                            chunk_results.append(encrypted_chunk)
                    
                    encrypted_results.extend(chunk_results)
                else:
                    # Empty cluster - add encrypted zero
                    encrypted_zero = crypto_scheme.scalar_multiply(encrypted_query[cluster_idx], 0)
                    encrypted_results.append(encrypted_zero)
        
        return encrypted_results
    
    def get_document_embeddings_for_urls(self, urls: List[str]) -> np.ndarray:
        """
        Get document embeddings for given URLs to enable semantic ranking.
        
        Args:
            urls: List of document URLs
            
        Returns:
            Array of document embeddings corresponding to the URLs
        """
        embeddings = []
        for url in urls:
            # Extract document index from URL (format: https://example.com/doc_X)
            try:
                doc_idx = int(url.split('_')[-1])
                if 0 <= doc_idx < len(self.document_embeddings):
                    embeddings.append(self.document_embeddings[doc_idx])
                else:
                    # Fallback: use zero embedding
                    embeddings.append(np.zeros(self.document_embeddings.shape[1]))
            except (ValueError, IndexError):
                # Fallback: use zero embedding
                embeddings.append(np.zeros(self.document_embeddings.shape[1]))
        
        return np.array(embeddings)
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the clustering setup."""
        return {
            "n_clusters": len(self.centroids) if self.centroids is not None else 0,
            "centroids_shape": self.centroids.shape if self.centroids is not None else None,
            "max_chunks": MAX_CHUNKS_HOLDER[0],
            "db_size": len(self.pir_db_by_chunk)
        }
