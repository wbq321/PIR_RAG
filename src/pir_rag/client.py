"""
PIR-RAG Client implementation.
"""

import sys
import time
from typing import List, Tuple, Dict, Any
import torch
from sentence_transformers import util
from phe import paillier

from .utils import decode_chunks_to_text


class PIRRAGClient:
    """
    Client-side implementation of the PIR-RAG system.
    
    Handles key generation, query encryption, and response decryption.
    """
    
    def __init__(self):
        self.centroids = None
        self.public_key = None
        self.private_key = None
    
    def setup(self, centroids: torch.Tensor, key_length: int = 2048) -> Dict[str, Any]:
        """
        Set up the client with cluster centroids and cryptographic keys.
        
        Args:
            centroids: Cluster centroids from the server
            key_length: Paillier key length in bits
            
        Returns:
            Dictionary with setup statistics
        """
        setup_start = time.perf_counter()
        
        print(f"  -> Setting up client with {len(centroids)} centroids, {key_length}-bit keys...")
        
        self.centroids = centroids
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        
        setup_time = time.perf_counter() - setup_start
        
        return {
            "setup_time": setup_time,
            "key_length": key_length,
            "n_centroids": len(centroids)
        }
    
    def find_relevant_clusters(self, query_embedding: torch.Tensor, top_k: int) -> List[int]:
        """
        Find the most relevant clusters for a query using semantic similarity.
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Number of top clusters to retrieve
            
        Returns:
            List of cluster indices sorted by relevance
        """
        if self.centroids is None:
            raise ValueError("Client not set up. Call setup() first.")
        
        similarities = util.cos_sim(query_embedding, self.centroids)[0]
        best_cluster_indices = torch.topk(similarities, k=min(top_k, len(self.centroids))).indices.tolist()
        
        return best_cluster_indices
    
    def generate_pir_query(self, cluster_idx: int, num_clusters: int) -> Tuple[List, int]:
        """
        Generate an encrypted PIR query vector for a specific cluster.
        
        Args:
            cluster_idx: Index of the target cluster
            num_clusters: Total number of clusters
            
        Returns:
            Tuple of (encrypted query vector, upload size in bytes)
        """
        if self.public_key is None:
            raise ValueError("Client not set up. Call setup() first.")
        
        # Create query vector: 1 for target cluster, 0 for others
        query_vec = []
        for i in range(num_clusters):
            if i == cluster_idx:
                query_vec.append(self.public_key.encrypt(1))
            else:
                query_vec.append(self.public_key.encrypt(0))
        
        # Calculate upload size
        upload_bytes = sum(sys.getsizeof(c.ciphertext()) for c in query_vec)
        
        return query_vec, upload_bytes
    
    def decode_pir_response(self, encrypted_chunks: List) -> Tuple[List[str], int]:
        """
        Decrypt and decode a PIR response into document URLs.
        
        Args:
            encrypted_chunks: List of encrypted response chunks
            
        Returns:
            Tuple of (document URLs, download size in bytes)
        """
        if self.private_key is None:
            raise ValueError("Client not set up. Call setup() first.")
        
        # Decrypt chunks
        retrieved_chunks = [self.private_key.decrypt(c) for c in encrypted_chunks]
        
        # Decode chunks to text and split into URLs
        retrieved_text = decode_chunks_to_text(retrieved_chunks)
        urls = list(filter(None, retrieved_text.split("|||")))
        
        # Calculate download size (much smaller for URLs vs documents)
        download_bytes = sum(sys.getsizeof(c.ciphertext()) for c in encrypted_chunks)
        
        return urls, download_bytes
    
    def pir_retrieve(self, server, cluster_indices: List[int]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Perform private retrieval of document URLs from multiple clusters.
        
        Args:
            server: PIRRAGServer instance
            cluster_indices: List of cluster indices to retrieve from
            
        Returns:
            Tuple of (retrieved URLs, performance metrics)
        """
        if self.centroids is None:
            raise ValueError("Client not set up. Call setup() first.")
        
        num_clusters = len(self.centroids)
        
        # Performance tracking
        metrics = {
            "total_query_gen_time": 0,
            "total_server_time": 0,
            "total_decode_time": 0,
            "total_upload_bytes": 0,
            "total_download_bytes": 0,
            "n_clusters_queried": len(cluster_indices)
        }
        
        candidate_urls = []

        for cluster_idx in cluster_indices:
            print(f"    [Client] Processing cluster {cluster_idx}...")
            
            # Generate encrypted query
            start_time = time.perf_counter()
            query_vec, upload_bytes = self.generate_pir_query(cluster_idx, num_clusters)
            metrics["total_query_gen_time"] += (time.perf_counter() - start_time)
            metrics["total_upload_bytes"] += upload_bytes
            
            # Server processing
            server_start_time = time.perf_counter()
            encrypted_chunks = server.handle_pir_query(query_vec, self.public_key)
            metrics["total_server_time"] += (time.perf_counter() - server_start_time)
            
            # Decode response
            start_time = time.perf_counter()
            urls, download_bytes = self.decode_pir_response(encrypted_chunks)
            metrics["total_decode_time"] += (time.perf_counter() - start_time)
            metrics["total_download_bytes"] += download_bytes
            
            candidate_urls.extend(urls)

        return candidate_urls, metrics
    
    def rerank_documents(self, query_embedding: torch.Tensor, urls: List[str], 
                        server, top_k: int = 10) -> List[str]:
        """
        Re-rank retrieved URLs using document embeddings from server.
        
        Args:
            query_embedding: Query embedding
            urls: List of candidate URLs
            server: PIRRAGServer instance to get document embeddings
            top_k: Number of top URLs to return
            
        Returns:
            List of top-k re-ranked URLs
        """
        if not urls:
            return []
        
        # Get document embeddings from server for semantic ranking
        doc_embeddings = server.get_document_embeddings_for_urls(urls)
        doc_embeddings = torch.tensor(doc_embeddings, dtype=torch.float32)
        
        # Normalize embeddings
        query_embedding = query_embedding / torch.norm(query_embedding)
        doc_embeddings = doc_embeddings / torch.norm(doc_embeddings, dim=1, keepdim=True)
        
        # Compute similarities and get top-k
        similarities = torch.mm(query_embedding.unsqueeze(0), doc_embeddings.T)[0]
        top_k_value = min(top_k, len(urls))
        
        if top_k_value > 0:
            top_k_indices = torch.topk(similarities, k=top_k_value).indices
            return [urls[i] for i in top_k_indices]
        
        return urls[:top_k_value]
