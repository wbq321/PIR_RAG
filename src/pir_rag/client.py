"""
PIR-RAG Client implementation with LinearHomomorphicScheme.
"""

import sys
import time
from typing import List, Tuple, Dict, Any
import torch
from sentence_transformers import util

# Import fast linear homomorphic scheme instead of Paillier
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tiptoe.crypto_fixed import SimpleLinearHomomorphicScheme

from .utils import decode_chunks_to_text


class PIRRAGClient:
    """
    Client-side implementation of the PIR-RAG system.
    
    Handles key generation, query encryption, and response decryption.
    """
    
    def __init__(self):
        self.centroids = None
        # Use SimpleLinearHomomorphicScheme instead of Paillier
        self.crypto_scheme = None
    
    def setup(self, centroids: torch.Tensor, key_length: int = 2048) -> Dict[str, Any]:
        """
        Set up the client with cluster centroids and fast linear homomorphic encryption.
        
        Args:
            centroids: Cluster centroids from the server
            key_length: Key length (kept for compatibility, but SimpleLinearHomomorphicScheme doesn't use it)
            
        Returns:
            Dictionary with setup statistics
        """
        setup_start = time.perf_counter()
        
        print(f"  -> Setting up client with {len(centroids)} centroids, fast linear scheme...")
        
        self.centroids = centroids
        # Initialize fast linear homomorphic scheme
        self.crypto_scheme = SimpleLinearHomomorphicScheme()
        
        setup_time = time.perf_counter() - setup_start
        
        return {
            "setup_time": setup_time,
            "scheme": "SimpleLinearHomomorphic",
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
        
        print(f"[PIR-RAG] DEBUG: Cluster selection - similarities: {similarities[:5]}... (first 5)")
        print(f"[PIR-RAG] DEBUG: Selected cluster indices: {best_cluster_indices}")
        
        return best_cluster_indices
    
    def generate_pir_query(self, cluster_idx: int, num_clusters: int) -> Tuple[List, int]:
        """
        Generate an encrypted PIR query vector for a specific cluster using fast linear scheme.
        
        Args:
            cluster_idx: Index of the target cluster
            num_clusters: Total number of clusters
            
        Returns:
            Tuple of (encrypted query vector, upload size in bytes)
        """
        if self.crypto_scheme is None:
            raise ValueError("Client not set up. Call setup() first.")
        
        print(f"[PIR-RAG] DEBUG: Generating PIR query for cluster {cluster_idx}/{num_clusters}")
        
        # Create query vector: 1 for target cluster, 0 for others
        query_vec = []
        for i in range(num_clusters):
            if i == cluster_idx:
                encrypted_val = self.crypto_scheme.encrypt(1)
                query_vec.append(encrypted_val)
            else:
                encrypted_val = self.crypto_scheme.encrypt(0)
                query_vec.append(encrypted_val)
        
        print(f"[PIR-RAG] DEBUG: PIR query vector created, length: {len(query_vec)}")
        print(f"[PIR-RAG] DEBUG: Sample encrypted values: {query_vec[0]}, {query_vec[1]}")
        
        # Calculate upload size (much smaller than Paillier)
        upload_bytes = len(query_vec) * 32  # Estimate for simple scheme
        
        return query_vec, upload_bytes
    
    def decode_pir_response(self, encrypted_chunks: List) -> Tuple[List[str], int]:
        """
        Decrypt and decode a PIR response into document URLs using fast linear scheme.
        
        Args:
            encrypted_chunks: List of encrypted response chunks
            
        Returns:
            Tuple of (document URLs, download size in bytes)
        """
        if self.crypto_scheme is None:
            raise ValueError("Client not set up. Call setup() first.")
        
        # Decrypt chunks using fast scheme
        retrieved_chunks = [self.crypto_scheme.decrypt(c) for c in encrypted_chunks]
        
        # Decode chunks to text and split into URLs
        retrieved_text = decode_chunks_to_text(retrieved_chunks)
        urls = list(filter(None, retrieved_text.split("|||")))
        
        # Calculate download size (much smaller than Paillier)
        download_bytes = len(encrypted_chunks) * 32  # Estimate for simple scheme
        
        return urls, download_bytes
    
    def pir_retrieve(self, server, cluster_indices: List[int]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Perform private retrieval of document URLs from multiple clusters.
        
        FIXED: Client properly decrypts encrypted URL data from server.
        
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
            
            # Server processing returns encrypted URL data
            server_start_time = time.perf_counter()
            encrypted_chunks = server.handle_pir_query(query_vec, self.crypto_scheme)
            metrics["total_server_time"] += (time.perf_counter() - server_start_time)
            
            # Client decrypts the encrypted URL data
            start_time = time.perf_counter()
            urls = self._decrypt_url_chunks(encrypted_chunks)
            metrics["total_decode_time"] += (time.perf_counter() - start_time)
            metrics["total_download_bytes"] += len(str(encrypted_chunks).encode('utf-8'))
            
            candidate_urls.extend(urls)

        return candidate_urls, metrics
    
    def _decrypt_url_chunks(self, encrypted_chunks: List) -> List[str]:
        """
        Decrypt encrypted URL chunks back to URL strings.
        
        Args:
            encrypted_chunks: List of encrypted URL chunk data
            
        Returns:
            List of decrypted URLs
        """
        print(f"[PIR-RAG] DEBUG: Decrypting {len(encrypted_chunks)} URL chunks")
        
        try:
            # Decrypt each chunk
            decrypted_ints = []
            for i, chunk in enumerate(encrypted_chunks):
                try:
                    decrypted_val = self.crypto_scheme.decrypt(chunk)
                    if decrypted_val > 0:  # Skip zero/empty chunks
                        decrypted_ints.append(decrypted_val)
                        if i < 3:  # Show first few for debugging
                            print(f"[PIR-RAG] DEBUG: Chunk {i}: decrypted to {decrypted_val}")
                except Exception as e:
                    print(f"[PIR-RAG] DEBUG: Failed to decrypt chunk {i}: {e}")
                    continue
            
            print(f"[PIR-RAG] DEBUG: Successfully decrypted {len(decrypted_ints)} chunks")
            
            # Convert integers back to bytes and then to text
            if not decrypted_ints:
                print(f"[PIR-RAG] DEBUG: No valid decrypted chunks found")
                return []
            
            byte_data = b''
            for int_val in decrypted_ints:
                try:
                    # Convert back to 4-byte chunks
                    chunk_bytes = int_val.to_bytes(4, 'big')
                    byte_data += chunk_bytes
                except (OverflowError, ValueError) as e:
                    print(f"[PIR-RAG] DEBUG: Failed to convert int {int_val} to bytes: {e}")
                    continue
            
            print(f"[PIR-RAG] DEBUG: Reconstructed {len(byte_data)} bytes")
            
            # Remove null bytes and decode
            byte_data = byte_data.rstrip(b'\x00')
            url_string = byte_data.decode('utf-8', errors='ignore')
            print(f"[PIR-RAG] DEBUG: Decoded string: '{url_string[:100]}...' (first 100 chars)")
            
            # Split by separator and filter out empty strings
            urls = [url.strip() for url in url_string.split('|||') if url.strip()]
            print(f"[PIR-RAG] DEBUG: Final URLs: {urls}")
            
            return urls
            
        except Exception as e:
            print(f"[PIR-RAG] WARNING: URL decryption failed: {e}")
            return []
    
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
