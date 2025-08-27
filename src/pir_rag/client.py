"""
PIR-RAG Client implementation with LinearHomomorphicScheme.
"""

import sys
import time
from typing import List, Tuple, Dict, Any
import torch
import numpy as np
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
        
        cluster_selection_start = time.perf_counter()
        
        similarities = util.cos_sim(query_embedding, self.centroids)[0]
        best_cluster_indices = torch.topk(similarities, k=min(top_k, len(self.centroids))).indices.tolist()
        
        cluster_selection_time = time.perf_counter() - cluster_selection_start
        
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
        
        # Create query vector: 1 for target cluster, 0 for others
        query_vec = []
        for i in range(num_clusters):
            if i == cluster_idx:
                encrypted_val = self.crypto_scheme.encrypt(1)
                query_vec.append(encrypted_val)
            else:
                encrypted_val = self.crypto_scheme.encrypt(0)
                query_vec.append(encrypted_val)
        
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
    
    def pir_retrieve(self, cluster_indices: List[int], server) -> Tuple[List[Tuple[str, np.ndarray]], Dict[str, Any]]:
        """
        Perform private retrieval of document URLs and embeddings from multiple clusters.
        
        PRIVACY FIX: Now returns both URLs and embeddings from PIR response
        to avoid revealing which documents user is interested in.
        
        Args:
            cluster_indices: List of cluster indices to query
            server: PIRRAGServer instance
            
        Returns:
            Tuple of (list of (URL, embedding) tuples, performance metrics)
        """
        if self.centroids is None:
            raise ValueError("Client not set up. Call setup() first.")

        pir_retrieve_start = time.perf_counter()
        
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
        
        candidate_docs = []  # Now stores (URL, embedding) tuples

        for cluster_idx in cluster_indices:
            cluster_start = time.perf_counter()
            
            # Generate encrypted query
            query_gen_start = time.perf_counter()
            query_vec, upload_bytes = self.generate_pir_query(cluster_idx, num_clusters)
            query_gen_time = time.perf_counter() - query_gen_start
            metrics["total_query_gen_time"] += query_gen_time
            metrics["total_upload_bytes"] += upload_bytes
            
            # Server processing returns encrypted URL+embedding data
            server_start_time = time.perf_counter()
            encrypted_chunks = server.handle_pir_query(query_vec, self.crypto_scheme)
            server_time = time.perf_counter() - server_start_time
            metrics["total_server_time"] += server_time
            
            # Client decrypts the encrypted URL+embedding data
            decrypt_start_time = time.perf_counter()
            urls, embeddings = self._decrypt_url_chunks(encrypted_chunks)
            decrypt_time = time.perf_counter() - decrypt_start_time
            metrics["total_decode_time"] += decrypt_time
            metrics["total_download_bytes"] += len(str(encrypted_chunks).encode('utf-8'))
            
            # Combine URLs and embeddings into tuples
            for url, embedding in zip(urls, embeddings):
                candidate_docs.append((url, embedding))

        total_pir_time = time.perf_counter() - pir_retrieve_start

        return candidate_docs, metrics
    
    def _decrypt_url_chunks(self, encrypted_chunks: List) -> Tuple[List[str], List[np.ndarray]]:
        """
        Decrypt encrypted URL+embedding chunks back to URLs and embeddings.
        
        PRIVACY FIX: Now returns both URLs and embeddings from single PIR response
        to avoid revealing which documents user is interested in.
        
        Args:
            encrypted_chunks: List of encrypted URL+embedding chunk data
            
        Returns:
            Tuple of (list of decrypted URLs, list of corresponding embeddings)
        """
        try:
            # Decrypt each chunk
            decrypted_ints = []
            for i, chunk in enumerate(encrypted_chunks):
                try:
                    decrypted_val = self.crypto_scheme.decrypt(chunk)
                    if decrypted_val > 0:  # Skip zero/empty chunks
                        decrypted_ints.append(decrypted_val)
                except Exception as e:
                    continue
            
            # Convert integers back to bytes and then to text
            if not decrypted_ints:
                return [], []
            
            byte_data = b''
            for int_val in decrypted_ints:
                try:
                    # Convert back to 4-byte chunks
                    chunk_bytes = int_val.to_bytes(4, 'big')
                    byte_data += chunk_bytes
                except (OverflowError, ValueError) as e:
                    continue
            
            # Remove null bytes and decode
            byte_data = byte_data.rstrip(b'\x00')
            combined_string = byte_data.decode('utf-8', errors='ignore')
            
            # Parse URL+embedding pairs
            urls = []
            embeddings = []
            
            # Split by document separator
            doc_pairs = [pair.strip() for pair in combined_string.split('###') if pair.strip()]
            
            for doc_pair in doc_pairs:
                try:
                    # Split URL from embedding: URL|||embedding_as_string
                    parts = doc_pair.split('|||')
                    if len(parts) >= 2:
                        url = parts[0].strip()
                        embedding_str = parts[1].strip()
                        
                        # Parse embedding from comma-separated string
                        embedding_values = [float(x) for x in embedding_str.split(',')]
                        embedding = np.array(embedding_values)
                        
                        urls.append(url)
                        embeddings.append(embedding)
                except (ValueError, IndexError) as e:
                    # Skip malformed entries
                    continue
            
            return urls, embeddings
            
        except Exception as e:
            return []
    
    def rerank_documents(self, query_embedding: torch.Tensor, doc_tuples: List[Tuple[str, np.ndarray]], 
                        top_k: int = 10) -> List[str]:
        """
        Re-rank retrieved documents using embeddings already obtained from PIR.
        
        PRIVACY FIX: No longer requests embeddings from server, uses PIR-obtained embeddings.
        
        Args:
            query_embedding: Query embedding
            doc_tuples: List of (URL, embedding) tuples from PIR retrieval
            top_k: Number of top URLs to return
            
        Returns:
            List of top-k re-ranked URLs
        """
        print(f"[PIR-RAG] DEBUG: ===== RERANKING START =====")
        rerank_start = time.perf_counter()
        
        if not doc_tuples:
            print(f"[PIR-RAG] DEBUG: No documents to rerank")
            return []
        
        print(f"[PIR-RAG] DEBUG: Reranking {len(doc_tuples)} documents")
        
        # Extract URLs and embeddings from tuples
        urls = [doc[0] for doc in doc_tuples]
        embeddings = [doc[1] for doc in doc_tuples]
        
        # Convert embeddings to torch tensor
        doc_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        
        # Normalize embeddings
        query_embedding = query_embedding / torch.norm(query_embedding)
        doc_embeddings = doc_embeddings / torch.norm(doc_embeddings, dim=1, keepdim=True)
        
        # Compute similarities and get top-k
        similarities = torch.mm(query_embedding.unsqueeze(0), doc_embeddings.T)[0]
        top_k_value = min(top_k, len(urls))
        
        if top_k_value > 0:
            top_k_indices = torch.topk(similarities, k=top_k_value).indices
            result_urls = [urls[i] for i in top_k_indices]
        else:
            result_urls = urls[:top_k_value]
        
        rerank_time = time.perf_counter() - rerank_start
        print(f"[PIR-RAG] DEBUG: Reranking took {rerank_time:.4f}s")
        print(f"[PIR-RAG] DEBUG: Returned {len(result_urls)} URLs")
        print(f"[PIR-RAG] DEBUG: ===== RERANKING COMPLETE =====")
        
        return result_urls
