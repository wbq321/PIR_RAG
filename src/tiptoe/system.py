"""
Tiptoe System Implementation

Main system class implementing the complete Tiptoe private search architecture.
Follows the same interface pattern as PIR-RAG and Graph-PIR for fair comparison.

Architecture:
1. Setup: PCA reduction + clustering + cryptographic preprocessing
2. Query: Two-phase PIR (cluster ranking + document retrieval)
3. Metrics: Detailed communication and performance tracking
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple

from .crypto import LinearHomomorphicScheme, LinearHomomorphicPIR, TiptoeHintSystem
from .clustering import TiptoeClustering

# Import PIR-RAG utilities for document encoding (reuse existing code)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pir_rag.utils import encode_text_to_chunks, decode_chunks_to_text


class TiptoeSystem:
    """
    Complete Tiptoe private search system implementation.
    
    Provides the same interface as PIRRAGSystem and GraphPIRSystem for
    fair three-way comparison in communication efficiency experiments.
    """
    
    def __init__(self, target_dim: int = 192, n_clusters: int = 1280, security_param: int = 128):
        """
        Initialize Tiptoe system.
        
        Args:
            target_dim: PCA target dimension (default 192, same as Tiptoe paper)
            n_clusters: Number of clusters (default 1280, same as Tiptoe paper)  
            security_param: Cryptographic security parameter
        """
        self.target_dim = target_dim
        self.n_clusters = n_clusters
        self.security_param = security_param
        
        # Core components
        self.clustering = TiptoeClustering(target_dim, n_clusters)
        self.crypto_scheme = None
        self.pir_system = None
        self.hint_system = None
        
        # Data storage
        self.documents = None
        self.embeddings = None
        self.cluster_databases = {}  # cluster_id -> database for PIR
        self.document_databases = {}  # cluster_id -> document chunks for PIR
        
        # Setup metrics
        self.setup_complete = False
    
    def setup(self, embeddings: np.ndarray, documents: List[str], **kwargs) -> Dict[str, Any]:
        """
        Setup Tiptoe system with document corpus.
        
        Args:
            embeddings: Document embeddings (e.g., 768-dimensional from BGE)
            documents: List of document texts
            **kwargs: Additional configuration parameters
            
        Returns:
            Setup metrics dictionary
        """
        setup_start = time.perf_counter()
        
        print(f"[Tiptoe] Setting up Tiptoe system...")
        print(f"[Tiptoe] Documents: {len(documents)}, Embeddings: {embeddings.shape}")
        
        # Store data
        self.embeddings = embeddings
        self.documents = documents
        
        # Phase 1: Clustering and PCA preprocessing
        print(f"[Tiptoe] Phase 1: Clustering and dimensionality reduction...")
        clustering_start = time.perf_counter()
        clustering_metrics = self.clustering.setup_clustering(embeddings)
        clustering_time = time.perf_counter() - clustering_start
        
        # Phase 2: Cryptographic setup
        print(f"[Tiptoe] Phase 2: Setting up linear homomorphic encryption...")
        crypto_start = time.perf_counter()
        self.crypto_scheme = LinearHomomorphicScheme(self.security_param)
        self.pir_system = LinearHomomorphicPIR(self.crypto_scheme)
        self.hint_system = TiptoeHintSystem(self.crypto_scheme)
        crypto_time = time.perf_counter() - crypto_start
        
        # Phase 3: Database preparation
        print(f"[Tiptoe] Phase 3: Preparing PIR databases...")
        db_start = time.perf_counter()
        db_metrics = self._prepare_pir_databases(documents)
        db_time = time.perf_counter() - db_start
        
        # Phase 4: Hint generation (offline preprocessing)
        print(f"[Tiptoe] Phase 4: Generating cryptographic hints...")
        hint_start = time.perf_counter()
        hint_metrics = self._generate_hints()
        hint_time = time.perf_counter() - hint_start
        
        total_setup_time = time.perf_counter() - setup_start
        self.setup_complete = True
        
        # Combine all metrics
        setup_metrics = {
            'total_setup_time': total_setup_time,
            'clustering_time': clustering_time,
            'crypto_setup_time': crypto_time,
            'database_prep_time': db_time,
            'hint_generation_time': hint_time,
            'target_dimension': self.target_dim,
            'n_clusters': self.n_clusters,
            'security_parameter': self.security_param,
            **clustering_metrics,
            **db_metrics,
            **hint_metrics
        }
        
        print(f"[Tiptoe] Setup complete in {total_setup_time:.2f}s")
        return setup_metrics
    
    def query(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[List[str], Dict[str, Any]]:
        """
        Perform Tiptoe two-phase private search query.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (retrieved_documents, query_metrics)
        """
        if not self.setup_complete:
            raise ValueError("System not setup. Call setup() first.")
        
        query_start = time.perf_counter()
        
        print(f"[Tiptoe] Starting two-phase query for top-{top_k} documents...")
        
        # Phase 1: Find relevant cluster (client-side, private)
        print(f"[Tiptoe] Phase 1: Private cluster selection...")
        phase1_start = time.perf_counter()
        cluster_id, cluster_metrics = self.clustering.find_nearest_cluster(query_embedding)
        print(f"[Tiptoe] Selected cluster {cluster_id}")
        
        # Phase 1: Private ranking within cluster
        ranking_docs, phase1_metrics = self._phase1_cluster_ranking(query_embedding, cluster_id, top_k)
        phase1_time = time.perf_counter() - phase1_start
        
        # Phase 2: Private document retrieval
        print(f"[Tiptoe] Phase 2: Private document retrieval...")
        phase2_start = time.perf_counter()
        retrieved_docs, phase2_metrics = self._phase2_document_retrieval(cluster_id, ranking_docs, top_k)
        phase2_time = time.perf_counter() - phase2_start
        
        total_query_time = time.perf_counter() - query_start
        
        # Combine metrics
        query_metrics = {
            'total_query_time': total_query_time,
            'phase1_time': phase1_time,
            'phase2_time': phase2_time,
            'selected_cluster': cluster_id,
            'documents_retrieved': len(retrieved_docs),
            **cluster_metrics,
            **phase1_metrics,
            **phase2_metrics
        }
        
        print(f"[Tiptoe] Query completed in {total_query_time:.3f}s")
        print(f"[Tiptoe] Retrieved {len(retrieved_docs)} documents")
        
        return retrieved_docs, query_metrics
    
    def _prepare_pir_databases(self, documents: List[str]) -> Dict[str, Any]:
        """
        Prepare PIR databases for both phases.
        
        Phase 1: Ranking database (reduced embeddings per cluster)
        Phase 2: Document database (encrypted document chunks per cluster)
        """
        print(f"[Tiptoe] Preparing databases for {self.clustering.n_clusters} clusters...")
        
        # Phase 1 databases: ranking matrices per cluster
        ranking_db_size = 0
        for cluster_id in range(self.clustering.n_clusters):
            ranking_matrix = self.clustering.prepare_ranking_database(cluster_id)
            self.cluster_databases[cluster_id] = ranking_matrix
            ranking_db_size += ranking_matrix.size
        
        # Phase 2 databases: document chunks per cluster  
        document_db_size = 0
        max_chunks_per_doc = 0
        
        for cluster_id in range(self.clustering.n_clusters):
            doc_indices = self.clustering.get_cluster_document_indices(cluster_id)
            cluster_docs = [documents[i] for i in doc_indices]
            
            # Convert documents to chunks (reuse PIR-RAG encoding)
            cluster_chunks = []
            for doc in cluster_docs:
                chunks = encode_text_to_chunks(doc)
                cluster_chunks.append(chunks)
                max_chunks_per_doc = max(max_chunks_per_doc, len(chunks))
            
            self.document_databases[cluster_id] = {
                'doc_indices': doc_indices,
                'document_chunks': cluster_chunks
            }
            document_db_size += sum(len(chunks) for chunks in cluster_chunks)
        
        metrics = {
            'ranking_database_size': ranking_db_size,
            'document_database_size': document_db_size,
            'max_chunks_per_document': max_chunks_per_doc,
            'clusters_prepared': len(self.cluster_databases)
        }
        
        print(f"[Tiptoe] Database preparation complete:")
        print(f"  - Ranking DB size: {ranking_db_size:,} elements")
        print(f"  - Document DB size: {document_db_size:,} chunks")
        print(f"  - Max chunks per doc: {max_chunks_per_doc}")
        
        return metrics
    
    def _generate_hints(self) -> Dict[str, Any]:
        """Generate cryptographic hints for all cluster databases."""
        total_hint_size = 0
        
        # Generate hints for a subset of clusters (efficiency)
        active_clusters = min(100, self.clustering.n_clusters)  # Limit for research prototype
        
        for cluster_id in range(active_clusters):
            if cluster_id in self.cluster_databases:
                ranking_matrix = self.cluster_databases[cluster_id]
                if ranking_matrix.size > 0:
                    # Convert matrix to list for hint generation
                    db_vectors = [ranking_matrix[:, i] for i in range(ranking_matrix.shape[1])]
                    hint = self.hint_system.generate_hint(db_vectors)
                    total_hint_size += hint['hint_size']
        
        hint_metrics = self.hint_system.calculate_hint_communication()
        hint_metrics['total_hint_size'] = total_hint_size
        hint_metrics['active_clusters_with_hints'] = active_clusters
        
        return hint_metrics
    
    def _phase1_cluster_ranking(self, query_embedding: np.ndarray, cluster_id: int, top_k: int) -> Tuple[List[int], Dict]:
        """
        Phase 1: Private ranking of documents within selected cluster.
        
        Returns:
            Tuple of (top_document_indices_in_cluster, phase1_metrics)
        """
        if cluster_id not in self.cluster_databases:
            # Empty cluster, return empty results
            return [], {
                'phase1_upload_bytes': 0,
                'phase1_download_bytes': 0,
                'phase1_pir_time': 0,
                'documents_in_cluster': 0
            }
        
        ranking_matrix = self.cluster_databases[cluster_id]
        if ranking_matrix.size == 0:
            return [], {
                'phase1_upload_bytes': 0,
                'phase1_download_bytes': 0,
                'phase1_pir_time': 0,
                'documents_in_cluster': 0
            }
        
        print(f"[Tiptoe] Phase 1: Ranking {ranking_matrix.shape[1]} documents in cluster {cluster_id}")
        
        # Apply PCA to query (same as done during clustering)
        from sklearn.preprocessing import normalize
        normalized_query = normalize(query_embedding.reshape(1, -1), norm='l2')
        reduced_query = self.clustering.pca_model.transform(normalized_query)[0]
        quantized_query = np.round(reduced_query * (1 << 5)).astype(np.int32)
        
        # Simulate PIR ranking computation
        # In real Tiptoe, this would be done homomorphically
        pir_start = time.perf_counter()
        
        # Compute similarity scores (this simulates the homomorphic inner product)
        scores = []
        for i in range(ranking_matrix.shape[1]):
            doc_vector = ranking_matrix[:, i]
            score = np.dot(quantized_query, doc_vector)
            scores.append((score, i))
        
        # Sort by score and take top-k
        scores.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc_idx for score, doc_idx in scores[:top_k]]
        
        pir_time = time.perf_counter() - pir_start
        
        # Calculate communication costs (simulated PIR)
        upload_bytes = len(quantized_query) * 64  # Encrypted query vector
        download_bytes = top_k * 32  # Encrypted scores
        
        phase1_metrics = {
            'phase1_upload_bytes': upload_bytes,
            'phase1_download_bytes': download_bytes,
            'phase1_pir_time': pir_time,
            'documents_in_cluster': ranking_matrix.shape[1],
            'ranking_matrix_size': ranking_matrix.size
        }
        
        return top_docs, phase1_metrics
    
    def _phase2_document_retrieval(self, cluster_id: int, doc_indices: List[int], top_k: int) -> Tuple[List[str], Dict]:
        """
        Phase 2: Private retrieval of actual document content.
        
        Args:
            cluster_id: Selected cluster
            doc_indices: Document indices within cluster (from Phase 1)
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (retrieved_documents, phase2_metrics)
        """
        if cluster_id not in self.document_databases:
            return [], {
                'phase2_upload_bytes': 0,
                'phase2_download_bytes': 0,
                'phase2_pir_time': 0,
                'documents_retrieved': 0
            }
        
        cluster_data = self.document_databases[cluster_id]
        doc_chunks = cluster_data['document_chunks']
        
        if not doc_indices or len(doc_chunks) == 0:
            return [], {
                'phase2_upload_bytes': 0,
                'phase2_download_bytes': 0,
                'phase2_pir_time': 0,
                'documents_retrieved': 0
            }
        
        print(f"[Tiptoe] Phase 2: Retrieving {len(doc_indices)} documents from cluster {cluster_id}")
        
        retrieved_docs = []
        total_upload = 0
        total_download = 0
        total_pir_time = 0
        
        # Retrieve each document using PIR
        for i, doc_idx in enumerate(doc_indices[:top_k]):
            if doc_idx < len(doc_chunks):
                pir_start = time.perf_counter()
                
                # Get document chunks
                chunks = doc_chunks[doc_idx]
                
                # Simulate PIR retrieval (in real system this would be homomorphic)
                # Each chunk retrieval requires a separate PIR query
                if isinstance(chunks, list):
                    doc_upload = len(chunks) * 64  # PIR query per chunk
                    doc_download = sum(len(str(chunk)) for chunk in chunks)  # Encrypted chunks
                    
                    # Decode chunks back to document text
                    doc_text = decode_chunks_to_text(chunks)
                    retrieved_docs.append(doc_text)
                else:
                    # Fallback for unexpected types
                    doc_upload = 64  # Single query
                    doc_download = 100  # Estimate
                    doc_text = f"Error retrieving document {doc_idx}"
                    retrieved_docs.append(doc_text)
                
                pir_time = time.perf_counter() - pir_start
                
                total_upload += doc_upload
                total_download += doc_download
                total_pir_time += pir_time
        
        phase2_metrics = {
            'phase2_upload_bytes': total_upload,
            'phase2_download_bytes': total_download,
            'phase2_pir_time': total_pir_time,
            'documents_retrieved': len(retrieved_docs),
            'avg_chunks_per_doc': total_download / max(1, len(retrieved_docs)) if retrieved_docs else 0
        }
        
        return retrieved_docs, phase2_metrics
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system configuration and statistics."""
        if not self.setup_complete:
            return {'status': 'not_setup'}
        
        return {
            'status': 'ready',
            'target_dimension': self.target_dim,
            'n_clusters': self.n_clusters,
            'security_parameter': self.security_param,
            'n_documents': len(self.documents) if self.documents else 0,
            'original_embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'clusters_with_data': len([k for k, v in self.cluster_databases.items() if v.size > 0]),
            'crypto_scheme': 'Linear Homomorphic (LWE-based)',
            'clustering_method': 'PCA + K-means'
        }
