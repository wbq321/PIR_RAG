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

from .piano_pir import PianoPIRClient, PianoPIRServer, PianoPIRConfig
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

    def _prepare_vector_database(self, vector_db: np.ndarray, embedding_dim: int,
                                graph_dict: Dict) -> List[int]:
        """
        Prepare vector database in uint64 format for PIR.

        Args:
            vector_db: Flattened embeddings array
            embedding_dim: Dimension of each embedding
            graph_dict: Graph adjacency information

        Returns:
            Database as list of uint64 values
        """
        import struct

        raw_db = []
        num_vectors = len(vector_db) // embedding_dim

        for i in range(num_vectors):
            # Get vector data
            start_idx = i * embedding_dim
            end_idx = start_idx + embedding_dim
            vector = vector_db[start_idx:end_idx]

            # Convert vector to bytes (float32 -> 4 bytes each)
            vector_bytes = vector.astype(np.float32).tobytes()

            # Get neighbors from graph (up to 16 neighbors)
            neighbors = graph_dict.get(i, [])[:16]
            # Pad neighbors to exactly 16 entries
            while len(neighbors) < 16:
                neighbors.append(-1)  # -1 indicates no neighbor

            # Convert neighbors to bytes (int32 -> 4 bytes each)
            neighbors_bytes = b''
            for neighbor in neighbors:
                neighbors_bytes += struct.pack('<i', neighbor)

            # Combine vector and neighbors
            entry_bytes = vector_bytes + neighbors_bytes

            # Convert to uint64 array (8 bytes per uint64)
            uint64_array = []
            for j in range(0, len(entry_bytes), 8):
                chunk = entry_bytes[j:j+8]
                if len(chunk) < 8:
                    chunk = chunk + b'\x00' * (8 - len(chunk))
                uint64_val = struct.unpack('<Q', chunk)[0]
                uint64_array.append(uint64_val)

            raw_db.extend(uint64_array)

        return raw_db

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

        # Convert embeddings to PIR database format with graph structure
        vector_db = embeddings.flatten().astype(np.float32)
        graph_dict = {}
        if hasattr(self.graph_search, 'graph'):
            graph_dict = self.graph_search.graph

        # Create PIR configuration for vectors
        vector_entry_size = embeddings.shape[1] * 4  # 4 bytes per float32
        # Add space for neighbor indices (assume max 16 neighbors * 4 bytes each)
        vector_entry_size += 16 * 4
        vector_config = PianoPIRConfig(
            db_size=len(embeddings),
            db_entry_byte_num=vector_entry_size,
            embedding_dim=embeddings.shape[1]  # Pass the actual embedding dimension
        )

        # Convert vector database to uint64 format
        vector_raw_db = self._prepare_vector_database(vector_db, embeddings.shape[1], graph_dict)

        # Initialize PIR server and client with proper arguments
        self.vector_pir_server = PianoPIRServer(vector_config, vector_raw_db)
        self.vector_pir_client = PianoPIRClient(vector_config)

        vector_pir_time = time.perf_counter() - vector_pir_start

        # 3. Set up document PIR database (just store documents, no separate PIR client)
        print("[GraphPIR] Setting up document PIR...")
        doc_pir_start = time.perf_counter()
        self.doc_pir_server = DocumentPIRServer()

        # Convert documents to PIR database format
        doc_setup_metrics = self.doc_pir_server.setup_database(documents)
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
        Phase 1: Use graph search + REAL vector PIR to find candidate document indices.
        Based on private-search-temp implementation.
        """
        print("[GraphPIR] Phase 1: Graph traversal with vector PIR...")

        # Track PIR communication costs
        total_upload_bytes = 0
        total_download_bytes = 0
        pir_query_count = 0

        # Initialize graph search state
        visited = set()
        candidates = []

        # Start with entry point(s) - use first few documents as entry points
        n_entry_points = min(5, len(self.embeddings))
        current_nodes = list(range(n_entry_points))

        # Add entry points to visited
        for node in current_nodes:
            visited.add(node)
            # Calculate distance to query
            dist = self._calculate_distance(query_embedding, self.embeddings[node])
            candidates.append((node, dist))

        max_iterations = 10  # Limit graph traversal steps
        nodes_per_step = 5   # Number of neighbors to explore per step

        for iteration in range(max_iterations):
            if len(candidates) >= num_candidates:
                break

            # Select best unvisited neighbors to explore
            next_nodes_to_query = []

            # Get neighbors of current best nodes using graph structure
            for node_id in current_nodes:
                if hasattr(self.graph_search, 'graph') and node_id in self.graph_search.graph:
                    neighbors = self.graph_search.graph[node_id][:nodes_per_step]
                    for neighbor in neighbors:
                        if neighbor not in visited and neighbor < len(self.embeddings):
                            next_nodes_to_query.append(neighbor)
                            visited.add(neighbor)

            if not next_nodes_to_query:
                break

            # Use REAL PIR to retrieve vectors for these nodes
            print(f"[GraphPIR] PIR query for {len(next_nodes_to_query)} vectors at iteration {iteration}")

            # Generate PIR query for the neighbor vectors
            pir_vectors, pir_metrics = self.vector_pir_client.query_vectors(
                self.vector_pir_server, next_nodes_to_query
            )

            total_upload_bytes += pir_metrics["upload_bytes"]
            total_download_bytes += pir_metrics["download_bytes"]
            pir_query_count += 1

            # Calculate distances and add to candidates
            current_nodes = []
            for i, node_id in enumerate(next_nodes_to_query):
                if i < len(pir_vectors):
                    # Use the retrieved vector to calculate distance
                    vector = pir_vectors[i]
                    dist = self._calculate_distance(query_embedding, vector)
                    candidates.append((node_id, dist))
                    current_nodes.append(node_id)

        # Sort candidates by distance and return top candidates
        candidates.sort(key=lambda x: x[1])
        candidate_indices = [idx for idx, _ in candidates[:num_candidates]]

        phase1_metrics = {
            "phase1_upload_bytes": total_upload_bytes,
            "phase1_download_bytes": total_download_bytes,
            "graph_traversal_steps": pir_query_count,
            "pir_queries_made": pir_query_count,
            "total_nodes_explored": len(visited)
        }

        print(f"[GraphPIR] Phase 1 complete: {pir_query_count} PIR queries, {len(visited)} nodes explored")

        return candidate_indices, phase1_metrics

    def _phase2_document_retrieval(self, candidate_indices: List[int]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Phase 2: Use REAL document PIR with Paillier encryption to retrieve actual document texts.
        Similar to PIR-RAG's approach but for individual documents.
        """
        from phe import paillier

        print(f"[GraphPIR] Phase 2: Document PIR with Paillier encryption for {len(candidate_indices)} documents")

        # Generate Paillier keys (same approach as PIR-RAG)
        print("[GraphPIR] Generating Paillier keys...")
        key_start = time.perf_counter()
        public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
        key_gen_time = time.perf_counter() - key_start

        upload_bytes = 0
        download_bytes = 0
        encryption_time = 0
        decryption_time = 0
        server_time = 0

        retrieved_docs = []

        for doc_idx in candidate_indices:
            # Generate PIR query for this document (similar to PIR-RAG)
            query_start = time.perf_counter()

            # Create PIR query vector: encrypt 1 for target document, 0 for others
            encrypted_query = []
            for i in range(len(self.documents)):
                if i == doc_idx:
                    encrypted_query.append(public_key.encrypt(1))
                else:
                    encrypted_query.append(public_key.encrypt(0))

            query_gen_time = time.perf_counter() - query_start
            encryption_time += query_gen_time

            # Calculate upload size (encrypted query vector)
            query_upload = sum(len(str(c.ciphertext())) for c in encrypted_query)
            upload_bytes += query_upload

            # Server processes PIR query using homomorphic operations
            server_start = time.perf_counter()
            encrypted_response = self._process_document_pir_query(encrypted_query, public_key)
            server_processing_time = time.perf_counter() - server_start
            server_time += server_processing_time

            # Calculate download size (encrypted response chunks)
            response_download = sum(len(str(c.ciphertext())) for c in encrypted_response)
            download_bytes += response_download

            # Client decrypts response
            decrypt_start = time.perf_counter()
            decrypted_chunks = [private_key.decrypt(c) for c in encrypted_response]
            doc_text = decode_chunks_to_text(decrypted_chunks)
            decrypt_time = time.perf_counter() - decrypt_start
            decryption_time += decrypt_time

            if doc_text.strip():  # Only add non-empty documents
                retrieved_docs.append(doc_text)

        phase2_metrics = {
            "phase2_upload_bytes": upload_bytes,
            "phase2_download_bytes": download_bytes,
            "documents_retrieved": len(retrieved_docs),
            "key_generation_time": key_gen_time,
            "encryption_time": encryption_time,
            "server_processing_time": server_time,
            "decryption_time": decryption_time,
            "total_pir_documents": len(candidate_indices)
        }

        print(f"[GraphPIR] Phase 2 complete: {len(retrieved_docs)} documents retrieved")
        print(f"[GraphPIR] Communication: {upload_bytes:,} upload, {download_bytes:,} download bytes")

        return retrieved_docs, phase2_metrics

    def _process_document_pir_query(self, encrypted_query: List, public_key) -> List:
        """
        Server-side processing of document PIR query using homomorphic encryption.
        Identical approach to PIR-RAG's handle_pir_query.
        """
        if not hasattr(self.doc_pir_server, 'doc_chunks_db'):
            raise ValueError("Document PIR server not properly set up")

        # Perform homomorphic computation for each chunk position (same as PIR-RAG)
        encrypted_response = []
        for chunk_db in self.doc_pir_server.doc_chunks_db:
            # Compute sum of (chunk_value * encrypted_query_bit) for all documents
            chunk_result = public_key.encrypt(0)  # Start with encrypted zero
            for i, chunk_value in enumerate(chunk_db):
                if i < len(encrypted_query):
                    # Homomorphic multiplication and addition
                    chunk_result += chunk_value * encrypted_query[i]
            encrypted_response.append(chunk_result)

        return encrypted_response

    def _calculate_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate distance between two vectors using cosine similarity.
        """
        # Handle different input types
        if isinstance(vec1, torch.Tensor):
            vec1 = vec1.numpy()
        if isinstance(vec2, torch.Tensor):
            vec2 = vec2.numpy()

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance

        cosine_sim = dot_product / (norm1 * norm2)
        return 1.0 - cosine_sim  # Convert to distance (0 = identical, 2 = opposite)

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
        for i, doc in enumerate(documents):
            if not doc or not isinstance(doc, str):
                print(f"Warning: Document {i} is empty or invalid, using placeholder")
                doc = f"Empty document {i}"

            try:
                chunks = encode_text_to_chunks(doc)
                if not isinstance(chunks, list):
                    print(f"Error: encode_text_to_chunks returned {type(chunks)} instead of list for doc {i}")
                    chunks = [0]  # Fallback
                doc_chunks.append(chunks)
            except Exception as e:
                print(f"Error encoding document {i}: {e}")
                doc_chunks.append([0])  # Fallback empty chunk

        # Find max chunks needed
        if not doc_chunks:
            self.max_chunks = 1
        else:
            chunk_lengths = []
            for chunks in doc_chunks:
                if isinstance(chunks, list):
                    chunk_lengths.append(len(chunks))
                else:
                    print(f"Warning: chunks is not a list: {type(chunks)}")
                    chunk_lengths.append(1)
            self.max_chunks = max(chunk_lengths) if chunk_lengths else 1

        print(f"[DocPIR] Max chunks per document: {self.max_chunks}")

        # Create PIR database organized by chunk position (similar to PIR-RAG)
        self.doc_chunks_db = [[] for _ in range(self.max_chunks)]
        for chunk_idx in range(self.max_chunks):
            for doc_chunks_list in doc_chunks:
                if isinstance(doc_chunks_list, list) and chunk_idx < len(doc_chunks_list):
                    self.doc_chunks_db[chunk_idx].append(doc_chunks_list[chunk_idx])
                else:
                    self.doc_chunks_db[chunk_idx].append(0)  # Padding

        return {
            "doc_pir_max_chunks": self.max_chunks,
            "doc_pir_db_size": len(documents)
        }
        encrypted_chunks = response_data["encrypted_chunks"]

        # In reality, would decrypt each chunk
        decrypted_chunks = encrypted_chunks  # Simulated

        # Decode chunks back to text (same as PIR-RAG)
        document_text = decode_chunks_to_text(decrypted_chunks)

        # Simulate download size
        download_bytes = len(encrypted_chunks) * 256  # Simulated

        return document_text, download_bytes
