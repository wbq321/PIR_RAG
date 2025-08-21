"""
Graph-based search engine with PIR integration.
Simplified implementation of the graph ANN search from private-search-temp.
"""

import numpy as np
import heapq
import random
import time
import math
from typing import List, Tuple, Dict, Any, Optional
from .piano_pir import SimpleBatchPianoPIR


class GraphSearchEngine:
    """
    Graph-based Approximate Nearest Neighbor search with PIR integration.
    Simplified version of the Go implementation in private-search-temp.
    """
    
    def __init__(self, vectors: np.ndarray, graph: Optional[np.ndarray] = None, 
                 neighbor_count: int = 32, private_mode: bool = True):
        """
        Initialize the graph search engine.
        
        Args:
            vectors: Document embedding vectors (n_docs, embedding_dim)
            graph: Pre-built graph adjacency matrix (n_docs, neighbor_count)
            neighbor_count: Number of neighbors per node in the graph
            private_mode: Whether to use PIR for private access
        """
        self.vectors = vectors
        self.n_docs, self.embedding_dim = vectors.shape
        self.neighbor_count = neighbor_count
        self.private_mode = private_mode
        
        # Build or use provided graph
        if graph is None:
            print(f"Building graph with {neighbor_count} neighbors per node...")
            self.graph = self._build_graph()
        else:
            self.graph = graph
            
        # Setup PIR database if in private mode
        if private_mode:
            self._setup_pir_database()
        
        # Search statistics
        self.total_vertex_accesses = 0
        self.total_search_time = 0
        self.search_count = 0
        
    def _build_graph(self) -> np.ndarray:
        """Build a simplified graph using random neighbors + some nearest neighbors"""
        graph = np.zeros((self.n_docs, self.neighbor_count), dtype=np.int32)
        
        print("  -> Building graph structure...")
        for i in range(self.n_docs):
            if i % 10000 == 0:
                print(f"     Processing node {i}/{self.n_docs}")
                
            # For simplicity, use random neighbors with some actual nearest neighbors
            neighbors = set()
            
            # Add some random neighbors
            while len(neighbors) < self.neighbor_count // 2:
                neighbor = random.randint(0, self.n_docs - 1)
                if neighbor != i:
                    neighbors.add(neighbor)
            
            # Add some actual nearest neighbors based on cosine similarity
            if len(neighbors) < self.neighbor_count:
                distances = np.dot(self.vectors[i], self.vectors.T)
                distances[i] = -float('inf')  # Exclude self
                
                # Get top candidates
                top_indices = np.argsort(distances)[-self.neighbor_count:]
                for idx in top_indices:
                    if len(neighbors) >= self.neighbor_count:
                        break
                    neighbors.add(idx)
            
            # Fill remaining slots with random if needed
            while len(neighbors) < self.neighbor_count:
                neighbor = random.randint(0, self.n_docs - 1)
                if neighbor != i:
                    neighbors.add(neighbor)
                    
            graph[i] = list(neighbors)[:self.neighbor_count]
            
        return graph
    
    def _setup_pir_database(self):
        """Setup PIR database for private vertex access"""
        print("  -> Setting up PIR database...")
        
        # Create database entries: [vector_data + neighbor_data]
        entry_size = self.embedding_dim * 4 + self.neighbor_count * 4  # float32 + int32
        raw_db = np.zeros((self.n_docs, entry_size // 4), dtype=np.uint32)
        
        for i in range(self.n_docs):
            # Pack vector data (float32 -> uint32)
            vector_bytes = self.vectors[i].astype(np.float32).tobytes()
            vector_uint32 = np.frombuffer(vector_bytes, dtype=np.uint32)
            
            # Pack neighbor data
            neighbor_bytes = self.graph[i].astype(np.int32).tobytes()
            neighbor_uint32 = np.frombuffer(neighbor_bytes, dtype=np.uint32)
            
            # Combine into database entry
            raw_db[i, :len(vector_uint32)] = vector_uint32
            raw_db[i, len(vector_uint32):len(vector_uint32)+len(neighbor_uint32)] = neighbor_uint32
        
        # Initialize batch PIR
        batch_size = min(100, max(10, int(math.sqrt(self.n_docs))))
        self.pir_system = SimpleBatchPianoPIR(
            db_size=self.n_docs,
            db_entry_byte_num=entry_size,
            batch_size=batch_size,
            raw_db=raw_db
        )
        
        # Preprocess PIR
        self.pir_system.preprocessing()
        print(f"  -> PIR setup complete. Batch size: {batch_size}")
    
    def _unpack_vertex_data(self, raw_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack raw database entry into vector and neighbors"""
        # Convert back to bytes then to proper types
        raw_bytes = raw_data.astype(np.uint32).tobytes()
        
        # Extract vector (first embedding_dim float32 values)
        vector_bytes = raw_bytes[:self.embedding_dim * 4]
        vector = np.frombuffer(vector_bytes, dtype=np.float32)
        
        # Extract neighbors (next neighbor_count int32 values)
        neighbor_bytes = raw_bytes[self.embedding_dim * 4:self.embedding_dim * 4 + self.neighbor_count * 4]
        neighbors = np.frombuffer(neighbor_bytes, dtype=np.int32)
        
        return vector, neighbors
    
    def _get_vertices_private(self, vertex_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve vertex information using PIR"""
        if not hasattr(self, 'pir_system'):
            raise RuntimeError("PIR system not initialized")
            
        # Query PIR system
        raw_results, comm_stats = self.pir_system.batch_query(vertex_ids)
        
        # Unpack results
        vertices = []
        for i, raw_data in enumerate(raw_results):
            vector, neighbors = self._unpack_vertex_data(raw_data)
            vertices.append({
                'id': vertex_ids[i],
                'vector': vector,
                'neighbors': neighbors,
                'communication_bytes': comm_stats['total_upload_bytes'] + comm_stats['total_download_bytes']
            })
        
        return vertices
    
    def _get_vertices_non_private(self, vertex_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve vertex information directly (non-private baseline)"""
        vertices = []
        for vid in vertex_ids:
            if 0 <= vid < self.n_docs:
                vertices.append({
                    'id': vid,
                    'vector': self.vectors[vid],
                    'neighbors': self.graph[vid],
                    'communication_bytes': 0  # No communication cost in non-private mode
                })
            else:
                # Return dummy vertex for out-of-bounds
                vertices.append({
                    'id': vid,
                    'vector': np.zeros(self.embedding_dim),
                    'neighbors': np.array([0] * self.neighbor_count),
                    'communication_bytes': 0
                })
        return vertices
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def search_knn(self, query_vector: np.ndarray, k: int, 
                   max_steps: int = 15, parallel_exploration: int = 2) -> Tuple[List[int], Dict[str, Any]]:
        """
        Perform k-nearest neighbor search using graph traversal with PIR.
        
        Args:
            query_vector: Query embedding vector
            k: Number of nearest neighbors to find
            max_steps: Maximum number of search steps
            parallel_exploration: Number of vertices to explore in parallel at each step
            
        Returns:
            Tuple of (top_k_indices, search_metrics)
        """
        start_time = time.perf_counter()
        
        # Initialize search
        visited = set()
        candidates = []  # Min-heap of (negative_similarity, vertex_id)
        
        # Start from random vertices (simplified start vertex selection)
        start_vertex_count = min(int(math.sqrt(self.n_docs)), 10)
        start_vertices = random.sample(range(self.n_docs), start_vertex_count)
        
        # Performance tracking
        metrics = {
            "total_communication_bytes": 0,
            "vertex_accesses": 0,
            "steps_taken": 0,
            "start_vertices": len(start_vertices)
        }
        
        # Get initial vertices
        if self.private_mode:
            initial_vertices = self._get_vertices_private(start_vertices)
        else:
            initial_vertices = self._get_vertices_non_private(start_vertices)
        
        metrics["total_communication_bytes"] += sum(v["communication_bytes"] for v in initial_vertices)
        metrics["vertex_accesses"] += len(initial_vertices)
        
        # Add initial candidates
        for vertex in initial_vertices:
            if vertex['id'] not in visited:
                similarity = self._cosine_similarity(query_vector, vertex['vector'])
                heapq.heappush(candidates, (-similarity, vertex['id']))  # Negative for max-heap
                visited.add(vertex['id'])
        
        # Graph traversal
        for step in range(max_steps):
            metrics["steps_taken"] += 1
            
            if not candidates:
                break
                
            # Select vertices to explore in this step
            current_exploration = []
            neighbors_to_visit = []
            
            # Get top candidates for this step
            step_candidates = []
            for _ in range(min(parallel_exploration, len(candidates))):
                if candidates:
                    neg_sim, vertex_id = heapq.heappop(candidates)
                    step_candidates.append((neg_sim, vertex_id))
            
            # Put them back and collect neighbors
            for neg_sim, vertex_id in step_candidates:
                heapq.heappush(candidates, (neg_sim, vertex_id))
                
                # We need to get the neighbors of this vertex
                if self.private_mode:
                    vertex_data = self._get_vertices_private([vertex_id])[0]
                else:
                    vertex_data = self._get_vertices_non_private([vertex_id])[0]
                
                metrics["total_communication_bytes"] += vertex_data["communication_bytes"]
                metrics["vertex_accesses"] += 1
                
                # Add unvisited neighbors to exploration list
                for neighbor_id in vertex_data['neighbors']:
                    if neighbor_id not in visited and 0 <= neighbor_id < self.n_docs:
                        neighbors_to_visit.append(neighbor_id)
                        visited.add(neighbor_id)
            
            # Batch query neighbors if any
            if neighbors_to_visit:
                if self.private_mode:
                    neighbor_vertices = self._get_vertices_private(neighbors_to_visit)
                else:
                    neighbor_vertices = self._get_vertices_non_private(neighbors_to_visit)
                
                metrics["total_communication_bytes"] += sum(v["communication_bytes"] for v in neighbor_vertices)
                metrics["vertex_accesses"] += len(neighbor_vertices)
                
                # Add neighbors as candidates
                for vertex in neighbor_vertices:
                    similarity = self._cosine_similarity(query_vector, vertex['vector'])
                    heapq.heappush(candidates, (-similarity, vertex['id']))
        
        # Extract top-k results
        final_candidates = []
        while candidates and len(final_candidates) < k:
            neg_sim, vertex_id = heapq.heappop(candidates)
            final_candidates.append((vertex_id, -neg_sim))
        
        # Sort by similarity (descending)
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        top_k_indices = [vertex_id for vertex_id, _ in final_candidates[:k]]
        
        # Update statistics
        search_time = time.perf_counter() - start_time
        metrics["search_time"] = search_time
        metrics["results_found"] = len(top_k_indices)
        
        self.total_vertex_accesses += metrics["vertex_accesses"]
        self.total_search_time += search_time
        self.search_count += 1
        
        return top_k_indices, metrics
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall search engine statistics"""
        stats = {
            "total_searches": self.search_count,
            "total_vertex_accesses": self.total_vertex_accesses,
            "total_search_time": self.total_search_time,
            "avg_vertex_accesses_per_search": self.total_vertex_accesses / max(1, self.search_count),
            "avg_search_time": self.total_search_time / max(1, self.search_count),
            "private_mode": self.private_mode,
            "graph_neighbor_count": self.neighbor_count,
            "database_size": self.n_docs,
            "embedding_dimension": self.embedding_dim
        }
        
        if self.private_mode and hasattr(self, 'pir_system'):
            pir_stats = self.pir_system.get_stats()
            stats.update({
                "pir_preprocessing_time": pir_stats["preprocessing_time"],
                "pir_total_communication_bytes": pir_stats["total_communication_bytes"],
                "pir_avg_communication_per_query": pir_stats["avg_communication_per_query"],
                "pir_partition_count": pir_stats["partition_count"]
            })
        
        return stats
