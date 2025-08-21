"""
Graph-based Approximate Nearest Neighbor search for Graph-PIR.

Implements HNSW-style graph construction and search.
Based on the graph search approach from private-search-temp.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import heapq
from collections import defaultdict


class GraphSearch:
    """
    Graph-based ANN search implementation.
    
    Builds a navigable small world graph for efficient nearest neighbor search.
    Similar to HNSW but simplified for PIR integration.
    """
    
    def __init__(self):
        self.graph = defaultdict(list)  # adjacency list
        self.embeddings = None
        self.n_nodes = 0
        self.entry_point = 0
        self.built = False
        
    def build_graph(self, embeddings: np.ndarray, k_neighbors: int = 16, 
                   ef_construction: int = 200, max_connections: int = 16):
        """
        Build the graph structure from embeddings.
        
        Args:
            embeddings: Document embeddings [n_docs, embedding_dim]
            k_neighbors: Number of neighbors to connect during construction
            ef_construction: Size of dynamic candidate list during construction
            max_connections: Maximum connections per node
        """
        print(f"[GraphSearch] Building graph for {len(embeddings)} nodes...")
        print(f"[GraphSearch] Parameters: k={k_neighbors}, ef={ef_construction}, max_conn={max_connections}")
        
        self.embeddings = embeddings.copy()
        self.n_nodes = len(embeddings)
        self.graph = defaultdict(list)
        
        # Build graph incrementally
        for i in range(self.n_nodes):
            if i == 0:
                self.entry_point = 0
                continue
                
            # Find nearest neighbors for current node
            candidates = self._search_layer(embeddings[i], ef_construction, 0)
            
            # Connect to k_neighbors closest nodes
            neighbors = self._select_neighbors(candidates, k_neighbors)
            
            for neighbor_idx, _ in neighbors:
                # Add bidirectional connections
                self.graph[i].append(neighbor_idx)
                self.graph[neighbor_idx].append(i)
                
                # Prune connections if too many
                if len(self.graph[neighbor_idx]) > max_connections:
                    self.graph[neighbor_idx] = self._prune_connections(
                        neighbor_idx, self.graph[neighbor_idx], max_connections
                    )
                    
            if (i + 1) % 100 == 0:
                print(f"[GraphSearch] Built {i + 1}/{self.n_nodes} nodes")
                
        self.built = True
        
        # Print graph statistics
        avg_degree = np.mean([len(neighbors) for neighbors in self.graph.values()])
        max_degree = max([len(neighbors) for neighbors in self.graph.values()])
        print(f"[GraphSearch] Graph built: avg_degree={avg_degree:.1f}, max_degree={max_degree}")
        
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[int]:
        """
        Search for top-k nearest neighbors using graph traversal.
        
        Args:
            query_embedding: Query vector
            top_k: Number of nearest neighbors to return
            
        Returns:
            List of document indices sorted by similarity
        """
        if not self.built:
            raise ValueError("Graph not built. Call build_graph() first.")
            
        # Use larger ef during search for better recall
        ef_search = max(top_k * 2, 50)
        
        # Search starting from entry point
        candidates = self._search_layer(query_embedding, ef_search, self.entry_point)
        
        # Return top-k candidates
        top_candidates = self._select_neighbors(candidates, top_k)
        return [idx for idx, _ in top_candidates]
        
    def _search_layer(self, query: np.ndarray, ef: int, entry_point: int) -> List[Tuple[int, float]]:
        """
        Search a single layer of the graph.
        
        Args:
            query: Query vector
            ef: Size of dynamic candidate list
            entry_point: Starting node for search
            
        Returns:
            List of (node_index, distance) tuples
        """
        visited = set()
        candidates = []  # min-heap: (distance, node_idx)
        dynamic_candidates = []  # max-heap: (-distance, node_idx)
        
        # Initialize with entry point
        dist = self._calculate_distance(query, self.embeddings[entry_point])
        heapq.heappush(candidates, (dist, entry_point))
        heapq.heappush(dynamic_candidates, (-dist, entry_point))
        visited.add(entry_point)
        
        while candidates:
            # Get closest unvisited candidate
            current_dist, current_node = heapq.heappop(candidates)
            
            # Check if we can improve dynamic candidates
            if len(dynamic_candidates) >= ef:
                worst_dist = -dynamic_candidates[0][0]
                if current_dist > worst_dist:
                    break
                    
            # Explore neighbors
            for neighbor in self.graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_dist = self._calculate_distance(query, self.embeddings[neighbor])
                    
                    # Add to candidates for further exploration
                    heapq.heappush(candidates, (neighbor_dist, neighbor))
                    
                    # Add to dynamic candidates (best ef candidates so far)
                    heapq.heappush(dynamic_candidates, (-neighbor_dist, neighbor))
                    
                    # Keep only ef best candidates
                    if len(dynamic_candidates) > ef:
                        heapq.heappop(dynamic_candidates)
                        
        # Convert back to (node_idx, distance) format
        result = [(node_idx, -dist) for dist, node_idx in dynamic_candidates]
        result.sort(key=lambda x: x[1])  # Sort by distance (ascending)
        
        return result
        
    def _select_neighbors(self, candidates: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
        """
        Select k best neighbors from candidates.
        
        Args:
            candidates: List of (node_index, distance) tuples
            k: Number of neighbors to select
            
        Returns:
            k best neighbors sorted by distance
        """
        # Sort by distance and take top k
        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        return sorted_candidates[:k]
        
    def _prune_connections(self, node_idx: int, connections: List[int], 
                          max_connections: int) -> List[int]:
        """
        Prune connections to keep only the best ones.
        
        Args:
            node_idx: Index of the node to prune
            connections: Current connections of the node  
            max_connections: Maximum allowed connections
            
        Returns:
            Pruned list of connections
        """
        if len(connections) <= max_connections:
            return connections
            
        # Calculate distances to all connections
        node_embedding = self.embeddings[node_idx]
        connection_distances = []
        
        for conn_idx in connections:
            dist = self._calculate_distance(node_embedding, self.embeddings[conn_idx])
            connection_distances.append((conn_idx, dist))
            
        # Keep closest connections
        connection_distances.sort(key=lambda x: x[1])
        return [conn_idx for conn_idx, _ in connection_distances[:max_connections]]
        
    def _calculate_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate distance between two vectors.
        Using negative cosine similarity (higher similarity = lower distance).
        """
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
            
        cosine_sim = dot_product / (norm1 * norm2)
        return 1.0 - cosine_sim  # Convert to distance (0 = identical, 2 = opposite)
        
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the constructed graph."""
        if not self.built:
            return {"status": "not_built"}
            
        degrees = [len(neighbors) for neighbors in self.graph.values()]
        
        return {
            "status": "built",
            "n_nodes": self.n_nodes,
            "n_edges": sum(degrees) // 2,  # Each edge counted twice
            "avg_degree": np.mean(degrees),
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "entry_point": self.entry_point
        }
