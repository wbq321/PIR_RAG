"""
Graph-PIR System: Unified interface for graph-based PIR search.
Provides compatibility with the PIR-RAG comparison framework.
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple
from .graph_search import GraphSearchEngine


class GraphPIRSystem:
    """
    Graph-based PIR system for comparison with PIR-RAG.
    Implements the same interface as PIR-RAG for fair comparison.
    """
    
    def __init__(self, private_mode: bool = True):
        self.private_mode = private_mode
        self.search_engine = None
        self.setup_time = 0
        self.is_ready = False
        
        # Performance tracking
        self.total_queries = 0
        self.total_communication_bytes = 0
        self.total_query_time = 0
        
    def setup(self, embeddings: np.ndarray, documents_text: List[str], 
              neighbor_count: int = 32) -> Dict[str, Any]:
        """
        Set up the graph-based PIR system.
        
        Args:
            embeddings: Document embeddings array (n_docs, embedding_dim)
            documents_text: List of document texts (for compatibility)
            neighbor_count: Number of neighbors per node in the graph
            
        Returns:
            Dictionary with setup statistics
        """
        setup_start = time.perf_counter()
        
        print(f"  -> Setting up Graph-PIR system with {len(embeddings)} docs, {neighbor_count} neighbors...")
        print(f"  -> Private mode: {self.private_mode}")
        
        # Initialize graph search engine
        self.search_engine = GraphSearchEngine(
            vectors=embeddings,
            neighbor_count=neighbor_count,
            private_mode=self.private_mode
        )
        
        self.setup_time = time.perf_counter() - setup_start
        self.is_ready = True
        
        print(f"  -> Graph-PIR setup complete in {self.setup_time:.2f}s")
        
        return {
            "setup_time": self.setup_time,
            "n_documents": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "neighbor_count": neighbor_count,
            "private_mode": self.private_mode,
            "graph_nodes": len(embeddings)
        }
    
    def search(self, query_embedding: np.ndarray, top_k: int, 
               max_steps: int = 15, parallel_exploration: int = 2) -> Tuple[List[int], Dict[str, Any]]:
        """
        Perform private graph-based search.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            max_steps: Maximum number of graph traversal steps
            parallel_exploration: Number of vertices to explore in parallel
            
        Returns:
            Tuple of (document_indices, search_metrics)
        """
        if not self.is_ready:
            raise RuntimeError("System not set up. Call setup() first.")
        
        # Ensure query_embedding is 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
            
        # Perform graph search
        doc_indices, search_metrics = self.search_engine.search_knn(
            query_vector=query_embedding,
            k=top_k,
            max_steps=max_steps,
            parallel_exploration=parallel_exploration
        )
        
        # Update global statistics
        self.total_queries += 1
        self.total_communication_bytes += search_metrics.get("total_communication_bytes", 0)
        self.total_query_time += search_metrics.get("search_time", 0)
        
        # Add system-level metrics
        search_metrics.update({
            "system_type": "graph_pir",
            "private_mode": self.private_mode,
            "max_steps": max_steps,
            "parallel_exploration": parallel_exploration
        })
        
        return doc_indices, search_metrics
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        if not self.is_ready:
            return {"status": "not_ready"}
        
        # Get search engine stats
        engine_stats = self.search_engine.get_overall_stats()
        
        # Combine with system-level stats
        system_stats = {
            "system_type": "graph_pir",
            "setup_time": self.setup_time,
            "total_queries": self.total_queries,
            "total_communication_bytes": self.total_communication_bytes,
            "total_query_time": self.total_query_time,
            "avg_communication_per_query": self.total_communication_bytes / max(1, self.total_queries),
            "avg_query_time": self.total_query_time / max(1, self.total_queries),
            "private_mode": self.private_mode
        }
        
        # Merge engine stats
        system_stats.update(engine_stats)
        
        return system_stats
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_queries = 0
        self.total_communication_bytes = 0
        self.total_query_time = 0
        
        if self.search_engine:
            self.search_engine.total_vertex_accesses = 0
            self.search_engine.total_search_time = 0
            self.search_engine.search_count = 0
    
    def get_communication_breakdown(self) -> Dict[str, Any]:
        """Get detailed communication cost breakdown"""
        if not self.is_ready or not self.private_mode:
            return {"error": "Not available in non-private mode"}
        
        if not hasattr(self.search_engine, 'pir_system'):
            return {"error": "PIR system not initialized"}
        
        pir_stats = self.search_engine.pir_system.get_stats()
        
        return {
            "pir_upload_bytes": pir_stats["total_upload_bytes"],
            "pir_download_bytes": pir_stats["total_download_bytes"],
            "pir_total_bytes": pir_stats["total_communication_bytes"],
            "pir_queries_made": pir_stats["total_queries_made"],
            "pir_avg_per_query": pir_stats["avg_communication_per_query"],
            "pir_partitions": pir_stats["partition_count"],
            "total_vertex_accesses": self.search_engine.total_vertex_accesses,
            "avg_vertex_accesses_per_search": (
                self.search_engine.total_vertex_accesses / max(1, self.search_engine.search_count)
            )
        }


def create_comparison_interface(embeddings: np.ndarray, documents_text: List[str], 
                              config: Dict[str, Any]) -> Tuple[GraphPIRSystem, Dict[str, Any]]:
    """
    Create Graph-PIR system for comparison with PIR-RAG.
    
    Args:
        embeddings: Document embeddings
        documents_text: Document texts (for compatibility)
        config: Configuration parameters
        
    Returns:
        Tuple of (system_instance, setup_stats)
    """
    system = GraphPIRSystem(private_mode=config.get("private_mode", True))
    
    setup_stats = system.setup(
        embeddings=embeddings,
        documents_text=documents_text,
        neighbor_count=config.get("neighbor_count", 32)
    )
    
    return system, setup_stats
