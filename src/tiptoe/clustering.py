"""
Tiptoe Clustering System

Implements Tiptoe's semantic clustering approach:
1. PCA dimensionality reduction (768 → 192 dimensions)
2. K-means clustering using scikit-learn (similar to Faiss)
3. Cluster assignment and centroid computation
4. Database preparation for PIR operations
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import pickle


class TiptoeClustering:
    """
    Handles Tiptoe's clustering pipeline matching the original implementation.

    Pipeline:
    1. Apply PCA to reduce embedding dimensions
    2. Perform K-means clustering on reduced embeddings
    3. Assign documents to clusters
    4. Prepare cluster databases for PIR operations
    """

    def __init__(self, target_dim: int = 192, n_clusters: int = 1280):
        """
        Initialize Tiptoe clustering system.

        Args:
            target_dim: Target dimension after PCA (default 192, same as Tiptoe)
            n_clusters: Number of clusters (default 1280, same as Tiptoe)
        """
        self.target_dim = target_dim
        self.n_clusters = n_clusters
        self.pca_model = None
        self.kmeans_model = None
        self.cluster_centroids = None
        self.cluster_assignments = None
        self.reduced_embeddings = None

    def setup_clustering(self, embeddings: np.ndarray, precision_bits: int = 5) -> Dict[str, Any]:
        """
        Complete clustering setup pipeline matching Tiptoe's approach.

        Args:
            embeddings: Original high-dimensional embeddings (e.g., 768-dim)
            precision_bits: Precision for quantization (default 5, same as Tiptoe)

        Returns:
            Setup metrics and cluster information
        """
        setup_start = time.perf_counter()

        print(f"[Tiptoe] Starting clustering setup for {len(embeddings)} embeddings...")
        print(f"[Tiptoe] Original embedding dimension: {embeddings.shape[1]}")
        print(f"[Tiptoe] Target dimension: {self.target_dim}")
        print(f"[Tiptoe] Number of clusters: {self.n_clusters}")

        # Step 1: PCA dimensionality reduction
        pca_start = time.perf_counter()
        self.reduced_embeddings = self._apply_pca_reduction(embeddings, precision_bits)
        pca_time = time.perf_counter() - pca_start

        print(f"[Tiptoe] PCA reduction completed in {pca_time:.2f}s")
        print(f"[Tiptoe] Reduced embeddings shape: {self.reduced_embeddings.shape}")

        # Step 2: K-means clustering
        clustering_start = time.perf_counter()
        self.cluster_assignments, self.cluster_centroids = self._perform_clustering(
            self.reduced_embeddings
        )
        clustering_time = time.perf_counter() - clustering_start

        print(f"[Tiptoe] K-means clustering completed in {clustering_time:.2f}s")

        # Step 3: Analyze cluster distribution
        cluster_stats = self._analyze_clusters()

        total_setup_time = time.perf_counter() - setup_start

        setup_metrics = {
            'total_setup_time': total_setup_time,
            'pca_time': pca_time,
            'clustering_time': clustering_time,
            'original_dim': embeddings.shape[1],
            'reduced_dim': self.target_dim,
            'n_clusters': self.n_clusters,
            'n_documents': len(embeddings),
            **cluster_stats
        }

        print(f"[Tiptoe] Clustering setup complete in {total_setup_time:.2f}s")
        return setup_metrics

    def _apply_pca_reduction(self, embeddings: np.ndarray, precision_bits: int) -> np.ndarray:
        """
        Apply PCA reduction matching Tiptoe's preprocessing.

        Args:
            embeddings: Original embeddings
            precision_bits: Quantization precision

        Returns:
            Reduced and quantized embeddings
        """
        # Determine actual target dimension (can't exceed number of samples)
        n_samples, n_features = embeddings.shape
        max_components = min(n_samples, n_features)
        actual_target_dim = min(self.target_dim, max_components)

        if actual_target_dim != self.target_dim:
            print(f"[Tiptoe] Warning: Reducing target dimension from {self.target_dim} to {actual_target_dim} due to sample size limit")
            self.target_dim = actual_target_dim  # Update for consistency

        # Step 1: Train PCA model
        print(f"[Tiptoe] Training PCA model...")
        self.pca_model = PCA(n_components=actual_target_dim, svd_solver='full')

        # Normalize embeddings first (common practice)
        normalized_embeddings = normalize(embeddings, norm='l2')

        # Fit PCA
        self.pca_model.fit(normalized_embeddings)

        # Step 2: Transform embeddings
        print(f"[Tiptoe] Applying PCA transformation...")
        reduced = self.pca_model.transform(normalized_embeddings)

        # Step 3: Apply precision quantization (matching Tiptoe's approach)
        # Tiptoe uses: numpy.round(v * (1 << precision_bits))
        print(f"[Tiptoe] Applying {precision_bits}-bit quantization...")
        quantized = np.round(reduced * (1 << precision_bits)).astype(np.int32)

        return quantized

    def _perform_clustering(self, reduced_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform K-means clustering on reduced embeddings.

        Args:
            reduced_embeddings: PCA-reduced embeddings

        Returns:
            Tuple of (cluster_assignments, cluster_centroids)
        """
        n_samples = len(reduced_embeddings)

        # Adjust number of clusters for small datasets
        actual_n_clusters = min(self.n_clusters, n_samples)
        if actual_n_clusters != self.n_clusters:
            print(f"[Tiptoe] Warning: Reducing clusters from {self.n_clusters} to {actual_n_clusters} due to dataset size")
            self.n_clusters = actual_n_clusters  # Update for consistency

        print(f"[Tiptoe] Running MiniBatch K-means clustering with {actual_n_clusters} clusters...")

        # Use MiniBatchKMeans for efficiency (like Tiptoe)
        self.kmeans_model = MiniBatchKMeans(
            n_clusters=actual_n_clusters,
            random_state=42,
            batch_size=min(1000, n_samples),  # Adjust batch size for small datasets
            max_iter=100
        )

        # Fit and predict
        cluster_assignments = self.kmeans_model.fit_predict(reduced_embeddings.astype(np.float32))
        cluster_centroids = self.kmeans_model.cluster_centers_

        return cluster_assignments, cluster_centroids

    def _analyze_clusters(self) -> Dict[str, Any]:
        """Analyze cluster distribution and quality."""
        unique_clusters, cluster_sizes = np.unique(self.cluster_assignments, return_counts=True)

        stats = {
            'actual_clusters_used': len(unique_clusters),
            'avg_cluster_size': float(np.mean(cluster_sizes)),
            'min_cluster_size': int(np.min(cluster_sizes)),
            'max_cluster_size': int(np.max(cluster_sizes)),
            'cluster_size_std': float(np.std(cluster_sizes))
        }

        print(f"[Tiptoe] Cluster analysis:")
        print(f"  - Clusters used: {stats['actual_clusters_used']}/{self.n_clusters}")
        print(f"  - Avg cluster size: {stats['avg_cluster_size']:.1f}")
        print(f"  - Cluster size range: {stats['min_cluster_size']} - {stats['max_cluster_size']}")

        return stats

    def find_nearest_cluster(self, query_embedding: np.ndarray) -> Tuple[int, Dict]:
        """
        Find nearest cluster for query embedding (client-side operation).

        Args:
            query_embedding: Query embedding vector

        Returns:
            Tuple of (cluster_id, metrics)
        """
        if self.pca_model is None or self.cluster_centroids is None:
            raise ValueError("Clustering not setup. Call setup_clustering() first.")

        start_time = time.perf_counter()

        # Step 1: Apply same PCA transformation as training
        normalized_query = normalize(query_embedding.reshape(1, -1), norm='l2')
        reduced_query = self.pca_model.transform(normalized_query)[0]

        # Step 2: Apply same quantization
        quantized_query = np.round(reduced_query * (1 << 5)).astype(np.int32)

        # Step 3: Find nearest cluster centroid (inner product, like Tiptoe)
        # Tiptoe uses: distances = numpy.matmul(centroids, query.T)
        distances = np.dot(self.cluster_centroids, quantized_query)
        nearest_cluster = np.argmax(distances)

        search_time = time.perf_counter() - start_time

        metrics = {
            'cluster_search_time': search_time,
            'selected_cluster': int(nearest_cluster),
            'cluster_score': float(distances[nearest_cluster]),
            'query_reduced_dim': len(quantized_query)
        }

        return int(nearest_cluster), metrics

    def get_cluster_database(self, cluster_id: int, embeddings: np.ndarray) -> np.ndarray:
        """
        Get all embeddings belonging to a specific cluster for PIR database.

        Args:
            cluster_id: Target cluster ID
            embeddings: Original or reduced embeddings

        Returns:
            Matrix of embeddings in the cluster
        """
        if self.cluster_assignments is None:
            raise ValueError("Clustering not setup. Call setup_clustering() first.")

        # Find all documents in this cluster
        cluster_mask = (self.cluster_assignments == cluster_id)
        cluster_embeddings = embeddings[cluster_mask]

        return cluster_embeddings

    def get_cluster_document_indices(self, cluster_id: int) -> List[int]:
        """
        Get indices of all documents in a specific cluster.

        Args:
            cluster_id: Target cluster ID

        Returns:
            List of document indices in the cluster
        """
        if self.cluster_assignments is None:
            raise ValueError("Clustering not setup. Call setup_clustering() first.")

        cluster_indices = np.where(self.cluster_assignments == cluster_id)[0]
        return cluster_indices.tolist()

    def prepare_ranking_database(self, cluster_id: int) -> np.ndarray:
        """
        Prepare ranking database matrix for a specific cluster (Phase 1 PIR).

        Args:
            cluster_id: Target cluster ID

        Returns:
            Ranking matrix for PIR operations
        """
        if self.reduced_embeddings is None:
            raise ValueError("Clustering not setup. Call setup_clustering() first.")

        # Get reduced embeddings for this cluster
        cluster_embeddings = self.get_cluster_database(cluster_id, self.reduced_embeddings)

        # Transpose to match Tiptoe's matrix format (columns = documents)
        ranking_matrix = cluster_embeddings.T

        return ranking_matrix

    def save_clustering_state(self, filepath: str):
        """Save clustering models and state."""
        state = {
            'pca_model': self.pca_model,
            'kmeans_model': self.kmeans_model,
            'cluster_centroids': self.cluster_centroids,
            'cluster_assignments': self.cluster_assignments,
            'reduced_embeddings': self.reduced_embeddings,
            'target_dim': self.target_dim,
            'n_clusters': self.n_clusters
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_clustering_state(self, filepath: str):
        """Load clustering models and state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.pca_model = state['pca_model']
        self.kmeans_model = state['kmeans_model']
        self.cluster_centroids = state['cluster_centroids']
        self.cluster_assignments = state['cluster_assignments']
        self.reduced_embeddings = state['reduced_embeddings']
        self.target_dim = state['target_dim']
        self.n_clusters = state['n_clusters']
