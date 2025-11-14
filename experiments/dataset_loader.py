"""
Dataset Loader for PIR-RAG Experiments

Supports loading three datasets:
1. LAION: Text-to-Image retrieval (text_emb -> img_emb, MRR@100)
2. MS_MARCO: Query-to-Corpus retrieval (query_emb -> corpus_emb, MRR@100)
3. SIFT: Query-to-Base retrieval (query -> base, Recall@10)

Each dataset returns:
- queries: np.ndarray of query embeddings
- database: np.ndarray of database embeddings
- ground_truth: ground truth relevance information
"""

import numpy as np
import json
import struct
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional


class DatasetLoader:
    """Unified loader for PIR-RAG evaluation datasets."""

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize dataset loader.

        Args:
            data_dir: Path to directory containing dataset folders
        """
        self.data_dir = Path(data_dir)
        self.datasets_info = {
            "LAION": {
                "query_file": "laion/text_emb_0.npy",
                "database_file": "laion/img_emb_0.npy",
                "metadata_file": "laion/metadata_0.parquet",
                "eval_metric": "MRR@100"
            },
            "MS_MARCO": {
                "query_file": "ms_marco/query_embeddings_192d.npy",
                "database_file": "ms_marco/corpus_embeddings_192d.npy",
                "ground_truth_file": "ms_marco/mappings.json",
                "eval_metric": "MRR@100"
            },
            "SIFT": {
                "query_file": "sifit/sift_query.fvecs",
                "database_file": "sifit/sift_base.fvecs",
                "ground_truth_file": "sifit/sift_groundtruth.ivecs",
                "eval_metric": "Recall@10"
            }
        }

    def _read_fvecs(self, filename: str, max_vectors: Optional[int] = None) -> np.ndarray:
        """
        Read .fvecs file format (used by SIFT dataset).

        Format: Each vector starts with int32 dimension, followed by float32 values

        Args:
            filename: Path to .fvecs file
            max_vectors: Maximum number of vectors to read (for memory efficiency)
        """
        vectors = []
        with open(filename, 'rb') as f:
            count = 0
            while True:
                # Check if we've read enough vectors
                if max_vectors is not None and count >= max_vectors:
                    break

                # Read dimension
                dim_bytes = f.read(4)
                if len(dim_bytes) < 4:
                    break
                dim = struct.unpack('i', dim_bytes)[0]

                # Read vector data
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) < dim * 4:
                    break
                vector = struct.unpack('{}f'.format(dim), vector_bytes)
                vectors.append(vector)
                count += 1

        return np.array(vectors, dtype=np.float32)

    def _read_ivecs(self, filename: str, max_vectors: Optional[int] = None) -> np.ndarray:
        """
        Read .ivecs file format (used for SIFT ground truth).

        Format: Each vector starts with int32 dimension, followed by int32 values

        Args:
            filename: Path to .ivecs file
            max_vectors: Maximum number of vectors to read (for memory efficiency)
        """
        vectors = []
        with open(filename, 'rb') as f:
            count = 0
            while True:
                # Check if we've read enough vectors
                if max_vectors is not None and count >= max_vectors:
                    break

                # Read dimension
                dim_bytes = f.read(4)
                if len(dim_bytes) < 4:
                    break
                dim = struct.unpack('i', dim_bytes)[0]

                # Read vector data
                vector_bytes = f.read(dim * 4)
                if len(vector_bytes) < dim * 4:
                    break
                vector = struct.unpack('{}i'.format(dim), vector_bytes)
                vectors.append(vector)
                count += 1

        return np.array(vectors, dtype=np.int32)

    def create_ground_truth_from_mappings(self, mappings_data, max_queries=None, max_docs=None):
        """
        According to MS MARCO mapping, generate ground truth

        Args:
            mappings_data (dict)
            max_queries (int, optional): Defaults to None.
            max_docs (int, optional): Defaults to None.

        Returns:
            dict: {0: [15], 1: [102], ...}
        """

        try:
            query_to_doc_id_str = mappings_data["query_to_positive_doc_id"]
            ordered_query_ids_str = mappings_data["query_ids_ordered"]
            ordered_doc_ids_str = mappings_data["doc_ids_ordered"]
        except KeyError as e:
            print(f"错误：Mappings JSON文件中缺少关键字段: {e}")
            return {}

        query_id_to_index = {qid: i for i, qid in enumerate(ordered_query_ids_str)}
        doc_id_to_index = {did: i for i, did in enumerate(ordered_doc_ids_str)}


        effective_num_queries = len(ordered_query_ids_str)
        if max_queries is not None:
            effective_num_queries = min(effective_num_queries, max_queries)

        effective_num_docs = len(ordered_doc_ids_str)
        if max_docs is not None:
            effective_num_docs = min(effective_num_docs, max_docs)

        ground_truth = {}

        for query_id_str, doc_id_str in query_to_doc_id_str.items():

            if query_id_str in query_id_to_index and doc_id_str in doc_id_to_index:

                query_idx = query_id_to_index[query_id_str]
                doc_idx = doc_id_to_index[doc_id_str]


                if query_idx < effective_num_queries and doc_idx < effective_num_docs:


                    ground_truth[query_idx] = [doc_idx]

        return ground_truth

    def load_laion(self, max_queries: Optional[int] = None, max_docs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load LAION dataset for text-to-image retrieval.

        Args:
            max_queries: Maximum number of queries to load
            max_docs: Maximum number of database documents to load

        Returns:
            queries: Text embeddings (queries)
            database: Image embeddings (database)
            ground_truth: Positional ground truth (query i matches doc i)
        """
        query_path = self.data_dir / self.datasets_info["LAION"]["query_file"]
        db_path = self.data_dir / self.datasets_info["LAION"]["database_file"]

        if not query_path.exists():
            raise FileNotFoundError(f"LAION query file not found: {query_path}")
        if not db_path.exists():
            raise FileNotFoundError(f"LAION database file not found: {db_path}")

        # Load embeddings
        queries = np.load(query_path)
        database = np.load(db_path)

        # Apply limits independently
        if max_queries is not None:
            queries = queries[:max_queries]
        if max_docs is not None:
            database = database[:max_docs]

        # Generate positional ground truth (query i corresponds to doc i)
        # Only generate ground truth for queries that have corresponding docs in the database
        n_queries = len(queries)
        n_database = len(database)

        # Create ground truth mapping: each query i maps to document i (if it exists in database)
        ground_truth = []
        for i in range(n_queries):
            if i < n_database:
                ground_truth.append(i)  # Query i maps to doc i
            else:
                # If doc i doesn't exist in database, this query has no ground truth
                ground_truth.append(-1)  # -1 indicates no ground truth available

        ground_truth = np.array(ground_truth)

        return queries, database, ground_truth

    def load_ms_marco(self, max_queries: Optional[int] = None, max_docs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load MS_MARCO dataset for query-to-corpus retrieval.

        Args:
            max_queries: Maximum number of queries to load
            max_docs: Maximum number of database documents to load

        Returns:
            queries: Query embeddings
            database: Corpus embeddings
            ground_truth: Dict mapping query indices to relevant doc indices
        """
        query_path = self.data_dir / self.datasets_info["MS_MARCO"]["query_file"]
        db_path = self.data_dir / self.datasets_info["MS_MARCO"]["database_file"]
        gt_path = self.data_dir / self.datasets_info["MS_MARCO"]["ground_truth_file"]

        if not query_path.exists():
            raise FileNotFoundError(f"MS_MARCO query file not found: {query_path}")
        if not db_path.exists():
            raise FileNotFoundError(f"MS_MARCO database file not found: {db_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"MS_MARCO ground truth file not found: {gt_path}")

        # Load embeddings
        queries = np.load(query_path)
        database = np.load(db_path)

        # Load ground truth mappings using the improved method
        with open(gt_path, 'r') as f:
            mappings_data = json.load(f)

        # Use the comprehensive ground truth creation method
        ground_truth = self.create_ground_truth_from_mappings(
            mappings_data, max_queries=max_queries, max_docs=max_docs
        )

        # Convert ground truth keys to strings for consistency with existing code
        ground_truth = {str(k): v for k, v in ground_truth.items()}

        # Apply limits to embeddings
        if max_queries is not None:
            queries = queries[:max_queries]
        if max_docs is not None:
            database = database[:max_docs]

        return queries, database, ground_truth

    def load_sift(self, max_queries: Optional[int] = None, max_docs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load SIFT dataset for nearest neighbor retrieval.

        Args:
            max_queries: Maximum number of queries to load
            max_docs: Maximum number of database documents to load

        Returns:
            queries: Query vectors
            database: Base vectors
            ground_truth: Ground truth nearest neighbors
        """
        query_path = self.data_dir / self.datasets_info["SIFT"]["query_file"]
        db_path = self.data_dir / self.datasets_info["SIFT"]["database_file"]
        gt_path = self.data_dir / self.datasets_info["SIFT"]["ground_truth_file"]

        if not query_path.exists():
            raise FileNotFoundError(f"SIFT query file not found: {query_path}")
        if not db_path.exists():
            raise FileNotFoundError(f"SIFT database file not found: {db_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"SIFT ground truth file not found: {gt_path}")

        # Load data with efficient limits
        queries = self._read_fvecs(str(query_path), max_vectors=max_queries)
        database = self._read_fvecs(str(db_path), max_vectors=max_docs)
        ground_truth = self._read_ivecs(str(gt_path), max_vectors=max_queries)

        # No need for additional slicing since we limited during reading
        # But ensure ground truth indices are valid for the loaded database
        if max_docs is not None:
            ground_truth = np.where(ground_truth < max_docs, ground_truth, -1)

        return queries, database, ground_truth

    def load_dataset(self, dataset_name: str, max_queries: Optional[int] = None,
                    max_docs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Any]:
        """
        Load specified dataset with unified interface.

        Args:
            dataset_name: One of "LAION", "MS_MARCO", "SIFT"
            max_queries: Maximum number of queries to load
            max_docs: Maximum number of database documents to load

        Returns:
            queries: Query embeddings/vectors
            database: Database embeddings/vectors
            ground_truth: Ground truth in dataset-specific format
        """
        if dataset_name == "LAION":
            return self.load_laion(max_queries, max_docs)
        elif dataset_name == "MS_MARCO":
            return self.load_ms_marco(max_queries, max_docs)
        elif dataset_name == "SIFT":
            return self.load_sift(max_queries, max_docs)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: LAION, MS_MARCO, SIFT")

    def get_dataset_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available datasets and their status.

        Returns:
            Dict with dataset info including file existence and shapes
        """
        info = {}

        for name, config in self.datasets_info.items():
            dataset_info = {
                "eval_metric": config["eval_metric"],
                "files_exist": True,
                "error": None
            }

            try:
                queries, database, ground_truth = self.load_dataset(name, max_queries=5, max_docs=5)
                dataset_info.update({
                    "queries_shape": queries.shape,
                    "database_shape": database.shape,
                    "ground_truth_type": type(ground_truth).__name__,
                    "sample_loaded": True
                })
            except Exception as e:
                dataset_info.update({
                    "files_exist": False,
                    "error": str(e),
                    "sample_loaded": False
                })

            info[name] = dataset_info

        return info


# Test functionality
if __name__ == "__main__":
    print("🔍 Testing Dataset Loader")
    print("=" * 40)

    loader = DatasetLoader()
    info = loader.get_dataset_info()

    for dataset_name, details in info.items():
        print(f"\n📊 {dataset_name}:")
        print(f"  Evaluation metric: {details['eval_metric']}")

        if details['sample_loaded']:
            print(f"  ✅ Loaded successfully")
            print(f"  Queries shape: {details['queries_shape']}")
            print(f"  Database shape: {details['database_shape']}")
            print(f"  Ground truth type: {details['ground_truth_type']}")
        else:
            print(f"  ❌ Failed to load: {details['error']}")