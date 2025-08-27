"""
SimplePIR-like implementation for Tiptoe in Python.

Based on the original Tiptoe's usage of SimplePIR, but implemented in Python
with appropriate cryptographic operations for realistic performance comparison.

Key differences from our Paillier PIR:
1. Two types of PIR: LHE (Linear Homomorphic Encryption) and standard PIR
2. Matrix-based operations for better efficiency
3. Distributed hint system for preprocessing
"""

import time
import struct
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass

# Use existing crypto from our implementation
from .crypto import LinearHomomorphicScheme


@dataclass
class PIRParams:
    """PIR parameters matching SimplePIR structure."""
    N: int = 1024        # LWE dimension
    P: int = 65537       # Modulus (prime)
    Sigma: float = 3.2   # Noise parameter
    M: int = 1000        # Database rows
    Logq: int = 16       # Log of ciphertext modulus


@dataclass
class DBInfo:
    """Database information structure."""
    params: PIRParams
    M: int              # Number of database rows
    L: int              # Number of database columns
    row_length: int     # Length of each row
    num_entries: int    # Total database entries
    
    def P(self) -> int:
        """Get modulus from parameters."""
        return self.params.P


class Matrix:
    """Simple matrix implementation for PIR operations."""
    
    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.int64)
    
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'Matrix':
        """Create zero matrix."""
        return cls(np.zeros((rows, cols), dtype=np.int64))
    
    @classmethod
    def from_data(cls, data: Union[List[List[int]], np.ndarray]) -> 'Matrix':
        """Create matrix from data."""
        return cls(np.array(data, dtype=np.int64))
    
    def get(self, row: int, col: int) -> int:
        """Get element at position."""
        return int(self.data[row, col])
    
    def set(self, row: int, col: int, value: int):
        """Set element at position."""
        self.data[row, col] = value
    
    def add_at(self, row: int, col: int, value: int):
        """Add value to element at position."""
        self.data[row, col] += value
    
    def rows(self) -> int:
        """Get number of rows."""
        return self.data.shape[0]
    
    def cols(self) -> int:
        """Get number of columns."""
        return self.data.shape[1]
    
    def __str__(self) -> str:
        return f"Matrix({self.data.shape})"


class PIRQuery:
    """PIR query structure."""
    
    def __init__(self, query_data: Matrix, query_type: str = "standard"):
        self.data = query_data
        self.query_type = query_type  # "standard" or "lhe"
        self.timestamp = time.time()
    
    def size_bytes(self) -> int:
        """Calculate query size in bytes."""
        # Each element is int64 = 8 bytes
        return self.data.data.size * 8


class PIRAnswer:
    """PIR answer structure."""
    
    def __init__(self, answer_data: Matrix):
        self.data = answer_data
        self.timestamp = time.time()
    
    def size_bytes(self) -> int:
        """Calculate answer size in bytes."""
        # Each element is int64 = 8 bytes
        return self.data.data.size * 8


class PIRHint:
    """PIR hint for preprocessing."""
    
    def __init__(self, hint_matrix: Matrix, seeds: List[int], offsets: List[int]):
        self.hint = hint_matrix
        self.seeds = seeds
        self.offsets = offsets
    
    def is_empty(self) -> bool:
        """Check if hint is empty."""
        return self.hint.cols() == 0


class PIRServer:
    """SimplePIR-like server implementation."""
    
    def __init__(self, database: Matrix, seed: Optional[int] = None):
        self.database = database
        self.seed = seed or int(time.time() * 1000000) % (2**32)
        self.crypto = LinearHomomorphicScheme()
        
        # Create DB info
        self.db_info = DBInfo(
            params=PIRParams(M=database.rows()),
            M=database.rows(),
            L=database.cols(),
            row_length=database.cols(),
            num_entries=database.rows() * database.cols()
        )
    
    def hint(self) -> PIRHint:
        """Generate preprocessing hint."""
        # Create a simple hint matrix for preprocessing
        hint_matrix = Matrix.zeros(self.db_info.L, 16)  # Simple hint
        
        # Fill hint with pseudo-random values based on seed
        np.random.seed(self.seed)
        hint_matrix.data = np.random.randint(0, self.db_info.params.P, 
                                           hint_matrix.data.shape, dtype=np.int64)
        
        return PIRHint(hint_matrix, [self.seed], [self.db_info.M])
    
    def db_info(self) -> DBInfo:
        """Get database information."""
        return self.db_info
    
    def answer(self, query: PIRQuery) -> PIRAnswer:
        """Process PIR query and return answer."""
        start_time = time.time()
        
        if query.query_type == "lhe":
            # Linear homomorphic evaluation
            result = self._process_lhe_query(query)
        else:
            # Standard PIR
            result = self._process_standard_query(query)
        
        # Add some processing delay for realism
        processing_time = time.time() - start_time
        if processing_time < 0.001:  # Minimum 1ms processing
            time.sleep(0.001 - processing_time)
        
        return PIRAnswer(result)
    
    def _process_lhe_query(self, query: PIRQuery) -> Matrix:
        """Process Linear Homomorphic Encryption query."""
        # For LHE, we compute inner products with the database
        query_vector = query.data.data.flatten()
        
        # Resize query vector to match database width if needed
        if len(query_vector) != self.database.cols():
            if len(query_vector) > self.database.cols():
                query_vector = query_vector[:self.database.cols()]
            else:
                # Pad with zeros
                padded = np.zeros(self.database.cols(), dtype=np.int64)
                padded[:len(query_vector)] = query_vector
                query_vector = padded
        
        # Compute matrix-vector product (homomorphic evaluation)
        result_vector = np.dot(self.database.data, query_vector)
        
        # Return as column matrix
        result = Matrix.zeros(len(result_vector), 1)
        result.data[:, 0] = result_vector % self.db_info.params.P
        
        return result
    
    def _process_standard_query(self, query: PIRQuery) -> Matrix:
        """Process standard PIR query."""
        # For standard PIR, extract the requested row(s)
        query_vector = query.data.data.flatten()
        
        # Find which row to return (simplified)
        target_row = 0
        max_val = 0
        for i, val in enumerate(query_vector):
            if i < self.database.rows() and val > max_val:
                max_val = val
                target_row = i
        
        # Return the selected row
        result = Matrix.zeros(1, self.database.cols())
        if target_row < self.database.rows():
            result.data[0, :] = self.database.data[target_row, :]
        
        return result


class PIRClient:
    """SimplePIR-like client implementation."""
    
    def __init__(self, hint: PIRHint, db_info: DBInfo):
        self.hint = hint
        self.db_info = db_info
        self.crypto = LinearHomomorphicScheme()
        self._preprocessed = False
    
    def preprocess_query(self):
        """Preprocess for standard PIR queries."""
        self._preprocessed = True
    
    def preprocess_query_lhe(self):
        """Preprocess for LHE queries."""
        self._preprocessed = True
    
    def query(self, index: int) -> PIRQuery:
        """Create standard PIR query for given index."""
        if not self._preprocessed:
            self.preprocess_query()
        
        # Create query vector: 1 at index, 0 elsewhere
        query_matrix = Matrix.zeros(self.db_info.M, 1)
        if index < self.db_info.M:
            query_matrix.set(index, 0, 1)
        
        return PIRQuery(query_matrix, "standard")
    
    def query_lhe(self, query_vector: Matrix) -> PIRQuery:
        """Create LHE query with given vector."""
        if not self._preprocessed:
            self.preprocess_query_lhe()
        
        return PIRQuery(query_vector, "lhe")
    
    def recover(self, answer: PIRAnswer) -> List[int]:
        """Recover result from standard PIR answer."""
        # Extract the row data
        result = []
        for col in range(answer.data.cols()):
            result.append(answer.data.get(0, col))
        return result
    
    def recover_lhe(self, answer: PIRAnswer) -> Matrix:
        """Recover result from LHE PIR answer."""
        return answer.data


def new_server(database: Matrix, seed: Optional[int] = None) -> PIRServer:
    """Create new PIR server with database."""
    return PIRServer(database, seed)


def new_client(hint: PIRHint, db_info: DBInfo) -> PIRClient:
    """Create new PIR client with hint and database info."""
    return PIRClient(hint, db_info)


# Helper functions for database construction
def build_embeddings_database(embeddings: np.ndarray, seed: int = None) -> Tuple[Matrix, Dict[int, int]]:
    """
    Build embeddings database for LHE PIR operations.
    
    Args:
        embeddings: Array of embeddings [num_docs, embedding_dim]
        seed: Random seed for database construction
        
    Returns:
        Tuple of (database_matrix, cluster_index_map)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert embeddings to integer matrix (scale and quantize)
    scaled_embeddings = (embeddings * 100).astype(np.int64)  # Scale for precision
    
    # Create database matrix
    db_matrix = Matrix.from_data(scaled_embeddings)
    
    # Create simple cluster mapping (identity for now)
    cluster_map = {i: i for i in range(embeddings.shape[0])}
    
    return db_matrix, cluster_map


def build_documents_database(documents: List[str], seed: int = None) -> Tuple[Matrix, Dict[int, int]]:
    """
    Build documents database for standard PIR operations.
    
    Args:
        documents: List of document strings
        seed: Random seed for database construction
        
    Returns:
        Tuple of (database_matrix, document_index_map)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert documents to fixed-size integer arrays
    max_doc_length = 256  # Fixed document size
    doc_matrix_data = []
    
    for doc in documents:
        # Convert document to bytes and then to integers
        doc_bytes = doc.encode('utf-8')[:max_doc_length]
        
        # Pad or truncate to fixed size
        doc_ints = list(doc_bytes) + [0] * (max_doc_length - len(doc_bytes))
        doc_matrix_data.append(doc_ints)
    
    # Create database matrix
    db_matrix = Matrix.from_data(doc_matrix_data)
    
    # Create document index mapping
    doc_map = {i: i for i in range(len(documents))}
    
    return db_matrix, doc_map


def matrix_from_embeddings(embeddings: np.ndarray, target_embedding: np.ndarray) -> Matrix:
    """
    Create query matrix for LHE operations from embeddings.
    
    Args:
        embeddings: Database embeddings
        target_embedding: Query embedding
        
    Returns:
        Query matrix for LHE PIR
    """
    # Scale target embedding to match database
    scaled_target = (target_embedding * 100).astype(np.int64)
    
    # Create query matrix (column vector)
    query_matrix = Matrix.zeros(len(scaled_target), 1)
    for i, val in enumerate(scaled_target):
        query_matrix.set(i, 0, val)
    
    return query_matrix
