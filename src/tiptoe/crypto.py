"""
Linear Homomorphic Encryption for Tiptoe

Implements simplified linearly homomorphic encryption based on Learning With Errors (LWE).
This is the core cryptographic primitive that enables Tiptoe's private information retrieval.

Key properties:
- Additive homomorphism: Enc(a) + Enc(b) = Enc(a + b)
- Used for private inner product computation in PIR
- Simpler than Paillier's multiplicative homomorphism
"""

import numpy as np
import time
import random
from typing import List, Tuple, Dict, Any
import struct


class LinearHomomorphicScheme:
    """
    Simplified linearly homomorphic encryption scheme for research purposes.

    Based on Learning With Errors (LWE) but simplified for fair comparison
    with PIR-RAG and Graph-PIR systems.
    """

    def __init__(self, security_param: int = 128, modulus: int = None):
        """
        Initialize linear homomorphic encryption scheme.

        Args:
            security_param: Security parameter (bit length)
            modulus: Modulus for operations (auto-calculated if None)
        """
        self.security_param = security_param
        self.modulus = modulus or (1 << 32)  # 32-bit modulus
        self.noise_bound = 100  # Simplified noise parameter

        # Generate keys
        self.public_key, self.private_key = self._generate_keys()

    def _generate_keys(self) -> Tuple[Dict, Dict]:
        """Generate public and private keys for LWE-based scheme."""
        # Simplified key generation for research prototype
        dimension = min(self.security_param, 256)  # LWE dimension

        # Private key: random binary vector
        private_key = {
            'secret': np.random.randint(0, 2, dimension, dtype=np.int32),
            'dimension': dimension
        }

        # Public key: (A, b = A*s + e) where s is secret, e is noise
        A = np.random.randint(0, self.modulus, (dimension, dimension), dtype=np.int64)
        noise = np.random.normal(0, self.noise_bound, dimension).astype(np.int32)
        b = (A @ private_key['secret'] + noise) % self.modulus

        public_key = {
            'A': A,
            'b': b,
            'modulus': self.modulus,
            'dimension': dimension
        }

        return public_key, private_key

    def encrypt(self, plaintext: int) -> Dict:
        """
        Encrypt a single integer using LWE-based encryption.

        Args:
            plaintext: Integer to encrypt

        Returns:
            Ciphertext dictionary with encrypted value
        """
        # LWE encryption: c = (u, v) where u = A^T * r, v = b^T * r + plaintext + e'
        r = np.random.randint(0, 2, self.public_key['dimension'], dtype=np.int32)
        noise = random.randint(-self.noise_bound, self.noise_bound)

        u = (self.public_key['A'].T @ r) % self.modulus
        v = (self.public_key['b'] @ r + plaintext + noise) % self.modulus

        return {
            'u': u,
            'v': v,
            'modulus': self.modulus
        }

    def decrypt(self, ciphertext: Dict) -> int:
        """
        Decrypt ciphertext to recover plaintext.

        Args:
            ciphertext: Encrypted value

        Returns:
            Decrypted integer
        """
        # LWE decryption: plaintext = v - u^T * s (mod modulus)
        inner_product = (ciphertext['u'] @ self.private_key['secret']) % self.modulus
        plaintext = (ciphertext['v'] - inner_product) % self.modulus

        # Handle negative values (convert back from modular arithmetic)
        if plaintext > self.modulus // 2:
            plaintext -= self.modulus

        return int(plaintext)

    def add_encrypted(self, ct1: Dict, ct2: Dict) -> Dict:
        """
        Homomorphic addition: Enc(a) + Enc(b) = Enc(a + b).

        Args:
            ct1, ct2: Ciphertexts to add

        Returns:
            Ciphertext of the sum
        """
        return {
            'u': (ct1['u'] + ct2['u']) % self.modulus,
            'v': (ct1['v'] + ct2['v']) % self.modulus,
            'modulus': self.modulus
        }

    def scalar_multiply(self, ciphertext: Dict, scalar: int) -> Dict:
        """
        Homomorphic scalar multiplication: scalar * Enc(a) = Enc(scalar * a).

        Args:
            ciphertext: Ciphertext to multiply
            scalar: Scalar multiplier

        Returns:
            Ciphertext of scalar * plaintext
        """
        return {
            'u': (scalar * ciphertext['u']) % self.modulus,
            'v': (scalar * ciphertext['v']) % self.modulus,
            'modulus': self.modulus
        }


class LinearHomomorphicPIR:
    """
    Private Information Retrieval using linearly homomorphic encryption.

    Implements the core PIR operations used in both phases of Tiptoe:
    1. Cluster ranking PIR (Phase 1)
    2. Document retrieval PIR (Phase 2)
    """

    def __init__(self, crypto_scheme: LinearHomomorphicScheme):
        """
        Initialize PIR system with homomorphic encryption scheme.

        Args:
            crypto_scheme: Linear homomorphic encryption instance
        """
        self.crypto = crypto_scheme

    def create_pir_query(self, target_index: int, database_size: int) -> Tuple[List[Dict], Dict]:
        """
        Create PIR query for retrieving item at target_index.

        Args:
            target_index: Index of item to retrieve privately
            database_size: Total size of database

        Returns:
            Tuple of (encrypted_query_vector, query_metrics)
        """
        start_time = time.perf_counter()

        # Create query vector: 1 at target_index, 0 elsewhere
        query_vector = [0] * database_size
        query_vector[target_index] = 1

        # Encrypt each element of query vector
        encrypted_query = []
        for value in query_vector:
            encrypted_query.append(self.crypto.encrypt(value))

        query_time = time.perf_counter() - start_time

        # Calculate communication cost (upload)
        upload_bytes = self._calculate_query_size(encrypted_query)

        metrics = {
            'query_generation_time': query_time,
            'upload_bytes': upload_bytes,
            'database_size': database_size,
            'target_index': target_index
        }

        return encrypted_query, metrics

    def process_pir_query(self, encrypted_query: List[Dict], database: List[np.ndarray]) -> Tuple[Dict, Dict]:
        """
        Server processes PIR query using homomorphic operations.

        Computes: Σ(query[i] * database[i]) homomorphically

        Args:
            encrypted_query: Encrypted query vector from client
            database: Server's database (list of vectors/documents)

        Returns:
            Tuple of (encrypted_response, server_metrics)
        """
        start_time = time.perf_counter()

        if len(encrypted_query) != len(database):
            raise ValueError(f"Query length {len(encrypted_query)} != database size {len(database)}")

        # Initialize result as encryption of zero
        result = self.crypto.encrypt(0)

        # Homomorphic inner product computation
        for i, (query_enc, db_item) in enumerate(zip(encrypted_query, database)):
            # For vectors: compute encrypted query[i] * db_item[j] for each dimension
            if isinstance(db_item, np.ndarray):
                # Handle vector database (Phase 1: embeddings)
                for j, db_value in enumerate(db_item):
                    # Homomorphic scalar multiplication: query[i] * db_item[j]
                    term = self.crypto.scalar_multiply(query_enc, int(db_value))
                    result = self.crypto.add_encrypted(result, term)
            else:
                # Handle scalar database (Phase 2: document indices)
                term = self.crypto.scalar_multiply(query_enc, int(db_item))
                result = self.crypto.add_encrypted(result, term)

        processing_time = time.perf_counter() - start_time

        # Calculate communication cost (download)
        download_bytes = self._calculate_response_size(result)

        metrics = {
            'server_processing_time': processing_time,
            'download_bytes': download_bytes,
            'database_items_processed': len(database)
        }

        return result, metrics

    def decrypt_pir_response(self, encrypted_response: Dict) -> int:
        """
        Client decrypts PIR response to get the retrieved item.

        Args:
            encrypted_response: Encrypted response from server

        Returns:
            Decrypted value/index
        """
        return self.crypto.decrypt(encrypted_response)

    def _calculate_query_size(self, encrypted_query: List[Dict]) -> int:
        """Calculate size of encrypted query in bytes."""
        # Estimate: each ciphertext has 2 vectors of dimension d, each element is ~8 bytes
        dimension = self.crypto.public_key['dimension']
        bytes_per_ciphertext = dimension * 2 * 8  # u and v vectors
        return len(encrypted_query) * bytes_per_ciphertext

    def _calculate_response_size(self, encrypted_response: Dict) -> int:
        """Calculate size of encrypted response in bytes."""
        dimension = self.crypto.public_key['dimension']
        return dimension * 2 * 8  # Single ciphertext size


class TiptoeHintSystem:
    """
    Implements Tiptoe's hint-based optimization system.

    The hint system allows most communication to happen offline,
    with only a small online component during actual queries.
    """

    def __init__(self, crypto_scheme: LinearHomomorphicScheme):
        self.crypto = crypto_scheme
        self.hint_data = None

    def generate_hint(self, database: List[np.ndarray]) -> Dict:
        """
        Generate cryptographic hint for database (offline preprocessing).

        Args:
            database: Server database to preprocess

        Returns:
            Hint data structure
        """
        start_time = time.perf_counter()

        # Simplified hint generation (in real Tiptoe this is much more complex)
        # Pre-encrypt some random values that can be reused
        hint_size = min(len(database), 1000)  # Limit hint size

        hint_data = {
            'database_size': len(database),
            'hint_size': hint_size,
            'preprocessed_values': [],
            'generation_time': 0
        }

        # Pre-generate some encrypted values for optimization
        for i in range(hint_size):
            hint_data['preprocessed_values'].append(self.crypto.encrypt(random.randint(0, 100)))

        hint_data['generation_time'] = time.perf_counter() - start_time
        self.hint_data = hint_data

        return hint_data

    def calculate_hint_communication(self) -> Dict:
        """Calculate communication cost of hint exchange."""
        if not self.hint_data:
            return {'hint_upload_bytes': 0, 'hint_download_bytes': 0}

        # Estimate hint communication cost
        hint_size = len(self.hint_data['preprocessed_values'])
        dimension = self.crypto.public_key['dimension']
        bytes_per_ciphertext = dimension * 2 * 8

        return {
            'hint_upload_bytes': hint_size * bytes_per_ciphertext * 0.1,  # Client sends small request
            'hint_download_bytes': hint_size * bytes_per_ciphertext,      # Server sends hint data
            'hint_generation_time': self.hint_data['generation_time']
        }


# === Real Homomorphic Encryption for Tiptoe Ranking (Pyfhel) ===
try:
    from Pyfhel import Pyfhel, PyCtxt
    _pyfhel_available = True
except ImportError:
    _pyfhel_available = False

class TiptoeHomomorphicRanking:
    """
    Real homomorphic encryption for Tiptoe ranking phase using Pyfhel (BFV).
    """
    def __init__(self, scheme: str = 'BFV', n_slots: int = 16384, t_bits: int = 20):
        if not _pyfhel_available:
            raise ImportError("Pyfhel is not installed. Please install it to use real homomorphic ranking.")
        self.HE = Pyfhel()
        try:
            # Use robust, recommended parameters for BFV
            self.HE.contextGen(scheme='BFV', n=n_slots, t_bits=t_bits)
            self.HE.keyGen()
        except Exception as e:
            raise RuntimeError(f"Pyfhel context/key generation failed: {e}")
        self.scheme = 'BFV'
        self.n_slots = n_slots
        self.t_bits = t_bits

    def encrypt_vector(self, vec):
        # Ensure input is np.int64 for BFV
        arr = np.array(vec, dtype=np.int64)
        return self.HE.encryptInt(arr)

    def decrypt_vector(self, ctxt):
        # Decrypt a ciphertext vector
        return self.HE.decryptFrac(ctxt) if self.scheme == 'CKKS' else self.HE.decryptInt(ctxt)

    def dot_product(self, ctxt_query, db_vecs):
        # db_vecs: shape (n_docs, dim)
        # ctxt_query: encrypted query vector
        # Returns: list of encrypted dot products (one per document)
        results = []
        for doc_vec in db_vecs:
            arr = np.array(doc_vec, dtype=np.int64)
            enc_doc = self.HE.encryptInt(arr)
            prod = ctxt_query * enc_doc
            dot = prod.sum()
            results.append(dot)
        return results

    def decrypt_scores(self, ctxt_scores):
        return [float(self.HE.decryptFrac(ctxt)) for ctxt in ctxt_scores]
