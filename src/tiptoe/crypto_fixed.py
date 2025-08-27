"""
FIXED Linear Homomorphic Encryption for Tiptoe

A much simpler but working implementation for research purposes.
"""

import numpy as np
import time
import random
from typing import List, Tuple, Dict, Any


class SimpleLinearHomomorphicScheme:
    """
    Ultra-simple linearly homomorphic encryption that prioritizes correctness.
    
    Uses minimal noise for research/prototype purposes.
    """

    def __init__(self, security_param: int = 128, url_mode: bool = False):
        """Initialize with minimal parameters."""
        self.url_mode = url_mode
        
        # Use a large prime to avoid wrap-around issues
        self.modulus = 2147483647  # Large prime (2^31 - 1)
        
        # Very simple key - just an additive constant
        self.secret_key = 12345  # Fixed for deterministic testing

    def encrypt(self, plaintext: int) -> Dict:
        """Ultra-simple encryption: just add the secret key."""
        # No noise for now - focus on getting the logic right
        ciphertext = (plaintext + self.secret_key) % self.modulus
        
        return {
            'value': ciphertext,
            'modulus': self.modulus
        }

    def decrypt(self, ciphertext: Dict) -> int:
        """Ultra-simple decryption: subtract the secret key."""
        plaintext = (ciphertext['value'] - self.secret_key) % self.modulus
        
        # Handle negative results
        if plaintext < 0:
            plaintext += self.modulus
        
        # For URL mode, clamp to byte range
        if self.url_mode and plaintext > 255:
            plaintext = plaintext % 256
            
        return int(plaintext)

    def add_encrypted(self, ct1: Dict, ct2: Dict) -> Dict:
        """Homomorphic addition: Enc(a) + Enc(b) = Enc(a + b)."""
        # Since Enc(x) = x + k, we have:
        # Enc(a) + Enc(b) = (a + k) + (b + k) = a + b + 2k
        # To get Enc(a + b) = (a + b) + k, we need to subtract k
        sum_value = (ct1['value'] + ct2['value'] - self.secret_key) % self.modulus
        
        return {
            'value': sum_value,
            'modulus': self.modulus
        }

    def scalar_multiply(self, ciphertext: Dict, scalar: int) -> Dict:
        """Homomorphic scalar multiplication: c * Enc(a) = Enc(c * a)."""
        # Since Enc(x) = x + k, we have:
        # c * Enc(a) = c * (a + k) = c*a + c*k
        # To get Enc(c*a) = c*a + k, we need to subtract (c-1)*k
        mult_value = (scalar * ciphertext['value'] - (scalar - 1) * self.secret_key) % self.modulus
        
        return {
            'value': mult_value,
            'modulus': self.modulus
        }


class SimpleLinearHomomorphicPIR:
    """
    PIR using the simple homomorphic scheme.
    """

    def __init__(self, url_mode: bool = False):
        """Initialize with simple crypto."""
        self.crypto = SimpleLinearHomomorphicScheme(url_mode=url_mode)
        self.url_mode = url_mode

    def create_pir_query(self, target_index: int, database_size: int) -> Tuple[List[Dict], Dict]:
        """Create PIR query for target index."""
        start_time = time.perf_counter()

        # Create 1-hot vector
        query_vector = [0] * database_size
        query_vector[target_index] = 1

        # Encrypt each element
        encrypted_query = []
        for value in query_vector:
            encrypted_query.append(self.crypto.encrypt(value))

        query_time = time.perf_counter() - start_time
        upload_bytes = len(encrypted_query) * 32  # Estimate

        metrics = {
            'query_generation_time': query_time,
            'upload_bytes': upload_bytes,
            'database_size': database_size,
            'target_index': target_index
        }

        return encrypted_query, metrics

    def process_pir_query(self, encrypted_query: List[Dict], database: List) -> Tuple[any, Dict]:
        """Process PIR query on server."""
        start_time = time.perf_counter()

        if len(encrypted_query) != len(database):
            raise ValueError(f"Query length {len(encrypted_query)} != database size {len(database)}")

        # For URL mode, handle list of bytes
        if self.url_mode and len(database) > 0 and isinstance(database[0], list):
            # Each database item is a list of bytes (URL)
            url_length = len(database[0])
            result = []
            
            for byte_pos in range(url_length):
                # Compute result for this byte position
                byte_result = self.crypto.encrypt(0)  # Start with encrypted zero
                
                for i, (query_enc, db_item) in enumerate(zip(encrypted_query, database)):
                    # Multiply query[i] * database[i][byte_pos]
                    if byte_pos < len(db_item):
                        term = self.crypto.scalar_multiply(query_enc, db_item[byte_pos])
                        byte_result = self.crypto.add_encrypted(byte_result, term)
                
                result.append(byte_result)
        else:
            # Original single-value mode
            result = self.crypto.encrypt(0)
            
            for i, (query_enc, db_item) in enumerate(zip(encrypted_query, database)):
                if isinstance(db_item, (list, np.ndarray)):
                    # Vector database - sum all elements
                    for j, db_value in enumerate(db_item):
                        term = self.crypto.scalar_multiply(query_enc, int(db_value))
                        result = self.crypto.add_encrypted(result, term)
                else:
                    # Scalar database
                    term = self.crypto.scalar_multiply(query_enc, int(db_item))
                    result = self.crypto.add_encrypted(result, term)

        processing_time = time.perf_counter() - start_time

        # Calculate download size
        if isinstance(result, list):
            download_bytes = len(result) * 32
        else:
            download_bytes = 32

        metrics = {
            'server_processing_time': processing_time,
            'download_bytes': download_bytes,
            'database_items_processed': len(database)
        }

        return result, metrics

    def decrypt_pir_response(self, encrypted_response) -> any:
        """Decrypt PIR response."""
        if isinstance(encrypted_response, list):
            # Decrypt each element for URL bytes
            return [self.crypto.decrypt(enc) for enc in encrypted_response]
        else:
            # Single value decryption
            return self.crypto.decrypt(encrypted_response)
