"""
Simplified PianoPIR implementation for communication efficiency comparison.
Based on the Go implementation in private-search-temp.
"""

import numpy as np
import hashlib
from typing import List, Tuple, Dict, Any
import time
import math
import random


class PianoPIRConfig:
    """Configuration for PianoPIR"""
    def __init__(self, db_size: int, db_entry_byte_num: int, 
                 chunk_size: int = 1024, set_size: int = 1024, 
                 failure_prob_log2: int = 40):
        self.db_size = db_size
        self.db_entry_byte_num = db_entry_byte_num
        self.db_entry_size = db_entry_byte_num // 8  # uint64 entries
        self.chunk_size = chunk_size
        self.set_size = set_size
        self.failure_prob_log2 = failure_prob_log2


class PianoPIRClient:
    """Simplified PianoPIR Client for communication efficiency measurement"""
    
    def __init__(self, config: PianoPIRConfig):
        self.config = config
        self.master_key = self._generate_key()
        
        # Calculate query limits based on Go implementation
        self.max_query_num = int(math.sqrt(config.db_size) * math.log(config.db_size))
        self.finished_query_num = 0
        
        # Communication tracking
        self.total_upload_bytes = 0
        self.total_download_bytes = 0
        self.query_count = 0
        
    def _generate_key(self) -> bytes:
        """Generate a random master key"""
        return hashlib.sha256(str(random.getrandbits(256)).encode()).digest()
    
    def preprocessing(self) -> Dict[str, Any]:
        """Simulate preprocessing phase"""
        start_time = time.perf_counter()
        
        # Simulate key expansion and hint generation
        # In real implementation, this would involve cryptographic operations
        time.sleep(0.001)  # Simulate computation
        
        preprocessing_time = time.perf_counter() - start_time
        
        return {
            "preprocessing_time": preprocessing_time,
            "max_queries": self.max_query_num
        }
    
    def generate_query(self, indices: List[int]) -> Tuple[List[bytes], int]:
        """
        Generate PIR query for multiple indices (batch query).
        
        Args:
            indices: List of database indices to query
            
        Returns:
            Tuple of (query_data, upload_bytes)
        """
        # Simulate query generation
        query_size_per_index = 32  # bytes per query (simplified)
        query_data = []
        
        for idx in indices:
            # Simulate query generation with PRF and encryption
            query_bytes = hashlib.sha256(
                self.master_key + idx.to_bytes(8, 'little')
            ).digest()
            query_data.append(query_bytes)
        
        # Calculate communication cost
        upload_bytes = len(indices) * query_size_per_index
        self.total_upload_bytes += upload_bytes
        self.query_count += len(indices)
        
        return query_data, upload_bytes
    
    def decode_response(self, response_data: List[bytes]) -> Tuple[List[np.ndarray], int]:
        """
        Decode PIR response to recover the requested data.
        
        Args:
            response_data: Encrypted response from server
            
        Returns:
            Tuple of (decoded_entries, download_bytes)
        """
        decoded_entries = []
        download_bytes = sum(len(resp) for resp in response_data)
        
        for resp in response_data:
            # Simulate decryption and decoding
            # In real implementation, this would involve cryptographic operations
            decoded_entry = np.frombuffer(
                hashlib.sha256(resp + self.master_key).digest()[:self.config.db_entry_byte_num], 
                dtype=np.uint8
            )
            decoded_entries.append(decoded_entry)
        
        self.total_download_bytes += download_bytes
        return decoded_entries, download_bytes
    
    def get_communication_stats(self) -> Dict[str, float]:
        """Get communication statistics"""
        return {
            "total_upload_bytes": self.total_upload_bytes,
            "total_download_bytes": self.total_download_bytes,
            "total_communication_bytes": self.total_upload_bytes + self.total_download_bytes,
            "avg_upload_per_query": self.total_upload_bytes / max(1, self.query_count),
            "avg_download_per_query": self.total_download_bytes / max(1, self.query_count),
            "query_count": self.query_count
        }


class PianoPIRServer:
    """Simplified PianoPIR Server for communication efficiency measurement"""
    
    def __init__(self, config: PianoPIRConfig, raw_db: np.ndarray):
        self.config = config
        self.raw_db = raw_db  # Database as numpy array
        
    def handle_query(self, query_data: List[bytes]) -> List[bytes]:
        """
        Handle PIR query and return encrypted response.
        
        Args:
            query_data: List of query bytes from client
            
        Returns:
            List of encrypted response bytes
        """
        response_data = []
        
        for query_bytes in query_data:
            # Simulate server-side PIR computation
            # In real implementation, this would involve cryptographic operations
            
            # Simulate database access pattern (for communication measurement)
            response_size = self.config.db_entry_byte_num + 16  # entry + MAC
            response_bytes = hashlib.sha256(query_bytes).digest()[:response_size]
            response_data.append(response_bytes)
        
        return response_data
    
    def non_private_query(self, indices: List[int]) -> List[np.ndarray]:
        """Non-private baseline for comparison"""
        results = []
        for idx in indices:
            if idx < len(self.raw_db):
                results.append(self.raw_db[idx])
            else:
                # Return zero entry if out of bounds
                results.append(np.zeros(self.config.db_entry_size, dtype=np.uint64))
        return results


class SimpleBatchPianoPIR:
    """
    Simplified Batch PianoPIR implementation.
    Manages multiple PIR instances for batch processing.
    """
    
    def __init__(self, db_size: int, db_entry_byte_num: int, 
                 batch_size: int, raw_db: np.ndarray):
        self.db_size = db_size
        self.db_entry_byte_num = db_entry_byte_num
        self.batch_size = batch_size
        self.raw_db = raw_db
        
        # Partitioning setup
        self.partition_num = max(1, batch_size // 2)  # Simplified partitioning
        self.partition_size = (db_size + self.partition_num - 1) // self.partition_num
        
        # Create sub-PIR instances for each partition
        self.sub_pir_clients = []
        self.sub_pir_servers = []
        
        for i in range(self.partition_num):
            start_idx = i * self.partition_size
            end_idx = min((i + 1) * self.partition_size, db_size)
            partition_size = end_idx - start_idx
            
            config = PianoPIRConfig(
                db_size=partition_size,
                db_entry_byte_num=db_entry_byte_num
            )
            
            client = PianoPIRClient(config)
            server = PianoPIRServer(config, raw_db[start_idx:end_idx])
            
            self.sub_pir_clients.append(client)
            self.sub_pir_servers.append(server)
        
        # Statistics
        self.total_queries_made = 0
        self.preprocessing_time = 0
        
    def preprocessing(self):
        """Preprocess all sub-PIR instances"""
        start_time = time.perf_counter()
        
        for client in self.sub_pir_clients:
            client.preprocessing()
            
        self.preprocessing_time = time.perf_counter() - start_time
    
    def batch_query(self, indices: List[int]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Perform batch PIR query for multiple indices.
        
        Args:
            indices: List of database indices to query
            
        Returns:
            Tuple of (results, communication_stats)
        """
        # Organize queries by partition
        partition_queries = [[] for _ in range(self.partition_num)]
        index_to_partition = {}
        
        for idx in indices:
            partition_id = min(idx // self.partition_size, self.partition_num - 1)
            local_idx = idx - partition_id * self.partition_size
            partition_queries[partition_id].append(local_idx)
            index_to_partition[idx] = (partition_id, len(partition_queries[partition_id]) - 1)
        
        # Execute queries on each partition
        results = [None] * len(indices)
        total_stats = {
            "total_upload_bytes": 0,
            "total_download_bytes": 0,
            "partitions_queried": 0
        }
        
        for partition_id, local_indices in enumerate(partition_queries):
            if not local_indices:
                continue
                
            total_stats["partitions_queried"] += 1
            client = self.sub_pir_clients[partition_id]
            server = self.sub_pir_servers[partition_id]
            
            # Generate and send query
            query_data, upload_bytes = client.generate_query(local_indices)
            response_data = server.handle_query(query_data)
            decoded_entries, download_bytes = client.decode_response(response_data)
            
            # Map results back to original indices
            for i, original_idx in enumerate([idx for idx in indices 
                                            if idx // self.partition_size == partition_id]):
                partition_pos = index_to_partition[original_idx][1]
                if partition_pos < len(decoded_entries):
                    results[indices.index(original_idx)] = decoded_entries[partition_pos]
            
            total_stats["total_upload_bytes"] += upload_bytes
            total_stats["total_download_bytes"] += download_bytes
        
        self.total_queries_made += len(indices)
        
        # Fill any missing results with zeros
        for i in range(len(results)):
            if results[i] is None:
                results[i] = np.zeros(self.db_entry_byte_num, dtype=np.uint8)
        
        return results, total_stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics"""
        client_stats = [client.get_communication_stats() for client in self.sub_pir_clients]
        
        total_upload = sum(stats["total_upload_bytes"] for stats in client_stats)
        total_download = sum(stats["total_download_bytes"] for stats in client_stats)
        
        return {
            "preprocessing_time": self.preprocessing_time,
            "total_queries_made": self.total_queries_made,
            "total_upload_bytes": total_upload,
            "total_download_bytes": total_download,
            "total_communication_bytes": total_upload + total_download,
            "partition_count": self.partition_num,
            "avg_communication_per_query": (total_upload + total_download) / max(1, self.total_queries_made)
        }
