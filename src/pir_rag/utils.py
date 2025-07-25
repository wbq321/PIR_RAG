"""
Utility functions for PIR-RAG system.
"""

import numpy as np
from typing import List, Tuple

# Global constants
CHUNK_SIZE = 64
MAX_CHUNKS_HOLDER = [0]


def encode_text_to_chunks(text_blob: str) -> List[int]:
    """
    Encode text into fixed-size integer chunks for PIR.
    
    Args:
        text_blob: Text to encode
        
    Returns:
        List of integers representing the text chunks
    """
    byte_data = text_blob.encode('utf-8')
    chunks = []
    for i in range(0, len(byte_data), CHUNK_SIZE):
        chunks.append(int.from_bytes(byte_data[i:i+CHUNK_SIZE], 'big'))
    return chunks


def decode_chunks_to_text(int_chunks: List[int]) -> str:
    """
    Decode integer chunks back to text.
    
    Args:
        int_chunks: List of integers representing text chunks
        
    Returns:
        Decoded text string
    """
    byte_data = b''.join(
        c.to_bytes((c.bit_length() + 7) // 8, 'big') for c in int_chunks if c != 0
    )
    return byte_data.decode('utf-8', errors='ignore')


def prepare_docs_by_size(all_docs: List[str], target_byte_size: int, 
                        tolerance: float, n_docs: int) -> List[str]:
    """
    Filter documents to match a target average byte size.
    
    Args:
        all_docs: List of all available documents
        target_byte_size: Target average byte size (e.g., 500, 1000, 2000)
        tolerance: Allowed size variance (e.g., 0.2 for 20%)
        n_docs: Number of documents to select
        
    Returns:
        List of selected documents
    """
    print(f"  -> Screening for {n_docs} docs with avg. byte size around {target_byte_size} bytes...")
    selected_docs = []
    
    # For reproducibility
    import random
    random.seed(42)
    shuffled_docs = random.sample(all_docs, len(all_docs))

    # Calculate target byte range
    lower_bound = (1 - tolerance) * target_byte_size
    upper_bound = (1 + tolerance) * target_byte_size

    for doc in shuffled_docs:
        if len(selected_docs) >= n_docs:
            break
        
        doc_byte_size = len(doc.encode('utf-8'))
        
        if lower_bound <= doc_byte_size <= upper_bound:
            selected_docs.append(doc)
    
    if len(selected_docs) < n_docs:
        print(f"Warning: Only found {len(selected_docs)} docs for target byte size {target_byte_size}. Using what we have.")
        if not selected_docs:
            raise ValueError(f"Could not find any documents for the specified byte size range [{lower_bound:.0f}, {upper_bound:.0f}].")

    return selected_docs[:n_docs]


def prepare_evaluation_data(corpus_texts: List[str], corpus_embeddings: np.ndarray, 
                          n_eval_queries: int = 100) -> Tuple[dict, dict]:
    """
    Create evaluation queries and ground truth from the corpus.
    
    Args:
        corpus_texts: List of corpus documents
        corpus_embeddings: Corresponding embeddings
        n_eval_queries: Number of evaluation queries to create
        
    Returns:
        Tuple of (eval_queries, eval_ground_truth) dictionaries
    """
    print(f"Preparing {n_eval_queries} evaluation queries...")
    eval_queries = {}
    eval_ground_truth = {}
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    sample_indices = np.random.choice(len(corpus_texts), n_eval_queries, replace=False)
    
    import torch
    
    for i, doc_idx in enumerate(sample_indices):
        qid = f"q_{i}"
        # Use the first 30 words as a query
        query_text = " ".join(corpus_texts[doc_idx].split()[:30])
        
        eval_queries[qid] = {
            "text": query_text,
            "embedding": torch.tensor(corpus_embeddings[doc_idx], dtype=torch.float32).unsqueeze(0)
        }
        # The ground truth is the original document itself
        eval_ground_truth[qid] = corpus_texts[doc_idx]
        
    return eval_queries, eval_ground_truth
