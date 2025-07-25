"""
PIR-RAG: Private Information Retrieval for Retrieval-Augmented Generation

A privacy-preserving RAG system that uses homomorphic encryption to enable
private document retrieval while maintaining high-quality semantic search.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .client import PIRRAGClient
from .server import PIRRAGServer
from .utils import encode_text_to_chunks, decode_chunks_to_text

__all__ = [
    "PIRRAGClient",
    "PIRRAGServer", 
    "encode_text_to_chunks",
    "decode_chunks_to_text"
]
