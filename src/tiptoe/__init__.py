"""
Tiptoe: Private Web Search System

Python implementation of Tiptoe private search system based on:
"Private Web Search with Tiptoe" by Henzinger et al. (SOSP 2023)

Architecture:
- Phase 1: Offline preprocessing with PCA + clustering + cryptographic setup
- Phase 2: Two-phase PIR (cluster ranking + document retrieval)
- SimplePIR-like system for realistic performance

Updated to use SimplePIR-inspired PIR implementation instead of Paillier.
"""

from .system import TiptoeSystem
from .crypto import LinearHomomorphicScheme, TiptoeHomomorphicRanking
from .clustering import TiptoeClustering
from .simplepir import PIRServer, PIRClient, Matrix

__all__ = [
    'TiptoeSystem',
    'LinearHomomorphicScheme', 
    'TiptoeHomomorphicRanking',
    'TiptoeClustering',
    'PIRServer',
    'PIRClient',
    'Matrix'
]
