"""
Tiptoe: Private Web Search System

Python implementation of Tiptoe private search system based on:
"Private Web Search with Tiptoe" by Henzinger et al. (SOSP 2023)

Architecture:
- Phase 1: Offline preprocessing with PCA + clustering + cryptographic setup
- Phase 2: Two-phase PIR (cluster ranking + document retrieval)
- Linearly homomorphic encryption for privacy
"""

from .system import TiptoeSystem
from .crypto import LinearHomomorphicPIR, LinearHomomorphicScheme
from .clustering import TiptoeClustering

__all__ = [
    'TiptoeSystem',
    'LinearHomomorphicPIR', 
    'LinearHomomorphicScheme',
    'TiptoeClustering'
]
