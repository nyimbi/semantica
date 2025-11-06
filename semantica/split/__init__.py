"""
Split Module

This module provides comprehensive document chunking and splitting capabilities
for optimal processing and semantic analysis, enabling efficient handling of
large documents through various chunking strategies.

Key Features:
    - Semantic-based chunking using NLP
    - Structure-aware chunking (headings, paragraphs, lists)
    - Sliding window chunking with overlap
    - Table-specific chunking
    - Chunk validation and quality assessment
    - Provenance tracking for data lineage

Main Classes:
    - SemanticChunker: Semantic-based chunking coordinator
    - StructuralChunker: Structure-aware chunking
    - SlidingWindowChunker: Fixed-size sliding window chunking
    - TableChunker: Table-specific chunking
    - ChunkValidator: Chunk quality validation
    - ProvenanceTracker: Chunk provenance tracking
    - Chunk: Chunk representation dataclass

Example Usage:
    >>> from semantica.split import SemanticChunker
    >>> chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
    >>> chunks = chunker.chunk(long_text)
    >>> from semantica.split import StructuralChunker
    >>> struct_chunker = StructuralChunker()
    >>> chunks = struct_chunker.chunk(structured_document)

Author: Semantica Contributors
License: MIT
"""

from .semantic_chunker import SemanticChunker, Chunk
from .structural_chunker import StructuralChunker
from .sliding_window_chunker import SlidingWindowChunker
from .table_chunker import TableChunker
from .chunk_validator import ChunkValidator
from .provenance_tracker import ProvenanceTracker

__all__ = [
    "SemanticChunker",
    "Chunk",
    "StructuralChunker",
    "SlidingWindowChunker",
    "TableChunker",
    "ChunkValidator",
    "ProvenanceTracker",
]
