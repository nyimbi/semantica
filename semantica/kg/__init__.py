"""
Knowledge Graph Management Module

This module provides comprehensive knowledge graph construction and management capabilities,
including temporal knowledge graph support for time-aware knowledge representation.

Exports:
    - GraphBuilder: Knowledge graph construction with temporal support
    - EntityResolver: Entity resolution and deduplication
    - GraphAnalyzer: Graph analytics with temporal evolution analysis
    - TemporalGraphQuery: Time-aware graph querying
    - TemporalPatternDetector: Temporal pattern detection
    - TemporalVersionManager: Temporal versioning and snapshots
    - ConflictDetector: Conflict detection and resolution
    - ProvenanceTracker: Provenance tracking and management
    - CentralityCalculator: Centrality measures calculation
    - CommunityDetector: Community detection
    - ConnectivityAnalyzer: Connectivity analysis
    - Deduplicator: Graph deduplication
    - GraphValidator: Graph validation
    - SeedManager: Seed data management
    - build: Module-level build function for knowledge graph construction
"""

from typing import Any, Dict, List, Optional, Union

from .graph_builder import GraphBuilder
from .entity_resolver import EntityResolver
from .graph_analyzer import GraphAnalyzer
from .temporal_query import TemporalGraphQuery, TemporalPatternDetector, TemporalVersionManager
from .conflict_detector import ConflictDetector
from .provenance_tracker import ProvenanceTracker
from .centrality_calculator import CentralityCalculator
from .community_detector import CommunityDetector
from .connectivity_analyzer import ConnectivityAnalyzer
from .deduplicator import Deduplicator
from .graph_validator import GraphValidator
from .seed_manager import SeedManager

__all__ = [
    "GraphBuilder",
    "EntityResolver",
    "GraphAnalyzer",
    "TemporalGraphQuery",
    "TemporalPatternDetector",
    "TemporalVersionManager",
    "ConflictDetector",
    "ProvenanceTracker",
    "CentralityCalculator",
    "CommunityDetector",
    "ConnectivityAnalyzer",
    "Deduplicator",
    "GraphValidator",
    "SeedManager",
    "build",
]


def build(
    sources: Union[List[Any], Any],
    merge_entities: bool = True,
    entity_resolution_strategy: str = "fuzzy",
    resolve_conflicts: bool = True,
    enable_temporal: bool = False,
    temporal_granularity: str = "day",
    track_history: bool = False,
    version_snapshots: bool = False,
    entity_resolver: Optional[EntityResolver] = None,
    **options
) -> Dict[str, Any]:
    """
    Build knowledge graph from sources (module-level convenience function).
    
    This is a user-friendly wrapper around GraphBuilder.build() that creates
    a GraphBuilder instance and builds the knowledge graph.
    
    Args:
        sources: List of sources (documents, entities, relationships, or dicts with entities/relationships)
        merge_entities: Whether to merge duplicate entities (default: True)
        entity_resolution_strategy: Strategy for entity resolution - "fuzzy", "exact", "semantic" (default: "fuzzy")
        resolve_conflicts: Whether to resolve conflicts (default: True)
        enable_temporal: Enable temporal knowledge graph features (default: False)
        temporal_granularity: Time granularity - "second", "minute", "hour", "day", etc. (default: "day")
        track_history: Track historical changes (default: False)
        version_snapshots: Create version snapshots (default: False)
        entity_resolver: Optional EntityResolver instance (default: None, creates one if needed)
        **options: Additional build options
        
    Returns:
        Dictionary containing:
            - entities: List of entities
            - relationships: List of relationships
            - metadata: Graph metadata including counts and timestamps
            
    Examples:
        >>> import semantica
        >>> result = semantica.kg.build(
        ...     sources=[{"entities": [...], "relationships": [...]}],
        ...     merge_entities=True,
        ...     resolve_conflicts=True
        ... )
        >>> print(f"Built graph with {result['metadata']['num_entities']} entities")
    """
    # Normalize sources to list
    if not isinstance(sources, list):
        sources = [sources]
    
    # Create GraphBuilder instance
    graph_builder = GraphBuilder(
        merge_entities=merge_entities,
        entity_resolution_strategy=entity_resolution_strategy,
        resolve_conflicts=resolve_conflicts,
        enable_temporal=enable_temporal,
        temporal_granularity=temporal_granularity,
        track_history=track_history,
        version_snapshots=version_snapshots,
        **options
    )
    
    # Build knowledge graph
    graph = graph_builder.build(sources, entity_resolver=entity_resolver, **options)
    
    return graph
