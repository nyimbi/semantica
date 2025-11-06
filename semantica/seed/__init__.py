"""
Seed Data System Module

This module provides comprehensive seed data management for initial knowledge
graph construction, enabling the framework to build on existing verified
knowledge from multiple sources.

Key Features:
    - Multi-source seed data loading (CSV, JSON, Database, API)
    - Foundation graph creation from seed data
    - Seed data quality validation
    - Integration with extracted data
    - Version management for seed sources
    - Export capabilities (JSON, CSV)

Main Classes:
    - SeedDataManager: Main coordinator for seed data operations
    - SeedDataSource: Seed data source definition
    - SeedData: Seed data container

Example Usage:
    >>> from semantica.seed import SeedDataManager
    >>> manager = SeedDataManager()
    >>> manager.register_source("entities", "json", "data/entities.json")
    >>> foundation = manager.create_foundation_graph()
    >>> validation = manager.validate_quality(foundation)

Author: Semantica Contributors
License: MIT
"""

from .seed_manager import (
    SeedDataManager,
    SeedDataSource,
    SeedData,
)

__all__ = [
    "SeedDataManager",
    "SeedDataSource",
    "SeedData",
]
