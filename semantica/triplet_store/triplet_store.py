"""
Triplet Store Core Module

This module provides the core triplet store interface and management classes,
providing a unified interface across multiple RDF store backends
(Blazegraph, Jena, RDF4J).

Key Features:
    - Unified triplet store interface
    - Multi-backend support (Blazegraph, Jena, RDF4J)
    - CRUD operations for RDF triplets
    - SPARQL query execution
    - Bulk loading and batch processing
    - Configuration management

Main Classes:
    - TripletStore: Main triplet store interface

Example Usage:
    >>> from semantica.triplet_store import TripletStore
    >>> store = TripletStore(backend="blazegraph", endpoint="http://localhost:9999/blazegraph")
    >>> store.add_triplet(triplet)
    >>> results = store.execute_query("SELECT * WHERE { ?s ?p ?o } LIMIT 10")

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union

from ..semantic_extract.triplet_extractor import Triplet
from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .bulk_loader import BulkLoader
from .config import triplet_store_config
from .query_engine import QueryEngine


class TripletStore:
    """
    Main triplet store interface.

    Provides a unified interface for working with RDF triple stores,
    supporting Blazegraph, Jena, and RDF4J backends.
    """

    SUPPORTED_BACKENDS = {"blazegraph", "jena", "rdf4j"}

    def __init__(
        self,
        backend: str = "blazegraph",
        endpoint: Optional[str] = None,
        **config,
    ):
        """
        Initialize triplet store.

        Args:
            backend: Backend type ("blazegraph", "jena", "rdf4j")
            endpoint: Store endpoint URL
            **config: Backend-specific configuration
        """
        self.logger = get_logger("triplet_store")
        self.progress_tracker = get_progress_tracker()

        # Validate backend
        if backend.lower() not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend}. "
                f"Supported backends are: {', '.join(sorted(self.SUPPORTED_BACKENDS))}"
            )

        self.backend_type = backend.lower()
        self.endpoint = endpoint
        self.config = config

        # Initialize store backend
        self._store_backend = None
        self._initialize_store_backend()

        # Initialize components
        self.query_engine = QueryEngine(self.config)
        self.bulk_loader = BulkLoader()

    def _initialize_store_backend(self) -> None:
        """Initialize the appropriate store backend based on backend type."""
        try:
            if self.backend_type == "blazegraph":
                from .blazegraph_store import BlazegraphStore
                
                # Merge config with defaults
                backend_config = self.config.copy()
                if self.endpoint:
                    backend_config["endpoint"] = self.endpoint
                
                self._store_backend = BlazegraphStore(**backend_config)

            elif self.backend_type == "jena":
                from .jena_store import JenaStore
                
                backend_config = self.config.copy()
                if self.endpoint:
                    # JenaStore might use different param names or structure
                    # Assuming JenaStore accepts endpoint or url
                    backend_config["url"] = self.endpoint 
                
                self._store_backend = JenaStore(**backend_config)

            elif self.backend_type == "rdf4j":
                from .rdf4j_store import RDF4JStore
                
                backend_config = self.config.copy()
                if self.endpoint:
                    backend_config["endpoint"] = self.endpoint
                
                self._store_backend = RDF4JStore(**backend_config)
                
            self.logger.info(f"Initialized {self.backend_type} backend")

        except Exception as e:
            self.logger.error(f"Failed to initialize {self.backend_type} backend: {e}")
            raise ProcessingError(f"Failed to initialize backend: {e}")

    def add_triplet(self, triplet: Triplet, **options) -> Dict[str, Any]:
        """
        Add a single triplet to the store.

        Args:
            triplet: Triplet object to add
            **options: Additional options

        Returns:
            Operation status
        """
        if not self._validate_triplet(triplet):
            raise ValidationError("Invalid triplet structure or confidence")

        return self._store_backend.add_triplet(triplet, **options)

    def add_triplets(
        self, 
        triplets: List[Triplet], 
        batch_size: int = 1000, 
        **options
    ) -> Dict[str, Any]:
        """
        Add multiple triplets to the store (bulk load).

        Args:
            triplets: List of Triplet objects
            batch_size: Batch size for processing
            **options: Additional options

        Returns:
            Operation status with stats
        """
        # Validate triplets first
        valid_triplets = [t for t in triplets if self._validate_triplet(t)]
        if len(valid_triplets) < len(triplets):
            self.logger.warning(
                f"Filtered {len(triplets) - len(valid_triplets)} invalid triplets"
            )

        # Use bulk loader for efficient processing
        progress = self.bulk_loader.load_triplets(
            valid_triplets, 
            self._store_backend, 
            batch_size=batch_size, 
            **options
        )

        return {
            "success": progress.failed_batches == 0,
            "total": progress.total_triplets,
            "processed": progress.processed_triplets,
            "failed": progress.failed_triplets,
            "batches": progress.total_batches
        }

    def get_triplets(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        **options,
    ) -> List[Triplet]:
        """
        Retrieve triplets matching criteria.

        Args:
            subject: Subject URI
            predicate: Predicate URI
            object: Object URI
            **options: Additional options

        Returns:
            List of matching Triplet objects
        """
        return self._store_backend.get_triplets(
            subject=subject, 
            predicate=predicate, 
            object=object, 
            **options
        )

    def delete_triplet(self, triplet: Triplet, **options) -> Dict[str, Any]:
        """
        Delete a triplet from the store.

        Args:
            triplet: Triplet to delete
            **options: Additional options

        Returns:
            Operation status
        """
        return self._store_backend.delete_triplet(triplet, **options)

    def update_triplet(
        self, 
        old_triplet: Triplet, 
        new_triplet: Triplet, 
        **options
    ) -> Dict[str, Any]:
        """
        Update a triplet (atomic delete + add).

        Args:
            old_triplet: Triplet to remove
            new_triplet: Triplet to add
            **options: Additional options

        Returns:
            Operation status
        """
        # Simple implementation: delete then add
        # Some backends might support atomic updates
        self.delete_triplet(old_triplet, **options)
        return self.add_triplet(new_triplet, **options)

    def execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None,
        **options
    ) -> Any:
        """
        Execute a SPARQL query.

        Args:
            query: SPARQL query string
            parameters: Query parameters
            **options: Additional options

        Returns:
            Query results (format depends on query type)
        """
        return self.query_engine.execute_query(query, self._store_backend, **options)

    def _validate_triplet(self, triplet: Triplet) -> bool:
        """Validate triplet structure."""
        if not triplet.subject or not triplet.predicate or not triplet.object:
            return False
        
        # Check confidence score if present
        if hasattr(triplet, 'confidence'):
            if triplet.confidence is not None and (triplet.confidence < 0 or triplet.confidence > 1):
                return False
                
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        if hasattr(self._store_backend, "get_stats"):
            return self._store_backend.get_stats()
        return {}
