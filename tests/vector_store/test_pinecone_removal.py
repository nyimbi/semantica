import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure semantica is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from semantica.vector_store.vector_store import VectorStore
from semantica.vector_store.registry import method_registry
from semantica.vector_store.config import vector_store_config

class TestPineconeRemoval(unittest.TestCase):
    """Verify that Pinecone has been completely removed from the system."""

    def test_pinecone_backend_rejected(self):
        """Test that initializing VectorStore with backend='pinecone' raises an error."""
        with self.assertRaises(ValueError) as context:
            VectorStore(backend="pinecone")
        
        # The error message might be generic "Unknown backend" or specific.
        # We just want to ensure it fails.
        self.assertTrue("pinecone" in str(context.exception).lower() or "unknown" in str(context.exception).lower())

    def test_registry_clean(self):
        """Test that no Pinecone methods are registered."""
        # Check all task types
        task_types = ["store", "search", "index", "hybrid_search", "metadata", "namespace"]
        
        for task in task_types:
            methods = method_registry.list_all(task)
            # Flatten if it's a dict
            if isinstance(methods, dict):
                method_names = methods.get(task, [])
            else:
                method_names = methods
                
            for name in method_names:
                self.assertNotIn("pinecone", name.lower(), f"Found pinecone reference in registry task {task}: {name}")

    def test_config_clean(self):
        """Test that configuration does not contain Pinecone keys."""
        config = vector_store_config.get_all()
        
        for key in config.keys():
            self.assertNotIn("pinecone", key.lower(), f"Found pinecone key in config: {key}")

    def test_stores_existence(self):
        """Verify that other stores exist but PineconeStore does not."""
        try:
            from semantica.vector_store import faiss_store
            from semantica.vector_store import weaviate_store
            from semantica.vector_store import qdrant_store
            from semantica.vector_store import milvus_store
        except ImportError as e:
            self.fail(f"Failed to import a required store: {e}")

        with self.assertRaises(ImportError):
            from semantica.vector_store import pinecone_store

if __name__ == '__main__':
    unittest.main()
