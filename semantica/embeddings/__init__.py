"""
Embeddings Generation Module

This module provides comprehensive embedding generation and management capabilities.

Exports:
    - EmbeddingGenerator: Main embedding generation class
    - TextEmbedder: Text embedding generation
    - ImageEmbedder: Image embedding generation
    - AudioEmbedder: Audio embedding generation
    - MultimodalEmbedder: Multi-modal embedding support
    - EmbeddingOptimizer: Embedding optimization and fine-tuning
    - ContextManager: Embedding context management
    - ProviderAdapters: Provider-specific adapters
    - build: Module-level build function for embedding generation
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from .embedding_generator import EmbeddingGenerator
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder
from .audio_embedder import AudioEmbedder
from .multimodal_embedder import MultimodalEmbedder
from .embedding_optimizer import EmbeddingOptimizer
from .context_manager import ContextManager, ContextWindow
from .provider_adapters import (
    ProviderAdapter,
    OpenAIAdapter,
    BGEAdapter,
    LlamaAdapter,
    ProviderAdapterFactory,
)
from .pooling_strategies import (
    PoolingStrategy,
    MeanPooling,
    MaxPooling,
    CLSPooling,
    AttentionPooling,
    HierarchicalPooling,
    PoolingStrategyFactory,
)

__all__ = [
    "EmbeddingGenerator",
    "TextEmbedder",
    "ImageEmbedder",
    "AudioEmbedder",
    "MultimodalEmbedder",
    "EmbeddingOptimizer",
    "ContextManager",
    "ContextWindow",
    # Provider adapters
    "ProviderAdapter",
    "OpenAIAdapter",
    "BGEAdapter",
    "LlamaAdapter",
    "ProviderAdapterFactory",
    # Pooling strategies
    "PoolingStrategy",
    "MeanPooling",
    "MaxPooling",
    "CLSPooling",
    "AttentionPooling",
    "HierarchicalPooling",
    "PoolingStrategyFactory",
    "build",
]


def build(
    data: Union[str, Path, List[Union[str, Path]]],
    data_type: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: int = 32,
    **options
) -> Dict[str, Any]:
    """
    Generate embeddings from data (module-level convenience function).
    
    This is a user-friendly wrapper around EmbeddingGenerator.generate_embeddings()
    that creates an EmbeddingGenerator instance and generates embeddings.
    
    Args:
        data: Input data - can be text string, file path, or list of texts/paths
        data_type: Data type - "text", "image", "audio" (auto-detected if None)
        model: Embedding model to use (default: None, uses default model)
        batch_size: Batch size for processing multiple items (default: 32)
        **options: Additional generation options
        
    Returns:
        Dictionary containing:
            - embeddings: Generated embeddings (numpy array or list of arrays)
            - metadata: Embedding metadata
            - statistics: Generation statistics
            
    Examples:
        >>> import semantica
        >>> result = semantica.embeddings.build(
        ...     data=["text1", "text2", "text3"],
        ...     data_type="text",
        ...     model="sentence-transformers"
        ... )
        >>> print(f"Generated {len(result['embeddings'])} embeddings")
    """
    # Create EmbeddingGenerator instance
    config = {}
    if model:
        config["model"] = model
    config.update(options)
    
    generator = EmbeddingGenerator(config=config, **options)
    
    # Normalize data to list if single item
    is_single = not isinstance(data, list)
    if is_single:
        data = [data]
    
    # Generate embeddings
    if len(data) == 1:
        # Single item
        embeddings = generator.generate_embeddings(data[0], data_type=data_type, **options)
        if is_single:
            # Return single embedding array
            return {
                "embeddings": embeddings,
                "metadata": {
                    "data_type": data_type or generator._detect_data_type(data[0]),
                    "model": model or "default",
                    "shape": embeddings.shape if isinstance(embeddings, np.ndarray) else None
                },
                "statistics": {
                    "total_items": 1,
                    "successful": 1,
                    "failed": 0
                }
            }
    else:
        # Batch processing
        results = generator.process_batch(data, **options)
        return {
            "embeddings": results["embeddings"],
            "metadata": {
                "data_type": data_type or "auto-detected",
                "model": model or "default",
                "batch_size": batch_size
            },
            "statistics": {
                "total_items": results["total"],
                "successful": results["success_count"],
                "failed": results["failure_count"]
            },
            "failed_items": results.get("failed", [])
        }
