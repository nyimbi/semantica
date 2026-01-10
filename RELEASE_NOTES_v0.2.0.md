# Semantica v0.2.0 Release Notes

We are excited to announce the release of Semantica v0.2.0! This release brings major enhancements to graph database support, document parsing, extraction robustness, and provenance tracking.

## 🚀 Highlights

### Amazon Neptune Support
- **Native Integration**: Added `AmazonNeptuneStore` for full integration with Amazon Neptune via Bolt and OpenCypher.
- **Enterprise Security**: Implemented `NeptuneAuthTokenManager` for AWS IAM SigV4 signing with automatic token refresh.
- **Resilience**: Added robust connection handling with retry logic and backoff for transient errors.

### Docling Integration
- **High-Fidelity Parsing**: New `DoclingParser` in `semantica.parse` leverages the Docling library for superior document understanding.
- **Multi-Format Support**: Parse PDF, DOCX, PPTX, XLSX, HTML, and images with state-of-the-art table extraction.

### Robust Extraction Fallbacks
- **No More Empty Results**: Implemented a "ML/LLM -> Pattern -> Last Resort" fallback chain across all extractors.
- **Last Resort Strategies**:
  - **NER**: Identifies capitalized words as generic entities when models fail.
  - **Relations**: Infers weak connections between adjacent entities.

### Provenance & Tracking
- **Traceability**: Added `batch_index` and `document_id` metadata to all extracted elements (entities, relations, triplets).
- **Transparency**: Added count tracking to batch processing logs.

## 📋 Changelog

### Added
- **Amazon Neptune Support**:
    - Added `AmazonNeptuneStore` providing Amazon Neptune graph database integration via Bolt protocol and OpenCypher.
    - Implemented `NeptuneAuthTokenManager` extending Neo4j AuthManager for AWS IAM SigV4 signing with automatic token refresh.
    - Added robust connection handling: retry logic with backoff for transient errors (signature expired, connection closed) and driver recreation.
    - Added `graph-amazon-neptune` optional dependency group (boto3, neo4j).
    - Comprehensive test suite covering all GraphStore interface methods.
- **Docling Integration**:
    - Added `DoclingParser` in `semantica.parse` for high-fidelity document parsing using the Docling library.
    - Supports multi-format parsing (PDF, DOCX, PPTX, XLSX, HTML, images) with superior table extraction and structure understanding.
    - Implemented as a standalone parser supporting local execution, OCR, and multiple export formats (Markdown, HTML, JSON).
- **Robust Extraction Fallbacks**:
    - Implemented comprehensive fallback chains ("ML/LLM" -> "Pattern" -> "Last Resort") across `NERExtractor`, `RelationExtractor`, and `TripletExtractor` to prevent empty result lists.
    - Added "Last Resort" pattern matching in `NERExtractor` to identify capitalized words as generic entities when all other methods fail.
    - Added "Last Resort" adjacency-based relation extraction in `RelationExtractor` to create weak connections between adjacent entities if no relations are found.
    - Added fallback logic in `TripletExtractor` to convert relations to triplets or use rule-based extraction if standard methods fail.
- **Provenance & Tracking**:
    - Added count tracking to batch processing logs in `NERExtractor`, `RelationExtractor`, and `TripletExtractor`.
    - Added `batch_index` and `document_id` to the metadata of all extracted entities, relations, triplets, semantic roles, and clusters for better traceability.
- **Semantic Extract Improvements**:
    - Introduced `auto-chunking` for long text processing in LLM extraction methods (`extract_entities_llm`, `extract_relations_llm`, `extract_triplets_llm`).
    - Added `silent_fail` parameter to LLM extraction methods for configurable error handling.
    - Implemented robust JSON parsing and automatic retry logic (3 attempts with exponential backoff) in `BaseProvider` for all LLM providers.
    - Enhanced `GroqProvider` with better diagnostics and connectivity testing.
    - Added comprehensive entity, relation, and triplet deduplication for chunked extraction.
    - Added `semantica/semantic_extract/schemas.py` with canonical Pydantic models for consistent structured output.
- **Testing**:
    - Added comprehensive robustness test suite `tests/semantic_extract/test_robustness_fallback.py` for validating extraction fallbacks and metadata propagation.
    - Added comprehensive unit test suite `tests/embeddings/test_model_switching.py` for verifying dynamic model transitions and dimension updates.
    - Added end-to-end integration test suite for Knowledge Graph pipeline validation (GraphBuilder -> EntityResolver -> GraphAnalyzer).
- **Other**:
    - Added missing dependencies `GitPython` and `chardet` to `pyproject.toml`.
    - Robustified ID extraction across `CentralityCalculator`, `CommunityDetector`, and `ConnectivityAnalyzer` to handle various entity formats.
    - Improved `Entity` class hashability and equality logic in `utils/types.py`.

### Changed
- **Deduplication & Conflict Logic**:
    - Removed internal deduplication logic from `NERExtractor`, `RelationExtractor`, and `TripletExtractor`.
    - Removed consistency/conflict checking from `ExtractionValidator` to defer to dedicated `semantica/conflicts` module.
    - Removed `_deduplicate_*` methods from `semantica/semantic_extract/methods.py`.
- **Batch Processing & Consistency**:
    - Standardized batch processing across all extractors (`NERExtractor`, `RelationExtractor`, `TripletExtractor`, `SemanticNetworkExtractor`, `EventDetector`, `SemanticAnalyzer`, `CoreferenceResolver`) using a unified `extract`/`analyze`/`resolve` method pattern with progress tracking.
    - Added provenance metadata (`batch_index`, `document_id`) to `SemanticNetwork` nodes/edges, `Event` objects, `SemanticRole` results, `CoreferenceChain` mentions, and `SemanticCluster` (tracking source `document_ids`).
    - Updated `SemanticClusterer.cluster` and `SemanticAnalyzer.cluster_semantically` to accept list of dictionaries (with `content` and `id` keys) for better document tracking during clustering.
    - Removed legacy `check_triplet_consistency` from `TripletExtractor`.
    - Removed `validate_consistency` and `_check_consistency` from `ExtractionValidator`.
- **Weighted Scoring**:
    - Clarified weighted confidence scoring (50% Method Confidence + 50% Type Similarity) in comments.
    - Explicitly labeled "Type Similarity" as "user-provided" in code comments to remove ambiguity.
- **Refactoring**:
    - Fixed orchestrator lazy property initialization and configuration normalization logic in `Orchestrator`.
    - Verified and aligned `FileObject.text` property usage in GraphRAG notebooks for consistent content decoding.

### Fixed
- **Critical Fixes**:
    - Resolved `NameError` in `extraction_validator.py` by adding missing `Union` import.
    - Resolved issues where extractors would return empty lists for valid input text when primary extraction methods failed.
    - Fixed metadata initialization issue in batch processing where `batch_index` and `document_id` were occasionally missing from extracted items.
    - Ensured `LLMExtraction` methods (`enhance_entities`, `enhance_relations`) return original input instead of failing or returning empty results when LLM providers are unavailable.
- **Component Fixes**:
    - Fixed model switching bug in `TextEmbedder` where internal state was not cleared, preventing dynamic updates between `fastembed` and `sentence_transformers` (#160).
    - Implemented model-intrinsic embedding dimension detection in `TextEmbedder` to ensure consistency between models and vector databases.
    - Updated `set_model` to properly refresh configuration and dimensions during model switches.
    - Fixed `TypeError: unhashable type: 'Entity'` in `GraphAnalyzer` when processing graphs with raw `Entity` objects or dictionaries in relationships (#159).
    - Resolved `AssertionError` in orchestrator tests by aligning test mocks with production component usage.
    - Fixed dependency compatibility issues by pinning `protobuf==4.25.3` and `grpcio==1.67.1`.
    - Fixed a bug in `TripletExtractor` where the `validate_triplets` method was shadowed by an internal attribute.
    - Fixed incorrect `TextSplitter` import path in the `semantic_extract.methods` module.
