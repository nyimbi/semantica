# Refactor: Remove Pinecone and Enhance Vector Store Backend Support

## Summary
This Pull Request removes the proprietary Pinecone vector store dependency and its associated adapter to align Semantica with a strictly open-source, self-hostable architecture. It simultaneously enhances the robustness of the remaining vector store backends (FAISS, Weaviate, Qdrant, Milvus) through stricter validation and comprehensive deep-dive testing.

## Motivation
Semantica aims to provide a fully controllable, privacy-focused, and open-source foundation for semantic applications. Relying on proprietary cloud-only services like Pinecone contradicts this goal for the core framework. By removing Pinecone, we reduce external dependencies and focus on robust support for backends that can be run locally or self-hosted (FAISS, Weaviate, Qdrant, Milvus).

## Changes

### 1. Dependency Removal
- **Deleted**: `semantica/vector_store/pinecone_adapter.py`.
- **Removed**: Pinecone references from `pyproject.toml` (optional dependencies).
- **Cleaned**: Removed `PineconeAdapter` imports and exports from `semantica/vector_store/__init__.py`.

### 2. Core Vector Store Logic
- **Strict Validation**: Updated `VectorStore` in `semantica/vector_store/vector_store.py` to strictly validate backend names.
  - Supported backends: `faiss`, `weaviate`, `qdrant`, `milvus`, `inmemory`.
  - Raises `ValueError` for unsupported backends (e.g., "pinecone") instead of silent failures or defaults.
- **Config**: Removed Pinecone-specific configuration keys from `semantica/vector_store/config.py`.

### 3. Documentation & Cookbooks
- **Cookbook Updates**:
  - Rewrote `cookbook/introduction/13_Vector_Store.ipynb` to replace Pinecone examples with **Weaviate** (self-hosted alternative) and FAISS.
  - Updated `cookbook/advanced/11_Advanced_Context_Engineering.ipynb` and other notebooks to remove Pinecone references.
- **Docs**:
  - Updated `docs/modules.md`, `docs/architecture.md`, and `docs/reference/vector_store.md` to reflect the current supported backends.
  - Updated `CHANGELOG.md` to document the breaking change.

### 4. Testing
- **New Test**: Created `tests/vector_store/test_pinecone_removal.py` to explicitly verify that:
  - `VectorStore(backend="pinecone")` raises an error.
  - Pinecone components are not loadable.
- **Deep Dive Test**: Created `tests/vector_store/test_vector_store_deepdive.py` to validate:
  - **In-Memory** backend (logic verification without external deps).
  - **FAISS, Weaviate, Qdrant, Milvus** adapters (via mocking).
  - **Hybrid Search** and **Method Registry** functionality.
  - **VectorStoreConfig** management.

## Verification
- **Test Suite**: Ran `python -m pytest tests/vector_store/ -v`.
  - **Result**: 20/20 tests passed.
- **Manual Verification**:
  - Grepped codebase for "pinecone" (case-insensitive) to ensure 0 residual references.
  - Verified `13_Vector_Store.ipynb` runs successfully with Weaviate/FAISS.

## Breaking Changes
- **Pinecone Support Removed**: Users currently using `backend="pinecone"` or `PineconeAdapter` must migrate to a supported backend (e.g., Weaviate, Qdrant, Milvus) or use a custom adapter.
- **API Strictness**: `VectorStore` constructor now raises `ValueError` for invalid backend strings.
