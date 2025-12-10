# PR: Context Module Testing & Validation

## Description
This PR adds comprehensive testing and validation for the **Context Engineering Module** (`semantica.context`). It includes unit tests for core components, verification of notebook examples, and a critical bug fix in the deduplication module.

## Changes

### 1. New Unit Tests (`tests/context/`)
Added `tests/context/test_context.py` covering:
- **AgentContext**: End-to-end storage and retrieval (RAG & GraphRAG).
- **AgentMemory**: Hierarchical memory management (short-term buffer vs. long-term vector store) and retention policies.
- **ContextGraph**: Node/edge addition and neighbor traversal.
- **EntityLinker**: URI assignment and entity linking logic.
- **ContextRetriever**: Hybrid retrieval strategies (Vector + Graph).

### 2. Notebook Verification
Verified functionality of the following notebooks by converting them to test scripts:
- `19_Context_Module.ipynb`: Verified high-level interface, token limits, and graph construction.
- `11_Advanced_Context_Engineering.ipynb`: Verified custom memory pruning, hybrid tuning, and custom graph builders.

### 3. Bug Fixes
- **`semantica/deduplication/merge_strategy.py`**: Fixed a `NameError` caused by a missing `Tuple` import. This was discovered during global import validation.

### 4. Verification
- All new tests passed.
- Global import check confirmed no other hidden dependency issues.
- Integration test `verify_context_sync.py` passed, confirming correct synchronization between memory, graph, and vector store.

## Testing Instructions
Run the new tests with:
```bash
python -m unittest tests/context/test_context.py
```
