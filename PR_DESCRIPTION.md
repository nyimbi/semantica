# Refactor Semantic Extract Module to Class-Based Interfaces

## 📝 Summary
This PR refactors the Semantic Extract module to promote a cleaner, object-oriented API for Entity, Relation, and Triple extraction. It standardizes the usage around `NERExtractor`, `RelationExtractor`, and `TripleExtractor` classes, replacing the previous low-level `get_entity_method` factory functions in user-facing code.

## 🚀 Motivation
The previous API relied heavily on factory functions (`get_entity_method("pattern")`), which made discovery and configuration difficult for users. The new class-based approach:
- Improves code readability and IDE auto-completion.
- Provides a consistent interface (`extractor.extract()`) across all extraction tasks.
- Aligns the documentation and cookbooks with the actual best practices.

## 🔍 Key Changes

### 1. API Refactoring
- **Standardized Classes**: Promoted `NERExtractor`, `RelationExtractor`, and `TripleExtractor` as the primary entry points.
- **Method Aliases**: Added `extract()` aliases to `extract_entities()` and `extract_relations()` for a uniform API surface.
- **Configuration**: Unified configuration passing via class constructors.

### 2. Documentation Updates (`docs/reference/semantic_extract.md`)
- Added missing documentation for **Semantic Networks**, **Coreference Resolution**, and **LLM Enhancement**.
- Updated all code examples to use the new class-based API.
- Added a "Semantic Networks" card to the overview for better discoverability.

### 3. Cookbook Updates
- **`05_Entity_Extraction.ipynb`**: Refactored to use `NERExtractor` for Pattern, Regex, ML, and LLM examples.
- **`06_Relation_Extraction.ipynb`**: Refactored to use `RelationExtractor` for dependency and pattern-based examples.
- **`11_Chunking_and_Splitting.ipynb`**: Updated to use consistent method names (`ner_method="ml"`).

### 4. Split Module Improvements
- **Method Aliasing**: Added aliases in `methods.py` to support "spacy" (mapping to "ml") and "ml" (mapping to "dependency" for relations), improving robustness and user experience.
- **Robustness**: Verified `EntityAwareChunker` and `RelationAwareChunker` fallback mechanisms.

### 5. Testing
- Added `tests/test_ner_configurations.py` to verify all NER method configurations.
- Added `tests/test_notebooks_verification.py` to ensure notebook examples run correctly.
- Added `tests/test_semantic_extract_deepdive.py` covering relation and triple extraction scenarios.

## 🧪 Verification
- [x] **Unit Tests**: All new tests pass, verifying correct instantiation and execution of extractors.
- [x] **Notebooks**: Verified that the updated cookbooks run without errors.
- [x] **Documentation**: previewed `semantic_extract.md` to ensure correct rendering of new sections.

## ✅ Checklist
- [x] Code follows the project's coding standards.
- [x] Documentation has been updated to reflect the changes.
- [x] Tests have been added to cover the new functionality.
- [x] Cookbooks have been updated and verified.
