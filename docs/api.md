# API Reference

## Core Classes

### Semantica

Main framework class for building semantic layers and knowledge graphs.

```python
from semantica import Semantica

semantica = Semantica(config=None)
```

**Methods:**
- `build_knowledge_base(sources, **kwargs)` - Build knowledge base from sources
- `process_document(source)` - Process a single document
- `extract_entities(text)` - Extract entities from text
- `extract_relationships(text)` - Extract relationships from text

## Modules

### Knowledge Graph (`semantica.kg`)

```python
semantica.kg.build_graph(sources)
semantica.kg.analyze(graph)
semantica.kg.visualize(graph)
```

### Semantic Extraction (`semantica.semantic_extract`)

```python
semantica.semantic_extract.extract_entities(text)
semantica.semantic_extract.extract_relationships(text)
semantica.semantic_extract.extract_triples(text)
```

### Embeddings (`semantica.embeddings`)

```python
semantica.embeddings.generate(text)
semantica.embeddings.generate_batch(texts)
```

### Export (`semantica.export`)

```python
semantica.export.to_rdf(kg, path)
semantica.export.to_json(kg, path)
semantica.export.to_csv(kg, path)
```

## Configuration

```python
from semantica import Config

config = Config(
    embeddings=True,
    graph=True,
    normalize=True
)

semantica = Semantica(config=config)
```

For full API documentation, see [MODULES_DOCUMENTATION.md](../MODULES_DOCUMENTATION.md)

