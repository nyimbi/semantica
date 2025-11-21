# Quick Start Guide

Get started with Semantica in 5 minutes!

## Basic Example

```python
from semantica import Semantica

# Initialize Semantica
semantica = Semantica()

# Build knowledge graph from a document
result = semantica.build_knowledge_base(
    sources=["document.pdf"],
    embeddings=True,
    graph=True
)

# Access results
kg = result["knowledge_graph"]
embeddings = result["embeddings"]
statistics = result["statistics"]

print(f"Extracted {len(kg['entities'])} entities")
print(f"Created {len(kg['relationships'])} relationships")
```

## Extract Entities and Relationships

```python
from semantica import Semantica

semantica = Semantica()

# Extract from text
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

result = semantica.semantic_extract.extract_entities(text)
entities = result["entities"]

for entity in entities:
    print(f"{entity['text']} - {entity['type']}")
```

## Build Knowledge Graph

```python
from semantica import Semantica

semantica = Semantica()

# Build KG from multiple sources
sources = [
    "document1.pdf",
    "document2.docx",
    "https://example.com/article"
]

kg = semantica.kg.build_graph(sources)
semantica.kg.visualize(kg)
```

## Export Knowledge Graph

```python
from semantica import Semantica

semantica = Semantica()
kg = semantica.kg.build_graph(["data.pdf"])

# Export to different formats
semantica.export.to_rdf(kg, "output.rdf")
semantica.export.to_json(kg, "output.json")
semantica.export.to_csv(kg, "output.csv")
```

## Next Steps

- [Full Documentation](../README.md)
- [API Reference](api.md)
- [More Examples](examples.md)

