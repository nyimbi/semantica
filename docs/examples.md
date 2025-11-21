# Examples

## Example 1: Basic Knowledge Graph

```python
from semantica import Semantica

semantica = Semantica()

# Build KG from PDF
result = semantica.build_knowledge_base(
    sources=["research_paper.pdf"],
    embeddings=True,
    graph=True
)

kg = result["knowledge_graph"]
print(f"Entities: {len(kg['entities'])}")
print(f"Relationships: {len(kg['relationships'])}")
```

## Example 2: Entity Extraction

```python
from semantica import Semantica

semantica = Semantica()

text = """
Apple Inc. is a technology company founded by Steve Jobs.
The company is headquartered in Cupertino, California.
"""

entities = semantica.semantic_extract.extract_entities(text)
for entity in entities:
    print(f"{entity['text']}: {entity['type']}")
```

## Example 3: Multi-Source Integration

```python
from semantica import Semantica

semantica = Semantica()

sources = [
    "documents/finance_report.pdf",
    "documents/market_analysis.docx",
    "https://example.com/news-article"
]

result = semantica.build_knowledge_base(sources)
```

## Example 4: Export Formats

```python
from semantica import Semantica

semantica = Semantica()
kg = semantica.kg.build_graph(["data.pdf"])

# Export to multiple formats
semantica.export.to_rdf(kg, "output.rdf")
semantica.export.to_json(kg, "output.json")
semantica.export.to_owl(kg, "output.owl")
```

## More Examples

- [Code Examples](../CodeExamples.md)
- [Cookbook Notebooks](../cookbook/)

