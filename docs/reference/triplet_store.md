# Triplet Store

> **Store and query RDF triplets with SPARQL support and semantic reasoning using industry-standard triplet stores.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } **RDF Storage**

    ---

    Store subject-predicate-object triplets in W3C-compliant RDF format

-   :material-code-braces:{ .lg .middle } **SPARQL Queries**

    ---

    Full W3C SPARQL 1.1 query language support for powerful semantic queries

-   :material-brain:{ .lg .middle } **Reasoning**

    ---

    RDFS and OWL reasoning for inference and knowledge discovery

-   :material-database-sync:{ .lg .middle } **Multiple Backends**

    ---

    Blazegraph, Apache Jena, and RDF4J support

-   :material-link-variant:{ .lg .middle } **Federation**

    ---

    Query across multiple triplet stores with SPARQL federation

-   :material-upload-multiple:{ .lg .middle } **Bulk Loading**

    ---

    High-performance bulk data loading with progress tracking

</div>

!!! tip "Choosing the Right Backend"
    - **Blazegraph**: High-performance, excellent for large datasets, GPU acceleration
    - **Apache Jena**: Full-featured, TDB2 storage, SHACL validation
    - **RDF4J**: Java-based, excellent tooling, multiple storage backends

---

## ⚙️ Algorithms Used

### Query Algorithms
- **SPARQL Query Optimization**: Join reordering with selectivity estimation
- **Triplet Pattern Matching**: Index-based lookup with B+ trees
- **Graph Pattern Matching**: Subgraph isomorphism with backtracking
- **Query Planning**: Cost-based optimization with statistics
- **Join Algorithms**: Hash join, merge join, nested loop join
- **Filter Pushdown**: Early filter application for performance

### Indexing
- **SPO Index**: Subject-Predicate-Object index for subject lookups
- **POS Index**: Predicate-Object-Subject index for predicate lookups
- **OSP Index**: Object-Subject-Predicate index for object lookups
- **Six-Index Scheme**: All permutations (SPO, SOP, PSO, POS, OSP, OPS) for optimal query performance
- **B+ Tree Indexing**: Efficient range queries and sorted access
- **Hash Indexing**: O(1) exact match lookups

### Reasoning Algorithms
- **RDFS Reasoning**: Subclass/subproperty inference, domain/range inference
- **OWL Reasoning**: Class hierarchy, property characteristics, cardinality constraints
- **Forward Chaining**: Materialization of inferred triplets
- **Backward Chaining**: On-demand inference during query execution
- **Rule-Based Inference**: Custom SWRL rules

### Bulk Loading
- **Batch Processing**: Chunked triplet insertion with configurable batch size
- **Parallel Loading**: Multi-threaded data loading
- **Index Building**: Deferred index construction for faster loading
- **Transaction Management**: Atomic batch commits with rollback support

---

## Main Classes

### TripletStore

Main interface for triplet store operations.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `__init__(backend, endpoint)` | Initialize triplet store | Factory pattern |
| `add_triplet(triplet)` | Add single triplet | Single insert |
| `add_triplets(triplets, batch_size)` | Add multiple triplets | Bulk load with batching |
| `get_triplets(s, p, o)` | Retrieve triplets | Pattern matching |
| `delete_triplet(triplet)` | Delete triplet | Pattern matching deletion |
| `execute_query(query)` | Execute SPARQL | Query engine delegation |

### BulkLoader

High-volume data loading utility.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `load_triplets(triplets, store)` | Bulk load triplets | Batch processing with retries |

### QueryEngine

SPARQL query execution and optimization engine.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `execute(query)` | Execute SPARQL query | Query execution |
| `optimize(query)` | Optimize SPARQL query | Query rewriting |

---

## 🚀 Usage

### Initialization

```python
from semantica.triplet_store import TripletStore

# Initialize Blazegraph store
store = TripletStore(
    backend="blazegraph",
    endpoint="http://localhost:9999/blazegraph"
)
```

### Adding Data

```python
from semantica.semantic_extract.triplet_extractor import Triplet

# Single triplet
triplet = Triplet("http://s", "http://p", "http://o")
store.add_triplet(triplet)

# Bulk load
triplets = [Triplet(f"http://s{i}", "http://p", "http://o") for i in range(1000)]
store.add_triplets(triplets)
```

### Querying

```python
query = """
SELECT ?s ?p ?o
WHERE {
  ?s ?p ?o
}
LIMIT 10
"""
results = store.execute_query(query)
```
