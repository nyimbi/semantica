import os
import shutil
import tempfile
from pathlib import Path
from semantica.ingest import ingest, ingest_ontology, OntologyData

def demo_ontology_ingestion():
    print("=== Ontology Ingestion Demo ===")
    
    # Create a sample ontology file
    sample_ttl = """
    @prefix : <http://example.org/demo/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    <http://example.org/demo/> rdf:type owl:Ontology ;
        rdfs:label "Demo Ontology" ;
        rdfs:comment "A simple ontology for demonstration." .

    :DemoClass rdf:type owl:Class ;
        rdfs:label "Demo Class" .
    """
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ttl", mode="w") as tmp:
        tmp.write(sample_ttl)
        tmp_path = tmp.name
        
    print(f"\nCreated temporary ontology file: {tmp_path}")
    
    try:
        # 1. Use ingest_ontology convenience function
        print("\n--- Method 1: ingest_ontology() ---")
        result = ingest_ontology(tmp_path)
        
        if isinstance(result, OntologyData):
            print(f"Success! Ingested ontology: {result.data.get('name')}")
            print(f"Format: {result.metadata.get('format')}")
            print(f"Classes found: {len(result.data.get('classes', []))}")
            for cls in result.data.get('classes', []):
                print(f"  - {cls.get('name')} ({cls.get('uri')})")
        else:
            print("Unexpected result type:", type(result))

        # 2. Use unified ingest function
        print("\n--- Method 2: Unified ingest() ---")
        # Explicitly setting source_type="ontology" ensures it uses OntologyIngestor
        unified_result = ingest(tmp_path, source_type="ontology")
        
        if "ontology" in unified_result:
            ont_data = unified_result["ontology"]
            if isinstance(ont_data, OntologyData):
                 print(f"Success! Ingested via unified interface.")
                 print(f"Ontology Name: {ont_data.data.get('name')}")
            else:
                 print(f"Got 'ontology' key but value is {type(ont_data)}")
        else:
            print("Unified ingest result keys:", unified_result.keys())

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"\nCleaned up temporary file.")

if __name__ == "__main__":
    demo_ontology_ingestion()
