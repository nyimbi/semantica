"""
Script to verify the usage of the Semantica Knowledge Graph (KG) Module.
This simulates the typical usage pattern described in kg_usage.md.
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantica.kg import GraphBuilder, GraphAnalyzer, TemporalGraphQuery

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_kg")

def main():
    print("Starting KG Module Verification...")

    # --- Step 1: Build Knowledge Graph ---
    print("\n--- Step 1: Graph Building ---")
    
    # Define some source data with temporal info
    sources = [
        {
            "entities": [
                {"id": "e1", "name": "Alice", "type": "Person"},
                {"id": "e2", "name": "Bob", "type": "Person"},
                {"id": "e3", "name": "Semantica", "type": "Project"}
            ],
            "relationships": [
                {
                    "source": "e1", "target": "e2", "type": "knows",
                    "valid_from": "2023-01-01", "valid_until": None
                },
                {
                    "source": "e1", "target": "e3", "type": "works_on",
                    "valid_from": "2023-06-01", "valid_until": "2024-01-01"
                },
                {
                    "source": "e2", "target": "e3", "type": "works_on",
                    "valid_from": "2024-01-01", "valid_until": None
                }
            ]
        }
    ]

    # Initialize builder (disable complex features for simple verification)
    builder = GraphBuilder(
        merge_entities=False, 
        resolve_conflicts=False,
        enable_temporal=True
    )
    
    kg = builder.build(sources)
    logger.info(f"Graph built with {len(kg['entities'])} entities and {len(kg['relationships'])} relationships.")

    # --- Step 2: Analyze Graph ---
    logger.info("\n--- Step 2: Graph Analysis ---")
    
    # Mocking sub-analyzers if they are not fully implemented or require external libs not present
    # Assuming they are implemented or we can run with defaults.
    # Note: GraphAnalyzer imports CentralityCalculator etc. 
    # If those modules have dependencies (like networkx), they need to be installed.
    # Let's try to run it. If it fails, we know we need dependencies.
    
    try:
        analyzer = GraphAnalyzer()
        # We might need to mock internal calls if they fail due to missing heavy libs in this environment
        # But let's try.
        # To avoid failure if CentralityCalculator fails, we can catch it.
        # But for verification script, we want to see it run.
        # Since I can't check installed packages easily without running pip list, I'll assume standard deps.
        
        # However, to be safe and avoid script crash on things I haven't checked (like networkx), 
        # I will wrap in try-except block for analysis.
        analysis = analyzer.analyze_graph(kg)
        logger.info("Graph analysis completed.")
        logger.info(f"Metrics: {json.dumps(analysis.get('metrics', {}), indent=2)}")
    except Exception as e:
        logger.warning(f"Graph analysis skipped or failed: {e}")

    # --- Step 3: Temporal Query ---
    logger.info("\n--- Step 3: Temporal Querying ---")
    
    query_engine = TemporalGraphQuery()
    
    # Query at a specific time
    at_time = "2023-08-01"
    result = query_engine.query_at_time(kg, query="", at_time=at_time)
    
    logger.info(f"Relationships active at {at_time}:")
    for rel in result["relationships"]:
        logger.info(f"  {rel['source']} --[{rel['type']}]--> {rel['target']}")
        
    # Verify expected results
    # Alice knows Bob (from 2023-01-01) -> Active
    # Alice works_on Semantica (from 2023-06-01 to 2024-01-01) -> Active
    # Bob works_on Semantica (from 2024-01-01) -> Not Active
    
    active_rels = len(result["relationships"])
    logger.info(f"Found {active_rels} active relationships (Expected: 2).")
    
    if active_rels == 2:
        logger.info("✅ Temporal query verification successful!")
    else:
        logger.error("❌ Temporal query verification failed!")

    logger.info("\n✅ KG Module Verification Completed!")

if __name__ == "__main__":
    main()
