"""
Script to verify the usage of the Semantica Core Module.
This simulates the typical usage pattern described in core_usage.md.
"""

import sys
import os
import logging

# Add project root to path to ensure we can import semantica
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantica import Semantica
from semantica.core import LifecycleManager, PluginRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("verify_core")

def custom_startup_hook():
    logger.info("✅ Custom startup hook executed!")

def custom_processing_method(sources, **kwargs):
    logger.info(f"✅ Custom processing method executed for sources: {sources}")
    return {"status": "success", "processed_items": len(sources)}

def main():
    logger.info("Starting Core Module Verification...")

    # 1. Initialize Semantica
    logger.info("\n--- Step 1: Initialization ---")
    config = {
        "project_name": "CoreVerification",
        "logging": {"level": "DEBUG"}
    }
    app = Semantica(config)
    logger.info("Semantica instance created.")

    # 2. Register Hooks via Lifecycle Manager
    logger.info("\n--- Step 2: Lifecycle Hooks ---")
    app.lifecycle_manager.register_startup_hook(custom_startup_hook, priority=10)
    logger.info("Startup hook registered.")

    # 3. Register Custom Method
    logger.info("\n--- Step 3: Method Registry ---")
    from semantica.core.registry import method_registry
    method_registry.register("knowledge_base", "custom_processor", custom_processing_method)
    logger.info("Custom method 'custom_processor' registered.")

    # 4. Start the System (Initialize)
    logger.info("\n--- Step 4: System Startup ---")
    app.initialize()
    
    # Check health
    health = app.lifecycle_manager.get_health_summary()
    logger.info(f"System Health: {'Healthy' if health['is_healthy'] else 'Unhealthy'}")
    if not health['is_healthy']:
        logger.warning(f"Unhealthy components: {health['unhealthy_components']}")

    # 5. Run a Workflow using the Custom Method
    logger.info("\n--- Step 5: Workflow Execution ---")
    sources = ["file1.txt", "file2.txt"]
    # We use the 'method' argument which the orchestrator (via methods.py) uses to look up the registry
    # Note: orchestrator.build_knowledge_base doesn't directly expose 'method' arg in signature but passes **kwargs to implementation
    # Let's check how methods.py is called. 
    # build_knowledge_base calls build_knowledge_base (wrapper) in methods.py? 
    # Wait, orchestrator.py: build_knowledge_base calls self._create_pipeline...
    
    # Actually, looking at orchestrator.py:
    # It calls self._create_pipeline(pipeline_config)
    # It doesn't seem to directly use 'method_registry' for the main 'build_knowledge_base' flow in the default implementation.
    # However, methods.py defines 'build_knowledge_base' which IS the implementation used if imported as functional API.
    # But Semantica class in orchestrator.py has its own build_knowledge_base method.
    
    # Let's see if we can use the method registry via the functional API or if we need to check how Semantica class uses it.
    # The Semantica class seems to have a hardcoded implementation in build_knowledge_base that creates a pipeline.
    # But wait, semantica/__init__.py likely exposes the class.
    
    # Let's try to invoke the custom method directly to verify registry, 
    # OR if Semantica class supports delegation (it might not currently).
    
    # Let's verify the functional API wrapper usage as well.
    from semantica.core.methods import build_knowledge_base as functional_build_kb
    
    result = functional_build_kb(sources, method="custom_processor", config=config)
    logger.info(f"Functional API Result: {result}")

    # 6. Shutdown
    logger.info("\n--- Step 6: Shutdown ---")
    app.lifecycle_manager.shutdown()
    logger.info("System shutdown completed.")

    logger.info("\n✅ Verification Completed Successfully!")

if __name__ == "__main__":
    main()
