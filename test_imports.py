#!/usr/bin/env python3
"""Simple test script to check for errors."""

print("Testing basic imports...")

try:
    import yaml
    print("✓ yaml imported")
except ImportError as e:
    print(f"✗ yaml import failed: {e}")

try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    print("✓ path setup")
except Exception as e:
    print(f"✗ path setup failed: {e}")

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ config.yaml loaded successfully")
    print(f"  - Found {len(config['experiments'])} experiment types")
    print(f"  - cluster_count enabled: {config['experiments']['cluster_count']['enabled']}")
except Exception as e:
    print(f"✗ config.yaml failed: {e}")

try:
    from pir_rag import PIRRAGClient, PIRRAGServer
    print("✓ PIR-RAG modules imported")
except Exception as e:
    print(f"✗ PIR-RAG import failed: {e}")

print("Test complete!")
