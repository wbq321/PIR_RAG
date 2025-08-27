#!/usr/bin/env python3
"""
PIR Experiment Examples

This script shows different ways to run PIR experiments with various data configurations.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"🏃 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print("✅ Success!")
        if result.stdout:
            # Show first few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[:10]:  # First 10 lines
                print(f"  {line}")
            if len(lines) > 10:
                print(f"  ... ({len(lines) - 10} more lines)")
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")

def main():
    """Run example experiments."""
    
    # Change to PIR_RAG directory
    pir_rag_dir = Path(__file__).parent
    os.chdir(pir_rag_dir)
    
    print("🧪 PIR Experiment Examples")
    print("Testing different data loading and experiment configurations")
    
    # Example 1: Test data loading
    run_command(
        "python test_data_loading.py",
        "Testing Data Loading Functionality"
    )
    
    # Example 2: Quick synthetic data experiment
    run_command(
        "python experiments/comprehensive_experiment.py --experiment single --n-docs 100 --n-queries 3",
        "Quick Single Experiment (Synthetic Data)"
    )
    
    # Example 3: Retrieval performance test
    run_command(
        "python test_retrieval_performance.py",
        "Retrieval Performance Test"
    )
    
    # Example 4: Small scalability test
    run_command(
        "python experiments/comprehensive_experiment.py --experiment scalability --n-queries 2",
        "Mini Scalability Test (Synthetic Data)"
    )
    
    # Example 5: Analysis (if results exist)
    run_command(
        "python experiments/analyze_results.py --generate-summary",
        "Generate Analysis Summary (if results exist)"
    )
    
    print(f"\n{'='*60}")
    print("🎉 Example Tests Complete!")
    print(f"{'='*60}")
    
    print("\n📚 What you can do next:")
    print("\n🔬 Run Full Experiments:")
    print("   python experiments/comprehensive_experiment.py --experiment all")
    print("   python experiments/comprehensive_experiment.py --experiment all --n-docs 2000")
    
    print("\n📊 Use Real Data (if available):")
    print("   python experiments/comprehensive_experiment.py --experiment all")
    print("   # Will auto-detect data/ folder contents")
    
    print("\n🎯 Custom Data:")
    print("   python experiments/comprehensive_experiment.py \\")
    print("       --embeddings-path 'your_embeddings.npy' \\")
    print("       --corpus-path 'your_corpus.csv'")
    
    print("\n📈 Analysis & Plotting:")
    print("   python experiments/analyze_results.py --generate-all")
    print("   python experiments/analyze_results.py --generate-timing")
    
    print("\n📖 Help & Documentation:")
    print("   python experiments/comprehensive_experiment.py --help")
    print("   cat DATA_CONFIGURATION.md")
    print("   cat EXPERIMENT_README.md")

if __name__ == "__main__":
    import os
    main()
