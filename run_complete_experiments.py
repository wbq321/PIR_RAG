#!/usr/bin/env python3
"""
Complete PIR Experiment Suite

This script demonstrates the full experimental capabilities including:
- Performance benchmarking (timing, scalability)
- Retrieval quality evaluation (precision, recall, NDCG)
- Comprehensive analysis and plotting
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_command(cmd, description, capture_output=False):
    """Run a command and display results."""
    print(f"\n{'='*60}")
    print(f"🏃 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            return result.stdout
        else:
            result = subprocess.run(cmd, shell=True, check=True)
            print("✅ Success!")
            return None
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print(f"Error: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"Output: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error: {e.stderr}")
        return None

def main():
    """Run the complete PIR experiment suite."""
    parser = argparse.ArgumentParser(description="Complete PIR Experiment Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests with small datasets")
    parser.add_argument("--full", action="store_true", help="Run full comprehensive experiments")
    parser.add_argument("--analysis-only", action="store_true", help="Only run analysis on existing results")
    parser.add_argument("--n-docs", type=int, default=1000, help="Number of documents")
    parser.add_argument("--n-queries", type=int, default=20, help="Number of queries")
    
    args = parser.parse_args()
    
    # Change to PIR_RAG directory
    pir_rag_dir = Path(__file__).parent
    os.chdir(pir_rag_dir)
    
    print("🧪 Complete PIR Experiment Suite")
    print("=" * 60)
    
    if args.analysis_only:
        print("📊 Running analysis on existing results...")
        
        # Run comprehensive analysis
        run_command(
            "python experiments/analyze_results.py --generate-all",
            "Comprehensive Timing & Performance Analysis"
        )
        
        # Run retrieval analysis
        run_command(
            "python experiments/analyze_retrieval_performance.py --generate-all",
            "Retrieval Quality Analysis"
        )
        
    else:
        # Determine experiment scale
        if args.quick:
            n_docs = min(args.n_docs, 200)
            n_queries = min(args.n_queries, 5)
            scale = "Quick Test"
        elif args.full:
            n_docs = max(args.n_docs, 2000)
            n_queries = max(args.n_queries, 50)
            scale = "Full Scale"
        else:
            n_docs = args.n_docs
            n_queries = args.n_queries
            scale = "Standard"
        
        print(f"🎯 Running {scale} Experiments")
        print(f"   Documents: {n_docs}")
        print(f"   Queries: {n_queries}")
        
        # Step 1: Test data loading
        run_command(
            "python test_data_loading.py",
            "Testing Data Loading Functionality"
        )
        
        # Step 2: Run comprehensive experiments
        if args.quick:
            # Quick single experiment
            run_command(
                f"python experiments/comprehensive_experiment.py --experiment single "
                f"--n-docs {n_docs} --n-queries {n_queries}",
                "Quick Performance Experiment"
            )
        else:
            # Full experiment suite
            run_command(
                f"python experiments/comprehensive_experiment.py --experiment all "
                f"--n-docs {n_docs} --n-queries {n_queries}",
                "Comprehensive Performance Experiments (All Systems)"
            )
        
        # Step 3: Run retrieval performance tests
        run_command(
            "python test_retrieval_performance.py",
            "Retrieval Quality Evaluation"
        )
        
        # Step 4: Generate all analysis
        run_command(
            "python experiments/analyze_results.py --generate-all",
            "Performance Analysis & Plotting"
        )
        
        run_command(
            "python experiments/analyze_retrieval_performance.py --generate-all",
            "Retrieval Quality Analysis & Plotting"
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 Experiment Suite Complete!")
    print(f"{'='*60}")
    
    print("\n📊 Results Generated:")
    print("   📁 results/              - Raw experiment data (JSON)")
    print("   📈 results/figures/      - Performance plots (PNG)")
    print("   📋 results/              - Summary reports (TXT)")
    
    print("\n📈 Available Plots:")
    print("   🔄 System Comparison     - Timing across all PIR systems")
    print("   📏 Scalability Analysis  - Performance vs dataset size")
    print("   ⚙️  Parameter Sensitivity - Effect of system parameters")
    print("   🎯 Retrieval Quality     - Precision, Recall, NDCG comparison")
    print("   📊 Query Distribution    - Performance variation analysis")
    
    print("\n🔍 Key Files to Check:")
    print("   📄 results/*_summary_*.txt     - Human-readable reports")
    print("   📈 results/figures/*.png       - All plots and visualizations")
    print("   📊 results/*_summary_*.csv     - Data for further analysis")
    
    print("\n🚀 Next Steps:")
    print("   1. Review summary reports for key findings")
    print("   2. Examine plots in results/figures/ directory")
    print("   3. Use CSV files for custom analysis or publication")
    print("   4. Re-run with --full for comprehensive evaluation")
    
    if not args.analysis_only:
        print("\n💡 Tips:")
        print("   • Use --quick for rapid development/debugging")
        print("   • Use --full for publication-quality results")
        print("   • Modify --n-docs and --n-queries to suit your needs")
        print("   • Place real data in data/ folder for automatic detection")

if __name__ == "__main__":
    main()
