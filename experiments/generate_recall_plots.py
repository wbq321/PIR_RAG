#!/usr/bin/env python3
"""
Script to generate recall-only plots from retrieval performance results.
This focuses only on recall metrics instead of the full IR metrics suite.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import analyze_retrieval_performance
sys.path.append(str(Path(__file__).parent))

from analyze_retrieval_performance import RecallOnlyAnalyzer


def main():
    """Generate recall-only plots from retrieval performance results."""
    parser = argparse.ArgumentParser(description="Generate Recall-Only Plots")
    parser.add_argument("--results-dir", default="results", help="Directory containing retrieval results")
    parser.add_argument("--output-dir", default="figures", help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize recall-only analyzer
    analyzer = RecallOnlyAnalyzer(args.results_dir)
    
    print("Recall-Only Analysis Tool")
    print("=" * 40)
    
    # Generate individual recall-focused plots
    analyzer.generate_individual_recall_plots(output_dir)
    
    print("\nRecall-only analysis complete!")
    print(f"Check the '{args.output_dir}' directory for the following plots:")
    print("  - recall_only_comparison.png (Bar chart of recall@k by system)")
    print("  - recall_vs_query_time.png (Scatter plot of recall vs query time)")
    print("  - recall_distribution.png (Box plot of recall distribution)")


if __name__ == "__main__":
    main()
