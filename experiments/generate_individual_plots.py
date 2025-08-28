#!/usr/bin/env python3
"""
Script to generate individual plots instead of combined subplot figures.
This provides an alternative to the combined analysis in analyze_results.py.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import analyze_results
sys.path.append(str(Path(__file__).parent))

from analyze_results import IndividualPlotAnalyzer, RETRIEVAL_ANALYSIS_AVAILABLE

if RETRIEVAL_ANALYSIS_AVAILABLE:
    from analyze_retrieval_performance import RetrievalAnalyzer

# Import the new recall-only analyzer
try:
    from analyze_recall_only import RecallOnlyAnalyzer
    RECALL_ANALYSIS_AVAILABLE = True
except ImportError:
    RECALL_ANALYSIS_AVAILABLE = False


def main():
    """Generate individual plots from PIR experiment results."""
    parser = argparse.ArgumentParser(description="Generate Individual PIR Plots")
    parser.add_argument("--results-dir", default="results", help="Directory containing results")
    parser.add_argument("--single-file", help="JSON file for single experiment analysis")
    parser.add_argument("--scalability-file", help="JSON file for scalability analysis")
    parser.add_argument("--generate-all", action="store_true", help="Generate all individual plots from latest files")
    parser.add_argument("--recall-only", action="store_true", help="Generate only recall-focused plots for retrieval analysis")
    
    args = parser.parse_args()
    
    analyzer = IndividualPlotAnalyzer(args.results_dir)
    
    if args.generate_all:
        # Find latest files and generate individual plots
        results_path = Path(args.results_dir)
        single_files = list(results_path.glob("single_experiment_*.json"))
        scalability_files = list(results_path.glob("scalability_*.json"))
        
        if single_files:
            latest_single = max(single_files, key=lambda x: x.stat().st_mtime)
            print(f"Generating individual plots for single experiment: {latest_single}")
            results = analyzer.load_json_results(latest_single.name)
            analyzer.plot_individual_timing("single_experiment_individual")
            print("✅ Individual timing plots generated")
        
        if scalability_files:
            # Try files in reverse chronological order until we find a valid one
            scalability_files_sorted = sorted(scalability_files, key=lambda x: x.stat().st_mtime, reverse=True)
            
            for scalability_file in scalability_files_sorted:
                try:
                    print(f"Generating individual plots for scalability: {scalability_file}")
                    results = analyzer.load_json_results(scalability_file.name)
                    analyzer.plot_individual_scalability(results, "scalability_individual")
                    print(f"✅ Individual scalability plots generated from: {scalability_file}")
                    break
                except Exception as e:
                    print(f"❌ Failed to analyze {scalability_file}: {e}")
                    continue
            else:
                print("❌ No valid scalability files found")
    
    else:
        if args.single_file:
            print(f"Generating individual plots for single experiment: {args.single_file}")
            results = analyzer.load_json_results(args.single_file)
            analyzer.plot_individual_timing("single_experiment_individual")
            print("✅ Individual timing plots generated")
        
        if args.scalability_file:
            try:
                print(f"Generating individual plots for scalability: {args.scalability_file}")
                results = analyzer.load_json_results(args.scalability_file)
                analyzer.plot_individual_scalability(results, "scalability_individual")
                print(f"✅ Individual scalability plots generated")
            except Exception as e:
                print(f"❌ Failed to analyze {args.scalability_file}: {e}")
    
    # Generate individual retrieval performance plots if available
    if args.generate_all and args.recall_only and RECALL_ANALYSIS_AVAILABLE:
        print("Generating individual recall-only plots...")
        recall_analyzer = RecallOnlyAnalyzer(args.results_dir)
        recall_analyzer.generate_all_recall_plots("individual_recall_analysis")
    elif args.generate_all and RETRIEVAL_ANALYSIS_AVAILABLE:
        print("Generating individual retrieval performance plots...")
        retrieval_analyzer = RetrievalAnalyzer(args.results_dir)
        retrieval_analyzer.generate_all_retrieval_plots(analyzer.figures_dir)
    
    print("Individual plot generation complete! Check the 'figures' directory for separate plot files.")
    print("Generated files will have names like:")
    print("  - single_experiment_individual_query_times.png")
    print("  - single_experiment_individual_communication_costs.png")
    print("  - scalability_individual_query_times.png")
    if args.recall_only:
        print("  - individual_recall_analysis_comparison.png")
        print("  - individual_recall_analysis_by_k.png")
        print("  - individual_recall_analysis_distribution.png")
    print("  - etc.")


if __name__ == "__main__":
    main()
