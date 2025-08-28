#!/usr/bin/env python3
"""
Recall-Only Retrieval Performance Analysis Script

This script focuses specifically on recall metrics from retrieval performance results,
providing individual plots for recall analysis only.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
import argparse
from datetime import datetime

class RecallOnlyAnalyzer:
    """Analyzer focused exclusively on recall metrics."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path("figures")
        self.figures_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # System name mapping for display
        self.system_display_names = {
            'pir_rag': 'PIR-RAG',
            'graph_pir': 'Graph-PIR', 
            'tiptoe': 'Tiptoe'
        }
    
    def get_display_name(self, system_name: str) -> str:
        """Get the display name for a system."""
        return self.system_display_names.get(system_name, system_name)
        
    def load_retrieval_results(self, pattern: str = "*retrieval_performance*.json") -> List[Dict[str, Any]]:
        """Load retrieval performance results from JSON files."""
        results = []
        
        # Check if results_dir is actually a specific file
        if self.results_dir.is_file() and self.results_dir.suffix == '.json':
            # Single file case
            try:
                with open(self.results_dir, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = self.results_dir.name
                    results.append(data)
                    print(f"Loaded: {self.results_dir.name}")
            except Exception as e:
                print(f"Error loading {self.results_dir}: {e}")
        else:
            # Directory case - find matching files
            for file_path in sorted(self.results_dir.glob(pattern)):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        data['source_file'] = file_path.name
                        results.append(data)
                        print(f"Loaded: {file_path.name}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return results
    
    def plot_recall_comparison(self, results: List[Dict], save_name: str = "recall_comparison"):
        """Plot recall metrics comparison across systems - individual figure."""
        if not results:
            print("No results to plot")
            return
            
        # Extract recall metrics for each system
        systems_data = {}
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                # Skip if system_data is None or not a dict
                if system_data is None or not isinstance(system_data, dict):
                    print(f"Warning: Skipping {system_name} - invalid data")
                    continue
                    
                if system_name not in systems_data:
                    display_name = self.get_display_name(system_name)
                    systems_data[display_name] = {'recall': []}
                
                # Extract recall data
                if 'avg_recall_at_k' in system_data:
                    display_name = self.get_display_name(system_name)
                    systems_data[display_name]['recall'].append(system_data['avg_recall_at_k'])
        
        if not systems_data:
            print("No valid system data found")
            return
            
        # Create individual recall comparison plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        systems = list(systems_data.keys())
        recall_values = [np.mean(systems_data[sys]['recall']) if systems_data[sys]['recall'] else 0 
                        for sys in systems]
        recall_errors = [np.std(systems_data[sys]['recall']) if len(systems_data[sys]['recall']) > 1 else 0 
                        for sys in systems]
        
        # Create bar plot
        bars = ax.bar(systems, recall_values, yerr=recall_errors, capsize=5, alpha=0.7, 
                     color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        
        ax.set_title('Recall@K Comparison Across Systems', fontweight='bold', fontsize=14)
        ax.set_ylabel('Recall@K', fontweight='bold', fontsize=12)
        ax.set_xlabel('System', fontweight='bold', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, recall_values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Set y-axis to start from 0 and add some padding
        ax.set_ylim(0, max(recall_values) * 1.1 if recall_values else 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"Recall comparison plot saved as: {save_name}.png and {save_name}.pdf")
    
    def plot_recall_by_k_values(self, results: List[Dict], save_name: str = "recall_by_k"):
        """Plot recall for different k values if available - individual figure."""
        if not results:
            print("No results to plot")
            return
        
        # Check if we have recall data for different k values
        systems_recall_k = {}
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                if system_data is None or not isinstance(system_data, dict):
                    continue
                
                display_name = self.get_display_name(system_name)
                
                # Look for recall at different k values
                if 'recall_at_k' in system_data and isinstance(system_data['recall_at_k'], dict):
                    if display_name not in systems_recall_k:
                        systems_recall_k[display_name] = {}
                    
                    for k, recall_val in system_data['recall_at_k'].items():
                        if k not in systems_recall_k[display_name]:
                            systems_recall_k[display_name][k] = []
                        systems_recall_k[display_name][k].append(recall_val)
        
        if not systems_recall_k:
            print("No recall@k data found for different k values")
            return
        
        # Create individual plot for recall@k curves
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (system, recall_data) in enumerate(systems_recall_k.items()):
            k_values = sorted([int(k) for k in recall_data.keys()])
            recall_means = [np.mean(recall_data[str(k)]) for k in k_values]
            recall_stds = [np.std(recall_data[str(k)]) if len(recall_data[str(k)]) > 1 else 0 
                          for k in k_values]
            
            color = colors[i % len(colors)]
            ax.plot(k_values, recall_means, 'o-', label=system, color=color, linewidth=2, markersize=6)
            ax.fill_between(k_values, 
                           [m - s for m, s in zip(recall_means, recall_stds)],
                           [m + s for m, s in zip(recall_means, recall_stds)],
                           alpha=0.2, color=color)
        
        ax.set_title('Recall@K Performance Curves', fontweight='bold', fontsize=14)
        ax.set_xlabel('K (Number of Retrieved Documents)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Recall@K', fontweight='bold', fontsize=12)
        ax.legend(fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"Recall@K curves plot saved as: {save_name}.png and {save_name}.pdf")
    
    def plot_recall_distribution(self, results: List[Dict], save_name: str = "recall_distribution"):
        """Plot recall value distribution across queries - individual figure."""
        if not results:
            print("No results to plot")
            return
        
        # Extract per-query recall values
        systems_recall_dist = {}
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                if system_data is None or not isinstance(system_data, dict):
                    continue
                
                display_name = self.get_display_name(system_name)
                
                # Look for per-query recall data
                if 'per_query_recall' in system_data:
                    if display_name not in systems_recall_dist:
                        systems_recall_dist[display_name] = []
                    systems_recall_dist[display_name].extend(system_data['per_query_recall'])
                elif 'recall_values' in system_data:
                    if display_name not in systems_recall_dist:
                        systems_recall_dist[display_name] = []
                    systems_recall_dist[display_name].extend(system_data['recall_values'])
        
        if not systems_recall_dist:
            print("No per-query recall distribution data found")
            return
        
        # Create individual distribution plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Create box plot for recall distributions
        systems = list(systems_recall_dist.keys())
        recall_data = [systems_recall_dist[sys] for sys in systems]
        
        box_plot = ax.boxplot(recall_data, labels=systems, patch_artist=True, 
                             boxprops=dict(alpha=0.7), medianprops=dict(linewidth=2))
        
        # Color the boxes
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Recall Distribution Across Queries', fontweight='bold', fontsize=14)
        ax.set_ylabel('Recall@K Values', fontweight='bold', fontsize=12)
        ax.set_xlabel('System', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"Recall distribution plot saved as: {save_name}.png and {save_name}.pdf")
    
    def generate_all_recall_plots(self, save_prefix: str = "recall_analysis"):
        """Generate all recall-focused plots."""
        print("Loading retrieval performance results...")
        results = self.load_retrieval_results()
        
        if not results:
            print("No retrieval results found!")
            return
        
        print(f"Generating recall-only analysis plots...")
        
        # Generate individual recall plots
        self.plot_recall_comparison(results, f"{save_prefix}_comparison")
        self.plot_recall_by_k_values(results, f"{save_prefix}_by_k")
        self.plot_recall_distribution(results, f"{save_prefix}_distribution")
        
        print("All recall analysis plots generated!")
        print(f"Files saved in: {self.figures_dir}")
        print("Generated plots:")
        print(f"  - {save_prefix}_comparison.png/pdf")
        print(f"  - {save_prefix}_by_k.png/pdf")
        print(f"  - {save_prefix}_distribution.png/pdf")


def main():
    """Main function to run recall-only analysis."""
    parser = argparse.ArgumentParser(description="Analyze Recall-Only Performance")
    parser.add_argument("--results-dir", default="results", help="Directory containing results")
    parser.add_argument("--results-file", help="Specific JSON file to analyze")
    parser.add_argument("--save-prefix", default="recall_analysis", help="Prefix for saved plot files")
    
    args = parser.parse_args()
    
    # Use specific file if provided, otherwise use directory
    results_path = args.results_file if args.results_file else args.results_dir
    
    analyzer = RecallOnlyAnalyzer(results_path)
    analyzer.generate_all_recall_plots(args.save_prefix)


if __name__ == "__main__":
    main()
