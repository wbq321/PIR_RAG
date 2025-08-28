#!/usr/bin/env python3
"""
Retrieval Performance Analysis Script

This script analyzes retrieval quality results from test_retrieval_performance.py
and generates comprehensive plots and reports for IR metrics.
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

class RetrievalAnalyzer:
    """Analyzer for retrieval performance results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
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
            return results
        
        # Directory case (original behavior)
        result_files = list(self.results_dir.glob(pattern))
        
        if not result_files:
            print(f"No retrieval results found in {self.results_dir}")
            return results
            
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = file_path.name
                    results.append(data)
                    print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return results
    
    def plot_retrieval_quality_comparison(self, results: List[Dict], save_path: str = None):
        """Plot retrieval quality metrics comparison across systems."""
        if not results:
            print("No results to plot")
            return
            
        # Extract metrics for each system
        systems_data = {}
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                # Skip if system_data is None or not a dict
                if system_data is None or not isinstance(system_data, dict):
                    print(f"Warning: Skipping {system_name} - invalid data (None or not dict)")
                    continue
                    
                if system_name not in systems_data:
                    display_name = self.get_display_name(system_name)
                    systems_data[display_name] = {
                        'precision': [],
                        'recall': [],
                        'ndcg': [],
                        'query_time': [],
                        'qps': []
                    }
                
                # Check if system_data is still valid before checking keys
                if system_data is not None and isinstance(system_data, dict) and 'avg_precision_at_k' in system_data:
                    display_name = self.get_display_name(system_name)
                    systems_data[display_name]['precision'].append(system_data['avg_precision_at_k'])
                    systems_data[display_name]['recall'].append(system_data['avg_recall_at_k'])
                    systems_data[display_name]['ndcg'].append(system_data['avg_ndcg_at_k'])
                    # Use the correct field name for query time
                    query_time = system_data.get('avg_total_query_time', system_data.get('avg_query_time', 0))
                    systems_data[display_name]['query_time'].append(query_time)
                    # Calculate QPS from query time if not available
                    qps = system_data.get('queries_per_second', 1.0 / query_time if query_time > 0 else 0)
                    systems_data[display_name]['qps'].append(qps)
        
        if not systems_data:
            print("No valid system data found")
            return
            
        # Generate individual plots for each metric
        self._plot_individual_metric(systems_data, 'precision', 'Precision@K Performance Comparison', save_path, '_precision')
        self._plot_individual_metric(systems_data, 'recall', 'Recall@K Performance Comparison', save_path, '_recall') 
        self._plot_individual_metric(systems_data, 'ndcg', 'NDCG@K Performance Comparison', save_path, '_ndcg')
        self._plot_individual_metric(systems_data, 'query_time', 'Query Time Comparison', save_path, '_query_time')
        self._plot_individual_metric(systems_data, 'qps', 'Queries Per Second Comparison', save_path, '_qps')
    
    def _plot_individual_metric(self, systems_data, metric_key, title, save_path, suffix):
        """Generate individual plot for a specific metric."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        systems = list(systems_data.keys())
        values = [np.mean(systems_data[sys][metric_key]) if systems_data[sys][metric_key] else 0 
                 for sys in systems]
        errors = [np.std(systems_data[sys][metric_key]) if len(systems_data[sys][metric_key]) > 1 else 0 
                 for sys in systems]
        
        bars = ax.bar(systems, values, yerr=errors, capsize=5, alpha=0.7,
                     color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel(metric_key.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        ax.set_xlabel('System', fontweight='bold', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.1 if values else 1)
        
        plt.tight_layout()
        
        if save_path:
            # Create individual file name
            base_path = save_path.replace('.png', '').replace('.pdf', '')
            individual_path = f"{base_path}{suffix}.png"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            individual_path_pdf = f"{base_path}{suffix}.pdf"  
            plt.savefig(individual_path_pdf, bbox_inches='tight')
            print(f"{metric_key.title()} plot saved to: {individual_path}")
        
        plt.close()
    
    def plot_precision_recall_curves(self, results: List[Dict], save_path: str = None):
        """Plot precision and recall curves as separate figures."""
        if not results:
            return
        
        # Generate separate plots for precision, recall, and precision-recall relationship
        self._plot_precision_curves(results, save_path)
        self._plot_recall_curves(results, save_path)
        self._plot_precision_recall_scatter(results, save_path)
    
    def _plot_precision_curves(self, results: List[Dict], save_path: str = None):
        """Plot precision curves across queries."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                if system_data is None or not isinstance(system_data, dict):
                    continue
                    
                if 'quality_metrics' in system_data:
                    precisions = [q['precision_at_k'] for q in system_data['quality_metrics']]
                    query_ids = list(range(1, len(precisions) + 1))
                    
                    display_name = self.get_display_name(system_name)
                    ax.plot(query_ids, precisions, 'o-', label=display_name, alpha=0.8, linewidth=2, markersize=4)
        
        ax.set_xlabel('Query Number', fontweight='bold', fontsize=12)
        ax.set_ylabel('Precision@K', fontweight='bold', fontsize=12)
        ax.set_title('Precision@K Performance Across Queries', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            base_path = save_path.replace('.png', '').replace('.pdf', '')
            precision_path = f"{base_path}_precision_curves.png"
            plt.savefig(precision_path, dpi=300, bbox_inches='tight')
            plt.savefig(f"{base_path}_precision_curves.pdf", bbox_inches='tight')
            print(f"Precision curves saved to: {precision_path}")
        
        plt.close()
    
    def _plot_recall_curves(self, results: List[Dict], save_path: str = None):
        """Plot recall curves across queries."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                if system_data is None or not isinstance(system_data, dict):
                    continue
                    
                if 'quality_metrics' in system_data:
                    recalls = [q['recall_at_k'] for q in system_data['quality_metrics']]
                    query_ids = list(range(1, len(recalls) + 1))
                    
                    display_name = self.get_display_name(system_name)
                    ax.plot(query_ids, recalls, 'o-', label=display_name, alpha=0.8, linewidth=2, markersize=4)
        
        ax.set_xlabel('Query Number', fontweight='bold', fontsize=12)
        ax.set_ylabel('Recall@K', fontweight='bold', fontsize=12)
        ax.set_title('Recall@K Performance Across Queries', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            base_path = save_path.replace('.png', '').replace('.pdf', '')
            recall_path = f"{base_path}_recall_curves.png"
            plt.savefig(recall_path, dpi=300, bbox_inches='tight')
            plt.savefig(f"{base_path}_recall_curves.pdf", bbox_inches='tight')
            print(f"Recall curves saved to: {recall_path}")
        
        plt.close()
    
    def _plot_precision_recall_scatter(self, results: List[Dict], save_path: str = None):
        """Plot precision vs recall scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                if system_data is None or not isinstance(system_data, dict):
                    continue
                    
                if 'quality_metrics' in system_data:
                    precisions = [q['precision_at_k'] for q in system_data['quality_metrics']]
                    recalls = [q['recall_at_k'] for q in system_data['quality_metrics']]
                    
                    display_name = self.get_display_name(system_name)
                    
                    # Plot scatter points
                    ax.scatter(recalls, precisions, label=display_name, alpha=0.6, s=50)
                    
                    # Plot average point
                    avg_precision = np.mean(precisions)
                    avg_recall = np.mean(recalls)
                    ax.scatter(avg_recall, avg_precision, 
                             label=f'{display_name} (avg)', 
                             s=200, marker='*', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('Recall@K', fontweight='bold', fontsize=12)
        ax.set_ylabel('Precision@K', fontweight='bold', fontsize=12)
        ax.set_title('Precision vs Recall Analysis', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            base_path = save_path.replace('.png', '').replace('.pdf', '')
            scatter_path = f"{base_path}_precision_recall_scatter.png"
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.savefig(f"{base_path}_precision_recall_scatter.pdf", bbox_inches='tight')
            print(f"Precision-recall scatter saved to: {scatter_path}")
        
        plt.close()
    
    def plot_query_distribution(self, results: List[Dict], save_path: str = None):
        """Plot distribution of all retrieval quality metrics as separate figures."""
        if not results:
            return
        
        # Generate separate distribution plots for each metric
        self._plot_metric_distribution(results, 'precision_at_k', 'Precision@K Distribution', save_path, '_precision_dist')
        self._plot_metric_distribution(results, 'recall_at_k', 'Recall@K Distribution', save_path, '_recall_dist')
        self._plot_metric_distribution(results, 'ndcg_at_k', 'NDCG@K Distribution', save_path, '_ndcg_dist')
    
    def _plot_metric_distribution(self, results: List[Dict], metric_key: str, title: str, save_path: str, suffix: str):
        """Plot distribution for a specific metric."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        color_idx = 0
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                if system_data is None or not isinstance(system_data, dict):
                    continue
                    
                if 'quality_metrics' in system_data:
                    display_name = self.get_display_name(system_name)
                    values = [q[metric_key] for q in system_data['quality_metrics']]
                    
                    ax.hist(values, bins=15, alpha=0.6, label=display_name, 
                           density=True, color=colors[color_idx % len(colors)])
                    color_idx += 1
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel(metric_key.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        ax.set_ylabel('Density', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.0)
        
        plt.tight_layout()
        
        if save_path:
            base_path = save_path.replace('.png', '').replace('.pdf', '')
            dist_path = f"{base_path}{suffix}.png"
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.savefig(f"{base_path}{suffix}.pdf", bbox_inches='tight')
            print(f"{metric_key.replace('_', ' ').title()} distribution saved to: {dist_path}")
        
        plt.close()
    
    def generate_retrieval_summary_report(self, results: List[Dict], save_path: str = None) -> str:
        """Generate a comprehensive text summary of retrieval performance."""
        if not results:
            return "No results to analyze."
            
        report = []
        report.append("="*80)
        report.append("RETRIEVAL PERFORMANCE ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total result files analyzed: {len(results)}")
        report.append("")
        
        # Aggregate results by system
        system_stats = {}
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                # Skip if system_data is None or not a dict
                if system_data is None or not isinstance(system_data, dict):
                    print(f"Warning: Skipping {system_name} - invalid data")
                    continue
                    
                if system_name not in system_stats:
                    display_name = self.get_display_name(system_name)
                    system_stats[display_name] = {
                        'precision_scores': [],
                        'recall_scores': [],
                        'ndcg_scores': [],
                        'query_times': [],
                        'qps_scores': [],
                        'total_queries': 0
                    }
                
                if 'avg_precision_at_k' in system_data:
                    display_name = self.get_display_name(system_name)
                    system_stats[display_name]['precision_scores'].append(system_data['avg_precision_at_k'])
                    system_stats[display_name]['recall_scores'].append(system_data['avg_recall_at_k'])
                    system_stats[display_name]['ndcg_scores'].append(system_data['avg_ndcg_at_k'])
                    # Use the correct field name for query time
                    query_time = system_data.get('avg_total_query_time', system_data.get('avg_query_time', 0))
                    system_stats[display_name]['query_times'].append(query_time)
                    # Calculate QPS from query time if not available
                    qps = system_data.get('queries_per_second', 1.0 / query_time if query_time > 0 else 0)
                    system_stats[display_name]['qps_scores'].append(qps)
                    system_stats[display_name]['total_queries'] += len(system_data.get('quality_metrics', []))
        
        # Generate system comparison
        report.append("SYSTEM PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        for system_name, stats in system_stats.items():
            if not stats['precision_scores']:
                continue
                
            report.append(f"\n📊 {system_name.upper()}")
            report.append(f"   Precision@K:  {np.mean(stats['precision_scores']):.4f} ± {np.std(stats['precision_scores']):.4f}")
            report.append(f"   Recall@K:     {np.mean(stats['recall_scores']):.4f} ± {np.std(stats['recall_scores']):.4f}")
            report.append(f"   NDCG@K:       {np.mean(stats['ndcg_scores']):.4f} ± {np.std(stats['ndcg_scores']):.4f}")
            report.append(f"   Query Time:   {np.mean(stats['query_times']):.4f}s ± {np.std(stats['query_times']):.4f}s")
            report.append(f"   Throughput:   {np.mean(stats['qps_scores']):.2f} queries/sec")
            report.append(f"   Total Queries: {stats['total_queries']}")
        
        # Best performing system analysis
        if system_stats:
            report.append("\n" + "="*50)
            report.append("BEST PERFORMING SYSTEMS")
            report.append("="*50)
            
            metrics = {
                'Precision@K': ('precision_scores', 'higher'),
                'Recall@K': ('recall_scores', 'higher'),
                'NDCG@K': ('ndcg_scores', 'higher'),
                'Query Speed': ('query_times', 'lower'),
                'Throughput': ('qps_scores', 'higher')
            }
            
            for metric_name, (metric_key, direction) in metrics.items():
                best_system = None
                best_value = None
                
                for system_name, stats in system_stats.items():
                    if not stats[metric_key]:
                        continue
                        
                    avg_value = np.mean(stats[metric_key])
                    
                    if best_system is None:
                        best_system = system_name
                        best_value = avg_value
                    elif direction == 'higher' and avg_value > best_value:
                        best_system = system_name
                        best_value = avg_value
                    elif direction == 'lower' and avg_value < best_value:
                        best_system = system_name
                        best_value = avg_value
                
                if best_system:
                    unit = "s" if "Time" in metric_name else ("queries/sec" if "Throughput" in metric_name else "")
                    report.append(f"🏆 Best {metric_name}: {best_system} ({best_value:.4f}{unit})")
        
        report.append("\n" + "="*80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Retrieval summary report saved to: {save_path}")
        
        return report_text
    
    def generate_all_retrieval_plots(self, output_dir: str = None, pattern: str = "*retrieval_performance*.json"):
        """Generate all retrieval performance plots and reports."""
        if output_dir is None:
            output_dir = self.results_dir / "retrieval_analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load results
        results = self.load_retrieval_results(pattern)
        
        if not results:
            print("❌ No retrieval results found to analyze")
            return
        
        print(f"📊 Generating retrieval performance analysis...")
        
        # Generate plots
        self.plot_retrieval_quality_comparison(
            results, 
            output_dir / f"retrieval_quality_comparison_{timestamp}.png"
        )
        
        self.plot_precision_recall_curves(
            results,
            output_dir / f"precision_recall_curves_{timestamp}.png"
        )
        
        self.plot_query_distribution(
            results,
            output_dir / f"query_distribution_{timestamp}.png"
        )
        
        # Generate report
        report = self.generate_retrieval_summary_report(
            results,
            output_dir / f"retrieval_summary_report_{timestamp}.txt"
        )
        
        print(f"\n✅ Retrieval analysis complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📈 Plots: retrieval_quality_comparison, precision_recall_curves, query_distribution")
        print(f"📄 Report: retrieval_summary_report_{timestamp}.txt")
        
        return report


def main():
    """Main analysis runner."""
    parser = argparse.ArgumentParser(description="Analyze retrieval performance results")
    parser.add_argument("--results-dir", default="results", help="Directory containing result files OR path to single JSON file")
    parser.add_argument("--input-file", help="Path to a specific JSON result file (alternative to --results-dir)")
    parser.add_argument("--data-pattern", default="*retrieval_performance*.json", help="Pattern to match result files (only used with --results-dir)")
    parser.add_argument("--output-dir", default=None, help="Output directory for analysis")
    parser.add_argument("--generate-all", action="store_true", help="Generate all plots and reports")
    parser.add_argument("--generate-comparison", action="store_true", help="Generate quality comparison plot")
    parser.add_argument("--generate-curves", action="store_true", help="Generate precision-recall curves")
    parser.add_argument("--generate-distribution", action="store_true", help="Generate query distribution plots")
    parser.add_argument("--generate-report", action="store_true", help="Generate summary report")
    
    args = parser.parse_args()
    
    # Handle input file vs directory
    if args.input_file:
        analyzer = RetrievalAnalyzer(args.input_file)
    else:
        analyzer = RetrievalAnalyzer(args.results_dir)
    
    if args.generate_all:
        analyzer.generate_all_retrieval_plots(args.output_dir, args.data_pattern)
    else:
        results = analyzer.load_retrieval_results(args.data_pattern)
        
        if args.generate_comparison:
            analyzer.plot_retrieval_quality_comparison(results)
        
        if args.generate_curves:
            analyzer.plot_retrieval_curves(results)
            
        if args.generate_distribution:
            analyzer.plot_query_distribution(results)
            
        if args.generate_report:
            report = analyzer.generate_retrieval_summary_report(results)
            print(report)
        
        if args.generate_curves:
            analyzer.plot_precision_recall_curves(results)
            
        if args.generate_distribution:
            analyzer.plot_query_distribution(results)
            
        if args.generate_report:
            report = analyzer.generate_retrieval_summary_report(results)
            print(report)


class RecallOnlyAnalyzer(RetrievalAnalyzer):
    """Analyzer that focuses only on recall metrics and generates individual plots."""
    
    def plot_recall_only_comparison(self, results: List[Dict], save_path: str = None):
        """Plot only recall metrics comparison across systems."""
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
                    print(f"Warning: Skipping {system_name} - invalid data (None or not dict)")
                    continue
                    
                if system_name not in systems_data:
                    display_name = self.get_display_name(system_name)
                    systems_data[display_name] = {
                        'recall': [],
                        'query_time': []
                    }
                
                # Check if system_data is still valid before checking keys
                if system_data is not None and isinstance(system_data, dict) and 'avg_recall_at_k' in system_data:
                    display_name = self.get_display_name(system_name)
                    systems_data[display_name]['recall'].append(system_data['avg_recall_at_k'])
                    # Use the correct field name for query time
                    query_time = system_data.get('avg_total_query_time', system_data.get('avg_query_time', 0))
                    systems_data[display_name]['query_time'].append(query_time)
        
        if not systems_data:
            print("No valid system data found")
            return
        
        # Create individual recall plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        systems = list(systems_data.keys())
        recall_values = [np.mean(systems_data[sys]['recall']) if systems_data[sys]['recall'] else 0 
                        for sys in systems]
        recall_errors = [np.std(systems_data[sys]['recall']) if len(systems_data[sys]['recall']) > 1 else 0 
                        for sys in systems]
        
        bars = ax.bar(systems, recall_values, yerr=recall_errors, capsize=5, alpha=0.7, color='skyblue')
        ax.set_title('Recall@K Comparison', fontweight='bold', fontsize=14)
        ax.set_ylabel('Recall@K', fontweight='bold', fontsize=12)
        ax.set_xlabel('Systems', fontweight='bold', fontsize=12)
        
        # Add value labels on bars
        for bar, value in zip(bars, recall_values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Recall-only plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_recall_vs_query_time(self, results: List[Dict], save_path: str = None):
        """Plot recall vs query time scatter plot."""
        if not results:
            print("No results to plot")
            return
            
        # Extract data for scatter plot
        systems_data = {}
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                if system_data is None or not isinstance(system_data, dict):
                    continue
                    
                if 'avg_recall_at_k' in system_data:
                    display_name = self.get_display_name(system_name)
                    if display_name not in systems_data:
                        systems_data[display_name] = {'recall': [], 'query_time': []}
                    
                    systems_data[display_name]['recall'].append(system_data['avg_recall_at_k'])
                    query_time = system_data.get('avg_total_query_time', system_data.get('avg_query_time', 0))
                    systems_data[display_name]['query_time'].append(query_time)
        
        if not systems_data:
            print("No valid system data found")
            return
        
        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(systems_data)))
        
        for i, (system, data) in enumerate(systems_data.items()):
            avg_recall = np.mean(data['recall']) if data['recall'] else 0
            avg_query_time = np.mean(data['query_time']) if data['query_time'] else 0
            
            ax.scatter(avg_query_time, avg_recall, s=100, alpha=0.7, 
                      color=colors[i], label=system, edgecolors='black')
            
            # Add system name labels
            ax.annotate(system, (avg_query_time, avg_recall), 
                       xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax.set_xlabel('Average Query Time (seconds)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Recall@K', fontweight='bold', fontsize=12)
        ax.set_title('Recall vs Query Time Trade-off', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Recall vs Query Time plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_recall_distribution(self, results: List[Dict], save_path: str = None):
        """Plot recall distribution across queries for each system."""
        if not results:
            print("No results to plot")
            return
        
        # Extract per-query recall data
        systems_data = {}
        
        for result in results:
            for system_name, system_data in result.items():
                if system_name in ['source_file', 'experiment_info']:
                    continue
                
                if system_data is None or not isinstance(system_data, dict):
                    continue
                    
                if 'per_query_metrics' in system_data:
                    display_name = self.get_display_name(system_name)
                    recalls = []
                    
                    for query_metrics in system_data['per_query_metrics']:
                        if 'recall_at_k' in query_metrics:
                            recalls.append(query_metrics['recall_at_k'])
                    
                    if recalls:
                        systems_data[display_name] = recalls
        
        if not systems_data:
            print("No per-query recall data found")
            return
        
        # Create box plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        systems = list(systems_data.keys())
        recall_data = [systems_data[sys] for sys in systems]
        
        box_plot = ax.boxplot(recall_data, labels=systems, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(systems)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Systems', fontweight='bold', fontsize=12)
        ax.set_ylabel('Recall@K', fontweight='bold', fontsize=12)
        ax.set_title('Recall Distribution Across Queries', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Recall distribution plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_individual_recall_plots(self, figures_dir: Path):
        """Generate all individual recall-focused plots."""
        results = self.load_retrieval_results()
        if not results:
            print("No retrieval results found")
            return
        
        print("Generating individual recall-focused plots...")
        
        # Create individual plots
        self.plot_recall_only_comparison(results, figures_dir / "recall_only_comparison.png")
        self.plot_recall_vs_query_time(results, figures_dir / "recall_vs_query_time.png")
        self.plot_recall_distribution(results, figures_dir / "recall_distribution.png")
        
        print("✅ Individual recall plots generated successfully!")


if __name__ == "__main__":
    main()
