#!/usr/bin/env python3
"""
Retrieval Performance Results Analyzer

Analyzes retrieval performance experiment results from the comprehensive experiment framework.
Provides detailed analysis of precision, recall, NDCG, and performance metrics.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Any

class RetrievalResultsAnalyzer:
    """Analyzer for retrieval performance experiment results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def find_retrieval_results(self) -> List[Path]:
        """Find all retrieval performance result files."""
        patterns = [
            "**/retrieval_performance_*.json",
            "**/retrieval_*.json", 
            "**/*retrieval*.json"
        ]
        
        files = []
        for pattern in patterns:
            files.extend(self.results_dir.glob(pattern))
            
        # Also check parent directories
        parent_dir = Path(".")
        for pattern in patterns:
            files.extend(parent_dir.glob(pattern))
            
        return sorted(set(files), key=lambda x: x.stat().st_mtime, reverse=True)
    
    def load_retrieval_results(self, file_path: Path) -> Dict[str, Any]:
        """Load and parse retrieval results from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ Loaded: {file_path.name}")
            return data
        except Exception as e:
            print(f"❌ Failed to load {file_path.name}: {e}")
            return None
    
    def analyze_single_result(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Analyze a single retrieval result file and create summary DataFrame."""
        
        summary_data = []
        
        # Extract experiment info
        exp_info = results.get('experiment_info', {})
        base_info = {
            'n_documents': exp_info.get('n_docs', 'N/A'),
            'n_queries': exp_info.get('n_queries', 'N/A'),
            'top_k': exp_info.get('top_k', 'N/A'),
            'data_type': exp_info.get('data_type', 'N/A'),
            'timestamp': exp_info.get('timestamp', 'N/A')
        }
        
        # Analyze each system
        systems = ['pir_rag', 'graph_pir', 'tiptoe']
        system_names = ['PIR-RAG', 'Graph-PIR', 'Tiptoe']
        
        for system_key, system_name in zip(systems, system_names):
            system_results = results.get(system_key)
            
            if system_results is None:
                # System failed
                row = base_info.copy()
                row.update({
                    'system': system_name,
                    'status': 'Failed',
                    'setup_time': 'N/A',
                    'avg_query_time': 'N/A',
                    'precision_at_k': 'N/A',
                    'recall_at_k': 'N/A',
                    'ndcg_at_k': 'N/A',
                    'communication_kb': 'N/A'
                })
            else:
                # System succeeded
                row = base_info.copy()
                
                # Check if this is hybrid approach results
                if system_results.get('hybrid_approach', False):
                    # New hybrid approach format
                    row.update({
                        'system': system_name,
                        'status': 'Success (Hybrid)',
                        'setup_time': f"{system_results.get('setup_time', 0):.3f}s",
                        'avg_query_time': f"{system_results.get('avg_pir_performance_time', 0):.3f}s",
                        'avg_simulation_time': f"{system_results.get('avg_quality_simulation_time', 0):.3f}s",
                        'precision_at_k': f"{system_results.get('avg_precision_at_k', 0):.3f}",
                        'recall_at_k': f"{system_results.get('avg_recall_at_k', 0):.3f}",
                        'ndcg_at_k': f"{system_results.get('avg_ndcg_at_k', 0):.3f}",
                        'communication_kb': f"{system_results.get('avg_communication_bytes', 0)/1024:.1f}",
                        'approach': 'Hybrid'
                    })
                else:
                    # Legacy format
                    row.update({
                        'system': system_name,
                        'status': 'Success (Legacy)',
                        'setup_time': f"{system_results.get('setup_time', 0):.3f}s",
                        'avg_query_time': f"{system_results.get('avg_query_time', 0):.3f}s",
                        'precision_at_k': f"{system_results.get('precision_at_10', 'N/A')}",
                        'recall_at_k': f"{system_results.get('recall_at_10', 'N/A')}",
                        'ndcg_at_k': f"{system_results.get('ndcg_at_10', 'N/A')}",
                        'communication_kb': f"{system_results.get('avg_download_bytes', 0)/1024:.1f}",
                        'approach': 'Legacy'
                    })
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def create_performance_comparison_plot(self, df: pd.DataFrame, save_path: Path):
        """Create performance comparison plots."""
        
        # Filter only successful results
        success_df = df[df['status'].str.contains('Success')].copy()
        
        if len(success_df) == 0:
            print("⚠️  No successful results to plot")
            return
        
        # Convert string metrics back to floats for plotting
        metrics = ['precision_at_k', 'recall_at_k', 'ndcg_at_k']
        for metric in metrics:
            success_df[metric] = pd.to_numeric(success_df[metric], errors='coerce')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Retrieval Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Retrieval Quality Metrics
        ax1 = axes[0, 0]
        x_pos = np.arange(len(success_df))
        width = 0.25
        
        ax1.bar(x_pos - width, success_df['precision_at_k'], width, label='Precision@K', alpha=0.8)
        ax1.bar(x_pos, success_df['recall_at_k'], width, label='Recall@K', alpha=0.8)
        ax1.bar(x_pos + width, success_df['ndcg_at_k'], width, label='NDCG@K', alpha=0.8)
        
        ax1.set_xlabel('System')
        ax1.set_ylabel('Score')
        ax1.set_title('Retrieval Quality Metrics')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(success_df['system'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # 2. Query Performance
        ax2 = axes[0, 1]
        query_times = []
        sim_times = []
        systems = []
        
        for _, row in success_df.iterrows():
            try:
                query_time = float(row['avg_query_time'].replace('s', ''))
                query_times.append(query_time)
                systems.append(row['system'])
                
                # Add simulation time if available
                if 'avg_simulation_time' in row and pd.notna(row['avg_simulation_time']):
                    sim_time = float(row['avg_simulation_time'].replace('s', ''))
                    sim_times.append(sim_time)
                else:
                    sim_times.append(0)
            except:
                continue
        
        if query_times:
            x_pos = np.arange(len(systems))
            ax2.bar(x_pos, query_times, label='PIR Query Time', alpha=0.8)
            if any(sim_times):
                ax2.bar(x_pos, sim_times, bottom=query_times, label='Simulation Time', alpha=0.8)
            
            ax2.set_xlabel('System')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title('Query Performance Timing')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(systems, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Communication Overhead
        ax3 = axes[1, 0]
        comm_data = []
        comm_systems = []
        
        for _, row in success_df.iterrows():
            try:
                comm_kb = float(row['communication_kb'])
                comm_data.append(comm_kb)
                comm_systems.append(row['system'])
            except:
                continue
        
        if comm_data:
            bars = ax3.bar(comm_systems, comm_data, alpha=0.8)
            ax3.set_xlabel('System')
            ax3.set_ylabel('Communication (KB)')
            ax3.set_title('Communication Overhead')
            plt.setp(ax3.get_xticklabels(), rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, comm_data):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}', ha='center', va='bottom')
        
        # 4. Setup Time Comparison
        ax4 = axes[1, 1]
        setup_times = []
        setup_systems = []
        
        for _, row in success_df.iterrows():
            try:
                setup_time = float(row['setup_time'].replace('s', ''))
                setup_times.append(setup_time)
                setup_systems.append(row['system'])
            except:
                continue
        
        if setup_times:
            bars = ax4.bar(setup_systems, setup_times, alpha=0.8, color='orange')
            ax4.set_xlabel('System')
            ax4.set_ylabel('Setup Time (seconds)')
            ax4.set_title('System Setup Time')
            plt.setp(ax4.get_xticklabels(), rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, setup_times):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved performance comparison plot: {save_path}")
        plt.close()
    
    def create_detailed_metrics_table(self, df: pd.DataFrame, save_path: Path):
        """Create a detailed metrics comparison table."""
        
        # Filter successful results
        success_df = df[df['status'].str.contains('Success')].copy()
        
        if len(success_df) == 0:
            print("⚠️  No successful results for table")
            return
        
        # Create a clean table
        table_data = []
        
        for _, row in success_df.iterrows():
            table_row = {
                'System': row['system'],
                'Approach': row.get('approach', 'Unknown'),
                'Documents': row['n_documents'],
                'Queries': row['n_queries'],
                'Setup Time': row['setup_time'],
                'Avg Query Time': row['avg_query_time'],
                'Precision@K': row['precision_at_k'],
                'Recall@K': row['recall_at_k'],
                'NDCG@K': row['ndcg_at_k'],
                'Communication (KB)': row['communication_kb']
            }
            table_data.append(table_row)
        
        table_df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_path = save_path.with_suffix('.csv')
        table_df.to_csv(csv_path, index=False)
        print(f"📋 Saved detailed metrics table: {csv_path}")
        
        return table_df
    
    def print_analysis_summary(self, df: pd.DataFrame):
        """Print a comprehensive analysis summary."""
        
        print(f"\n{'='*70}")
        print(f"RETRIEVAL PERFORMANCE ANALYSIS SUMMARY")
        print(f"{'='*70}")
        
        # Basic stats
        total_experiments = len(df)
        successful_experiments = len(df[df['status'].str.contains('Success')])
        failed_experiments = total_experiments - successful_experiments
        
        print(f"📊 Experiment Overview:")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Successful: {successful_experiments}")
        print(f"  Failed: {failed_experiments}")
        
        if successful_experiments == 0:
            print("❌ No successful experiments to analyze")
            return
        
        # System performance comparison
        success_df = df[df['status'].str.contains('Success')].copy()
        
        print(f"\n🏆 System Performance Ranking:")
        print(f"{'System':<12} {'Precision@K':<12} {'Recall@K':<10} {'NDCG@K':<10} {'Query Time':<12}")
        print("-" * 60)
        
        for _, row in success_df.iterrows():
            print(f"{row['system']:<12} {row['precision_at_k']:<12} {row['recall_at_k']:<10} "
                  f"{row['ndcg_at_k']:<10} {row['avg_query_time']:<12}")
        
        # Find best performing system
        try:
            # Convert metrics to numeric for comparison
            metrics_df = success_df.copy()
            for col in ['precision_at_k', 'recall_at_k', 'ndcg_at_k']:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
            
            best_precision = metrics_df.loc[metrics_df['precision_at_k'].idxmax(), 'system']
            best_recall = metrics_df.loc[metrics_df['recall_at_k'].idxmax(), 'system']
            best_ndcg = metrics_df.loc[metrics_df['ndcg_at_k'].idxmax(), 'system']
            
            print(f"\n🥇 Best Performance:")
            print(f"  Highest Precision@K: {best_precision}")
            print(f"  Highest Recall@K: {best_recall}")
            print(f"  Highest NDCG@K: {best_ndcg}")
            
        except Exception as e:
            print(f"⚠️  Could not determine best performing system: {e}")
        
        # Efficiency analysis
        print(f"\n⚡ Efficiency Analysis:")
        for _, row in success_df.iterrows():
            try:
                query_time = float(row['avg_query_time'].replace('s', ''))
                comm_kb = float(row['communication_kb'])
                precision = float(row['precision_at_k'])
                
                efficiency_score = precision / (query_time + comm_kb/1000)  # Simple efficiency metric
                print(f"  {row['system']}: {efficiency_score:.3f} (precision per second per KB)")
            except:
                print(f"  {row['system']}: Unable to calculate efficiency")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if successful_experiments >= 2:
            print(f"  1. Compare retrieval quality vs. performance trade-offs")
            print(f"  2. Consider communication overhead for your use case")
            print(f"  3. Test with larger datasets to validate scalability")
            print(f"  4. Run multiple trials for statistical significance")
        else:
            print(f"  1. Run experiments with all three systems for comparison")
            print(f"  2. Test with different dataset sizes")
            print(f"  3. Experiment with different parameter settings")
    
    def analyze_all_results(self):
        """Find and analyze all retrieval results."""
        
        print("🔍 Searching for retrieval performance results...")
        
        result_files = self.find_retrieval_results()
        
        if not result_files:
            print("❌ No retrieval performance result files found!")
            print("💡 Make sure to run retrieval experiments first:")
            print("   python experiments/comprehensive_experiment.py --experiment retrieval")
            return
        
        print(f"📁 Found {len(result_files)} result files:")
        for f in result_files:
            print(f"  - {f}")
        
        # Analyze the most recent file
        latest_file = result_files[0]
        print(f"\n📊 Analyzing most recent: {latest_file.name}")
        
        results = self.load_retrieval_results(latest_file)
        if results is None:
            print("❌ Failed to load results")
            return
        
        # Create analysis
        df = self.analyze_single_result(results)
        
        # Print summary
        self.print_analysis_summary(df)
        
        # Generate plots and tables
        timestamp = latest_file.stem.split('_')[-1] if '_' in latest_file.stem else 'latest'
        
        plot_path = self.figures_dir / f"retrieval_performance_analysis_{timestamp}.png"
        self.create_performance_comparison_plot(df, plot_path)
        
        table_path = self.figures_dir / f"retrieval_metrics_table_{timestamp}"
        self.create_detailed_metrics_table(df, table_path)
        
        print(f"\n✅ Analysis complete! Results saved to: {self.figures_dir}")
        
        return df

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze Retrieval Performance Results")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory containing result files")
    parser.add_argument("--file", type=str, 
                       help="Specific result file to analyze")
    
    args = parser.parse_args()
    
    analyzer = RetrievalResultsAnalyzer(args.results_dir)
    
    if args.file:
        # Analyze specific file
        file_path = Path(args.file)
        if file_path.exists():
            results = analyzer.load_retrieval_results(file_path)
            if results:
                df = analyzer.analyze_single_result(results)
                analyzer.print_analysis_summary(df)
        else:
            print(f"❌ File not found: {args.file}")
    else:
        # Analyze all results
        analyzer.analyze_all_results()

if __name__ == "__main__":
    main()
