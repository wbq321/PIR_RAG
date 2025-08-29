"""
Analysis and Plotting Script for PIR Experiments

This script loads experiment results and creates comprehensive visualizations
for timing analysis, scalability assessment, and performance comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.patches as mpatches

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set global font sizes for all plots
plt.rcParams.update({
    'font.size': 18,           # Base font size
    'axes.titlesize': 22,      # Title font size
    'axes.labelsize': 20,      # Axis label font size
    'xtick.labelsize': 18,     # X-axis tick labels
    'ytick.labelsize': 18,     # Y-axis tick labels
    'legend.fontsize': 18,     # Legend font size
    'figure.titlesize': 24     # Figure title font size
})

# Import retrieval analysis
try:
    from analyze_retrieval_performance import RetrievalAnalyzer
    RETRIEVAL_ANALYSIS_AVAILABLE = True
except ImportError:
    RETRIEVAL_ANALYSIS_AVAILABLE = False


class PIRAnalyzer:
    """Analyzer for PIR experiment results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def load_json_results(self, json_file: str) -> Dict[str, Any]:
        """Load results from JSON file with error handling."""
        file_path = self.results_dir / json_file
        try:
            print(f"Loading JSON file: {file_path}")

            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Check file size
            file_size = file_path.stat().st_size
            print(f"File size: {file_size} bytes")

            if file_size == 0:
                raise ValueError(f"File is empty: {file_path}")

            with open(file_path, 'r') as f:
                content = f.read()
                if not content.strip():
                    raise ValueError(f"File contains only whitespace: {file_path}")

                # Try to parse JSON
                import json
                try:
                    results = json.loads(content)
                    print(f"✅ Successfully loaded JSON with {len(results)} top-level keys")
                    return results
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing error in {file_path}:")
                    print(f"   Error: {e}")
                    print(f"   Content preview (first 200 chars): {content[:200]}")
                    raise

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            raise

    def plot_single_experiment_timing(self, results: Dict[str, Any], save_name: str = "timing_breakdown"):
        """Plot detailed timing breakdown for single experiment as separate figures."""

        systems = []
        setup_times = []
        query_times = []
        upload_bytes = []
        download_bytes = []

        for system_name, result in results.items():
            if result is None:
                continue

            systems.append(result['system'])
            setup_times.append(result['setup_time'])
            query_times.append(result['avg_query_time'])
            upload_bytes.append(result['avg_upload_bytes'])
            download_bytes.append(result['avg_download_bytes'])

        # Individual figure 1: Setup time comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        bars = ax.bar(systems, setup_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Setup Time Comparison', fontweight='bold', fontsize=18)
        ax.set_ylabel('Time (seconds)', fontweight='bold', fontsize=16)
        ax.tick_params(axis='x', rotation=45)

        # Add values on bars
        for i, v in enumerate(setup_times):
            ax.text(i, v + max(setup_times) * 0.01, f'{v:.2f}s',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}_setup_time.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}_setup_time.pdf", bbox_inches='tight')
        plt.close()

        # Individual figure 2: Query time comparison with error bars
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        query_stds = [results[list(results.keys())[i]]['std_query_time'] if list(results.keys())[i] in results else 0
                     for i in range(len(systems))]

        bars = ax.bar(systems, query_times, yerr=query_stds, capsize=5,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_title('Average Query Time Comparison', fontweight='bold', fontsize=18)
        ax.set_ylabel('Time (seconds)', fontweight='bold', fontsize=16)
        ax.tick_params(axis='x', rotation=45)

        # Add values on bars
        for i, v in enumerate(query_times):
            ax.text(i, v + max(query_times) * 0.01, f'{v:.3f}s',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}_query_time.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}_query_time.pdf", bbox_inches='tight')
        plt.close()

        # Individual figure 3: Communication cost comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        x = np.arange(len(systems))
        width = 0.35

        ax.bar(x - width/2, upload_bytes, width, label='Upload', color='#FF9999')
        ax.bar(x + width/2, download_bytes, width, label='Download', color='#99CCFF')
        ax.set_title('Communication Cost Comparison', fontweight='bold', fontsize=18)
        ax.set_ylabel('Bytes', fontweight='bold', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(systems, rotation=45)
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}_communication.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}_communication.pdf", bbox_inches='tight')
        plt.close()

        # Individual figure 4: Step-by-step timing breakdown
        if 'pir_rag' in results and results['pir_rag'] is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            step_times = results['pir_rag']['step_times'][0]  # First query
            steps = list(step_times.keys())
            times = list(step_times.values())

            ax.pie(times, labels=steps, autopct='%1.1f%%', startangle=90)
            ax.set_title('PIR-RAG Step Breakdown (First Query)', fontweight='bold', fontsize=18)

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{save_name}_step_breakdown.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / f"{save_name}_step_breakdown.pdf", bbox_inches='tight')
            plt.close()

        print(f"Generated individual timing plots: {save_name}_setup_time, {save_name}_query_time, {save_name}_communication, {save_name}_step_breakdown")

    def plot_scalability_analysis(self, results: Dict[str, Any], save_name: str = "scalability"):
        """Plot scalability analysis across different dataset sizes as separate figures."""
        doc_sizes = results['doc_sizes']

        # Extract data for each system
        systems_data = {}
        colors = {'PIR-RAG': '#FF6B6B', 'Graph-PIR': '#4ECDC4', 'Tiptoe': '#45B7D1'}

        for system_key, system_name in [('pir_rag_results', 'PIR-RAG'),
                                       ('graph_pir_results', 'Graph-PIR'),
                                       ('tiptoe_results', 'Tiptoe')]:
            if system_key in results:
                setup_times = []
                query_times = []
                upload_bytes = []
                download_bytes = []
                valid_sizes = []

                for i, result in enumerate(results[system_key]):
                    if result is not None:
                        setup_times.append(result['setup_time'])
                        query_times.append(result['avg_query_time'])
                        upload_bytes.append(result['avg_upload_bytes'])
                        download_bytes.append(result['avg_download_bytes'])
                        valid_sizes.append(doc_sizes[i])

                systems_data[system_name] = {
                    'sizes': valid_sizes,
                    'setup_times': setup_times,
                    'query_times': query_times,
                    'upload_bytes': upload_bytes,
                    'download_bytes': download_bytes
                }

        # Individual figure 1: Setup time scalability
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for system_name, data in systems_data.items():
            ax.plot(data['sizes'], data['setup_times'], 'o-',
                  label=system_name, color=colors[system_name], linewidth=2, markersize=8)

        ax.set_title('Setup Time Scalability', fontweight='bold', fontsize=18)
        ax.set_xlabel('Number of Documents', fontweight='bold', fontsize=16)
        ax.set_ylabel('Setup Time (seconds)', fontweight='bold', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}_setup_time.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}_setup_time.pdf", bbox_inches='tight')
        plt.close()

        # Individual figure 2: Query time scalability
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for system_name, data in systems_data.items():
            ax.plot(data['sizes'], data['query_times'], 'o-',
                  label=system_name, color=colors[system_name], linewidth=2, markersize=8)

        ax.set_title('Query Time Scalability', fontweight='bold', fontsize=18)
        ax.set_xlabel('Number of Documents', fontweight='bold', fontsize=16)
        ax.set_ylabel('Average Query Time (seconds)', fontweight='bold', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}_query_time.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}_query_time.pdf", bbox_inches='tight')
        plt.close()

        # Individual figure 3: Upload bytes scalability
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for system_name, data in systems_data.items():
            ax.plot(data['sizes'], data['upload_bytes'], 'o-',
                  label=system_name, color=colors[system_name], linewidth=2, markersize=8)

        ax.set_title('Upload Communication Scalability', fontweight='bold', fontsize=18)
        ax.set_xlabel('Number of Documents', fontweight='bold', fontsize=16)
        ax.set_ylabel('Upload Bytes per Query', fontweight='bold', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}_upload_bytes.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}_upload_bytes.pdf", bbox_inches='tight')
        plt.close()

        # Individual figure 4: Download bytes scalability
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for system_name, data in systems_data.items():
            ax.plot(data['sizes'], data['download_bytes'], 'o-',
                  label=system_name, color=colors[system_name], linewidth=2, markersize=8)

        ax.set_title('Download Communication Scalability', fontweight='bold', fontsize=18)
        ax.set_xlabel('Number of Documents', fontweight='bold', fontsize=16)
        ax.set_ylabel('Download Bytes per Query', fontweight='bold', fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.figures_dir / f"{save_name}_download_bytes.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f"{save_name}_download_bytes.pdf", bbox_inches='tight')
        plt.close()

        print(f"Generated individual scalability plots: {save_name}_setup_time, {save_name}_query_time, {save_name}_upload_bytes, {save_name}_download_bytes")

    def plot_detailed_step_timing(self, results: Dict[str, Any], save_name: str = "step_timing"):
        """Plot detailed step-by-step timing analysis as separate figures."""

        system_colors = {'PIR-RAG': '#FF6B6B', 'Graph-PIR': '#4ECDC4', 'Tiptoe': '#45B7D1'}

        # Individual figure 1: PIR-RAG step timing
        if 'pir_rag' in results and results['pir_rag'] is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            step_times = results['pir_rag']['step_times']
            avg_steps = {}

            for step in step_times[0].keys():
                avg_steps[step] = np.mean([st[step] for st in step_times])

            steps = list(avg_steps.keys())
            times = list(avg_steps.values())

            wedges, texts, autotexts = ax.pie(times, labels=steps, autopct='%1.1f%%',
                                              startangle=90, colors=plt.cm.Set3.colors)
            ax.set_title('PIR-RAG Step Breakdown', fontweight='bold', fontsize=18)

            # Make percentage text bold and larger
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{save_name}_pir_rag.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / f"{save_name}_pir_rag.pdf", bbox_inches='tight')
            plt.close()

        # Individual figure 2: Graph-PIR step timing
        if 'graph_pir' in results and results['graph_pir'] is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            step_times = results['graph_pir']['step_times']
            avg_steps = {}

            for step in step_times[0].keys():
                avg_steps[step] = np.mean([st[step] for st in step_times])

            steps = list(avg_steps.keys())
            times = list(avg_steps.values())

            wedges, texts, autotexts = ax.pie(times, labels=steps, autopct='%1.1f%%',
                                              startangle=90, colors=plt.cm.Set2.colors)
            ax.set_title('Graph-PIR Step Breakdown', fontweight='bold', fontsize=18)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{save_name}_graph_pir.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / f"{save_name}_graph_pir.pdf", bbox_inches='tight')
            plt.close()

        # Individual figure 3: Tiptoe step timing
        if 'tiptoe' in results and results['tiptoe'] is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            step_times = results['tiptoe']['step_times']
            avg_steps = {}

            for step in step_times[0].keys():
                avg_steps[step] = np.mean([st[step] for st in step_times])

            steps = list(avg_steps.keys())
            times = list(avg_steps.values())

            wedges, texts, autotexts = ax.pie(times, labels=steps, autopct='%1.1f%%',
                                              startangle=90, colors=plt.cm.Pastel1.colors)
            ax.set_title('Tiptoe Step Breakdown', fontweight='bold', fontsize=18)

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{save_name}_tiptoe.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / f"{save_name}_tiptoe.pdf", bbox_inches='tight')
            plt.close()

        print(f"Generated individual step timing plots: {save_name}_pir_rag, {save_name}_graph_pir, {save_name}_tiptoe")

    def plot_parameter_sensitivity(self, results: Dict[str, Any], save_name: str = "parameter_sensitivity"):
        """Plot parameter sensitivity analysis as separate figures."""

        # Individual figure 1: PIR-RAG k_clusters sensitivity
        if 'pir_rag_k_clusters' in results:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax1_twin = ax1.twinx()

            k_values = []
            setup_times = []
            query_times = []

            for result in results['pir_rag_k_clusters']:
                k_values.append(result['parameters']['k_clusters'])
                setup_times.append(result['setup_time'])
                query_times.append(result['avg_query_time'])

            line1 = ax1.plot(k_values, setup_times, 'o-', color='#FF6B6B',
                           linewidth=2, markersize=8, label='Setup Time')
            line2 = ax1_twin.plot(k_values, query_times, 's-', color='#4ECDC4',
                                linewidth=2, markersize=8, label='Query Time')

            ax1.set_xlabel('Number of Clusters (k)', fontweight='bold', fontsize=12)
            ax1.set_ylabel('Setup Time (seconds)', color='#FF6B6B', fontweight='bold', fontsize=12)
            ax1_twin.set_ylabel('Query Time (seconds)', color='#4ECDC4', fontweight='bold', fontsize=12)
            ax1.set_title('PIR-RAG: Effect of k_clusters', fontweight='bold', fontsize=14)
            ax1.grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{save_name}_pir_rag_k_clusters.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / f"{save_name}_pir_rag_k_clusters.pdf", bbox_inches='tight')
            plt.close()

        # Individual figure 2: Graph-PIR k_neighbors sensitivity
        if 'graph_pir_k_neighbors' in results:
            fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))
            ax2_twin = ax2.twinx()

            k_values = []
            setup_times = []
            query_times = []

            for result in results['graph_pir_k_neighbors']:
                k_values.append(result['parameters']['k_neighbors'])
                setup_times.append(result['setup_time'])
                query_times.append(result['avg_query_time'])

            line1 = ax2.plot(k_values, setup_times, 'o-', color='#FF6B6B',
                           linewidth=2, markersize=8, label='Setup Time')
            line2 = ax2_twin.plot(k_values, query_times, 's-', color='#4ECDC4',
                                linewidth=2, markersize=8, label='Query Time')

            ax2.set_xlabel('Number of Neighbors (k)', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Setup Time (seconds)', color='#FF6B6B', fontweight='bold', fontsize=12)
            ax2_twin.set_ylabel('Query Time (seconds)', color='#4ECDC4', fontweight='bold', fontsize=12)
            ax2.set_title('Graph-PIR: Effect of k_neighbors', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')

            plt.tight_layout()
            plt.savefig(self.figures_dir / f"{save_name}_graph_pir_k_neighbors.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / f"{save_name}_graph_pir_k_neighbors.pdf", bbox_inches='tight')
            plt.close()

        print(f"Generated individual parameter sensitivity plots: {save_name}_pir_rag_k_clusters, {save_name}_graph_pir_k_neighbors")

    def generate_summary_report(self, results: Dict[str, Any], output_file: str = "summary_report.txt"):
        """Generate a comprehensive text summary report."""
        report_path = self.results_dir / output_file

        with open(report_path, 'w') as f:
            f.write("PIR SYSTEMS PERFORMANCE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Single experiment summary
            if any(key in results for key in ['pir_rag', 'graph_pir', 'tiptoe']):
                f.write("SINGLE EXPERIMENT SUMMARY\n")
                f.write("-" * 25 + "\n")

                for system_key, system_name in [('pir_rag', 'PIR-RAG'),
                                               ('graph_pir', 'Graph-PIR'),
                                               ('tiptoe', 'Tiptoe')]:
                    if system_key in results and results[system_key] is not None:
                        result = results[system_key]
                        f.write(f"\n{system_name}:\n")
                        f.write(f"  Setup Time: {result['setup_time']:.3f}s\n")
                        f.write(f"  Average Query Time: {result['avg_query_time']:.3f}s ± {result['std_query_time']:.3f}s\n")
                        f.write(f"  Average Upload: {result['avg_upload_bytes']:,} bytes\n")
                        f.write(f"  Average Download: {result['avg_download_bytes']:,} bytes\n")
                        f.write(f"  Total Communication: {result['avg_upload_bytes'] + result['avg_download_bytes']:,} bytes\n")
                        f.write(f"  Documents: {result['n_documents']}\n")
                        f.write(f"  Embedding Dimension: {result['embedding_dim']}\n")

                # Performance comparison
                if 'pir_rag' in results and 'graph_pir' in results:
                    pir_rag = results['pir_rag']
                    graph_pir = results['graph_pir']

                    if pir_rag is not None and graph_pir is not None:
                        f.write(f"\nPERFORMANCE COMPARISON:\n")
                        setup_ratio = graph_pir['setup_time'] / pir_rag['setup_time']
                        query_ratio = graph_pir['avg_query_time'] / pir_rag['avg_query_time']
                        comm_ratio = ((graph_pir['avg_upload_bytes'] + graph_pir['avg_download_bytes']) /
                                     (pir_rag['avg_upload_bytes'] + pir_rag['avg_download_bytes']))

                        f.write(f"  Graph-PIR vs PIR-RAG Setup Time Ratio: {setup_ratio:.2f}x\n")
                        f.write(f"  Graph-PIR vs PIR-RAG Query Time Ratio: {query_ratio:.2f}x\n")
                        f.write(f"  Graph-PIR vs PIR-RAG Communication Ratio: {comm_ratio:.2f}x\n")

            # Scalability analysis
            if 'doc_sizes' in results:
                f.write(f"\n\nSCALABILITY ANALYSIS\n")
                f.write("-" * 19 + "\n")

                doc_sizes = results['doc_sizes']
                f.write(f"Tested document sizes: {doc_sizes}\n\n")

                for system_key, system_name in [('pir_rag_results', 'PIR-RAG'),
                                               ('graph_pir_results', 'Graph-PIR'),
                                               ('tiptoe_results', 'Tiptoe')]:
                    if system_key in results:
                        f.write(f"{system_name} Scalability:\n")

                        for i, result in enumerate(results[system_key]):
                            if result is not None:
                                f.write(f"  {doc_sizes[i]} docs: Setup {result['setup_time']:.2f}s, "
                                       f"Query {result['avg_query_time']:.3f}s, "
                                       f"Comm {result['avg_upload_bytes'] + result['avg_download_bytes']:,} bytes\n")
                        f.write("\n")

            f.write("\nReport generated on: " + str(pd.Timestamp.now()) + "\n")

        print(f"Summary report saved to: {report_path}")


def main():
    """Main analysis runner."""
    parser = argparse.ArgumentParser(description="Analyze PIR Experiment Results")
    parser.add_argument("--results-dir", default="results", help="Directory containing results")
    parser.add_argument("--single-file", help="JSON file for single experiment analysis")
    parser.add_argument("--scalability-file", help="JSON file for scalability analysis")
    parser.add_argument("--sensitivity-file", help="JSON file for parameter sensitivity analysis")
    parser.add_argument("--generate-all", action="store_true", help="Generate all plots from latest files")

    args = parser.parse_args()

    analyzer = PIRAnalyzer(args.results_dir)

    if args.generate_all:
        # Find latest files
        results_path = Path(args.results_dir)
        single_files = list(results_path.glob("single_experiment_*.json"))
        scalability_files = list(results_path.glob("scalability_*.json"))
        sensitivity_files = list(results_path.glob("parameter_sensitivity_*.json"))

        if single_files:
            latest_single = max(single_files, key=lambda x: x.stat().st_mtime)
            print(f"Analyzing single experiment: {latest_single}")
            results = analyzer.load_json_results(latest_single.name)
            analyzer.plot_single_experiment_timing(results)
            analyzer.plot_detailed_step_timing(results)
            analyzer.generate_summary_report(results)

        if scalability_files:
            # Try files in reverse chronological order until we find a valid one
            scalability_files_sorted = sorted(scalability_files, key=lambda x: x.stat().st_mtime, reverse=True)

            for scalability_file in scalability_files_sorted:
                try:
                    print(f"Trying to analyze scalability: {scalability_file}")
                    results = analyzer.load_json_results(scalability_file.name)
                    analyzer.plot_scalability_analysis(results)
                    print(f"✅ Successfully analyzed: {scalability_file}")
                    break
                except Exception as e:
                    print(f"❌ Failed to analyze {scalability_file}: {e}")
                    continue
            else:
                print("❌ No valid scalability files found")

        if sensitivity_files:
            latest_sensitivity = max(sensitivity_files, key=lambda x: x.stat().st_mtime)
            print(f"Analyzing sensitivity: {latest_sensitivity}")
            results = analyzer.load_json_results(latest_sensitivity.name)
            analyzer.plot_parameter_sensitivity(results)

    else:
        if args.single_file:
            print(f"Analyzing single experiment: {args.single_file}")
            results = analyzer.load_json_results(args.single_file)
            analyzer.plot_single_experiment_timing(results)
            analyzer.plot_detailed_step_timing(results)
            analyzer.generate_summary_report(results)

        if args.scalability_file:
            try:
                print(f"Analyzing scalability: {args.scalability_file}")
                results = analyzer.load_json_results(args.scalability_file)
                analyzer.plot_scalability_analysis(results)
                print(f"✅ Successfully analyzed: {args.scalability_file}")
            except Exception as e:
                print(f"❌ Failed to analyze {args.scalability_file}: {e}")
                print("💡 Try using --generate-all to automatically find valid files")

        if args.sensitivity_file:
            print(f"Analyzing sensitivity: {args.sensitivity_file}")
            results = analyzer.load_json_results(args.sensitivity_file)
            analyzer.plot_parameter_sensitivity(results)

        # Add retrieval performance analysis
        if args.generate_all and RETRIEVAL_ANALYSIS_AVAILABLE:
            print("Analyzing retrieval performance results...")
            retrieval_analyzer = RetrievalAnalyzer(args.results_dir)
            retrieval_analyzer.generate_all_retrieval_plots(analyzer.figures_dir)

    print("Analysis complete! Check the 'figures' directory for plots.")


if __name__ == "__main__":
    main()
    main()
