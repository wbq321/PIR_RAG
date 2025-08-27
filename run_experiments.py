#!/usr/bin/env python3
"""
Simple runner for comprehensive PIR experiments
"""

import subprocess
import sys
import os
from pathlib import Path

def run_experiments():
    """Run all PIR experiments and generate analysis."""
    
    print("Comprehensive PIR Experiment Runner")
    print("===================================")
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Add src to Python path
    src_path = script_dir / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        print("\n1. Running comprehensive experiments...")
        print("   This may take several minutes...")
        
        # Run comprehensive experiments
        cmd = [
            sys.executable, "experiments/comprehensive_experiment.py",
            "--experiment", "all",
            "--n-docs", "1000",
            "--n-queries", "10"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Experiments failed!")
            print(f"Error: {result.stderr}")
            return False
        
        print("✅ Experiments completed successfully!")
        
        print("\n2. Generating analysis plots...")
        
        # Run analysis
        cmd = [
            sys.executable, "experiments/analyze_results.py",
            "--generate-all"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"⚠️ Analysis failed, but experiment data is available")
            print(f"Error: {result.stderr}")
        else:
            print("✅ Analysis completed successfully!")
        
        # Show results
        results_dir = script_dir / "results"
        figures_dir = results_dir / "figures"
        
        print(f"\n📊 Results Summary:")
        print(f"==================")
        print(f"📁 Data files: {results_dir}")
        print(f"📈 Plots: {figures_dir}")
        
        if results_dir.exists():
            json_files = list(results_dir.glob("*.json"))
            csv_files = list(results_dir.glob("*.csv"))
            txt_files = list(results_dir.glob("*.txt"))
            
            print(f"\nGenerated files:")
            print(f"- {len(json_files)} JSON data files")
            print(f"- {len(csv_files)} CSV summary files")
            print(f"- {len(txt_files)} text reports")
            
            if figures_dir.exists():
                plot_files = list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.pdf"))
                print(f"- {len(plot_files)} plot files")
        
        print(f"\n🎯 Key Analyses:")
        print(f"- Setup time comparison across all three systems")
        print(f"- Query latency and step-by-step timing breakdown")
        print(f"- Communication cost analysis (upload/download)")
        print(f"- Scalability testing across dataset sizes")
        print(f"- Parameter sensitivity analysis")
        print(f"- Real vs simulated crypto performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        return False


if __name__ == "__main__":
    success = run_experiments()
    if not success:
        sys.exit(1)
    
    print(f"\n🚀 All experiments completed!")
    print(f"Check the results directory for detailed analysis.")
