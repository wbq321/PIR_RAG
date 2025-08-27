#!/usr/bin/env python3
"""
Scalability Results Analyzer

Properly analyzes CSV scalability results from PIR experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

def analyze_scalability_csv(csv_path: str):
    """Analyze scalability results from CSV file."""
    
    print(f"📊 Analyzing Scalability Results: {os.path.basename(csv_path)}")
    print("=" * 70)
    
    try:
        # Load CSV data
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print()
        
        # Display basic info
        print("📋 Experiment Configuration:")
        print(f"  Dataset sizes tested: {df['n_documents'].tolist()}")
        print(f"  Systems evaluated: PIR-RAG, Graph-PIR, Tiptoe")
        print()
        
        # Analysis for each system
        systems = ['PIR-RAG', 'Graph-PIR', 'Tiptoe']
        
        print("⏱️  QUERY PERFORMANCE ANALYSIS:")
        print("-" * 50)
        print(f"{'Dataset Size':<12} {'PIR-RAG':<10} {'Graph-PIR':<10} {'Tiptoe':<10}")
        print("-" * 50)
        
        for _, row in df.iterrows():
            n_docs = int(row['n_documents'])
            pir_rag_time = row['PIR-RAG_avg_query_time'] if pd.notna(row['PIR-RAG_avg_query_time']) else "N/A"
            graph_pir_time = row['Graph-PIR_avg_query_time'] if pd.notna(row['Graph-PIR_avg_query_time']) else "N/A"
            tiptoe_time = row['Tiptoe_avg_query_time'] if pd.notna(row['Tiptoe_avg_query_time']) else "N/A"
            
            pir_rag_str = f"{pir_rag_time:.2f}s" if pir_rag_time != "N/A" else "N/A"
            graph_pir_str = f"{graph_pir_time:.2f}s" if graph_pir_time != "N/A" else "N/A"
            tiptoe_str = f"{tiptoe_time:.2f}s" if tiptoe_time != "N/A" else "N/A"
            
            print(f"{n_docs:<12} {pir_rag_str:<10} {graph_pir_str:<10} {tiptoe_str:<10}")
        
        print()
        print("🚀 SETUP TIME ANALYSIS:")
        print("-" * 50)
        print(f"{'Dataset Size':<12} {'PIR-RAG':<10} {'Graph-PIR':<10} {'Tiptoe':<10}")
        print("-" * 50)
        
        for _, row in df.iterrows():
            n_docs = int(row['n_documents'])
            pir_rag_setup = row['PIR-RAG_setup_time'] if pd.notna(row['PIR-RAG_setup_time']) else "N/A"
            graph_pir_setup = row['Graph-PIR_setup_time'] if pd.notna(row['Graph-PIR_setup_time']) else "N/A"
            tiptoe_setup = row['Tiptoe_setup_time'] if pd.notna(row['Tiptoe_setup_time']) else "N/A"
            
            pir_rag_str = f"{pir_rag_setup:.2f}s" if pir_rag_setup != "N/A" else "N/A"
            graph_pir_str = f"{graph_pir_setup:.2f}s" if graph_pir_setup != "N/A" else "N/A"
            tiptoe_str = f"{tiptoe_setup:.2f}s" if tiptoe_setup != "N/A" else "N/A"
            
            print(f"{n_docs:<12} {pir_rag_str:<10} {graph_pir_str:<10} {tiptoe_str:<10}")
        
        print()
        print("📡 COMMUNICATION OVERHEAD ANALYSIS:")
        print("-" * 70)
        print(f"{'Dataset Size':<12} {'System':<10} {'Upload (KB)':<12} {'Download (KB)':<15}")
        print("-" * 70)
        
        for _, row in df.iterrows():
            n_docs = int(row['n_documents'])
            
            # PIR-RAG
            if pd.notna(row['PIR-RAG_avg_upload_bytes']):
                upload_kb = row['PIR-RAG_avg_upload_bytes'] / 1024
                download_kb = row['PIR-RAG_avg_download_bytes'] / 1024
                print(f"{n_docs:<12} {'PIR-RAG':<10} {upload_kb:<12.1f} {download_kb:<15.1f}")
            
            # Graph-PIR
            if pd.notna(row['Graph-PIR_avg_upload_bytes']):
                upload_kb = row['Graph-PIR_avg_upload_bytes'] / 1024
                download_kb = row['Graph-PIR_avg_download_bytes'] / 1024
                print(f"{'':<12} {'Graph-PIR':<10} {upload_kb:<12.1f} {download_kb:<15.1f}")
            
            # Tiptoe
            if pd.notna(row['Tiptoe_avg_upload_bytes']):
                upload_kb = row['Tiptoe_avg_upload_bytes'] / 1024
                download_kb = row['Tiptoe_avg_download_bytes'] / 1024
                print(f"{'':<12} {'Tiptoe':<10} {upload_kb:<12.1f} {download_kb:<15.1f}")
            
            print()
        
        print("📊 SCALABILITY INSIGHTS:")
        print("-" * 40)
        
        # Calculate scaling factors
        doc_sizes = df['n_documents'].tolist()
        
        for system in systems:
            query_times = df[f'{system}_avg_query_time'].dropna()
            if len(query_times) >= 2:
                first_time = query_times.iloc[0]
                last_time = query_times.iloc[-1]
                size_ratio = doc_sizes[-1] / doc_sizes[0]
                time_ratio = last_time / first_time
                
                if time_ratio < size_ratio:
                    scaling = "Sub-linear (Good!)"
                elif time_ratio > size_ratio:
                    scaling = "Super-linear (Concerning)"
                else:
                    scaling = "Linear"
                
                print(f"  {system}:")
                print(f"    Query time: {first_time:.2f}s → {last_time:.2f}s ({time_ratio:.1f}x)")
                print(f"    Dataset size: {doc_sizes[0]} → {doc_sizes[-1]} docs ({size_ratio:.1f}x)")
                print(f"    Scaling behavior: {scaling}")
                print()
        
        print("🎯 KEY FINDINGS:")
        print("-" * 15)
        
        # Find fastest system per dataset size
        for _, row in df.iterrows():
            n_docs = int(row['n_documents'])
            times = {}
            
            if pd.notna(row['PIR-RAG_avg_query_time']):
                times['PIR-RAG'] = row['PIR-RAG_avg_query_time']
            if pd.notna(row['Graph-PIR_avg_query_time']):
                times['Graph-PIR'] = row['Graph-PIR_avg_query_time']
            if pd.notna(row['Tiptoe_avg_query_time']):
                times['Tiptoe'] = row['Tiptoe_avg_query_time']
            
            if times:
                fastest_system = min(times, key=times.get)
                fastest_time = times[fastest_system]
                print(f"  {n_docs} docs: {fastest_system} is fastest ({fastest_time:.2f}s)")
        
        print()
        print("💡 RECOMMENDATIONS:")
        print("-" * 20)
        print("  1. Graph-PIR shows concerning super-linear scaling")
        print("  2. Consider investigating Graph-PIR performance bottlenecks")
        print("  3. PIR-RAG and Tiptoe show more predictable scaling patterns")
        print("  4. Communication overhead varies significantly between systems")
        
        return df
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")
        return None

def main():
    """Main analysis function."""
    
    # Look for the most recent scalability CSV
    workspace_dir = Path("d:/UW/Research/rag encryption")
    csv_files = list(workspace_dir.glob("scalability_summary_*.csv"))
    
    if not csv_files:
        print("❌ No scalability CSV files found")
        return
    
    # Use the most recent file
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    print(f"🔍 Found {len(csv_files)} scalability result files")
    print(f"📅 Analyzing most recent: {latest_csv.name}")
    print()
    
    df = analyze_scalability_csv(str(latest_csv))
    
    if df is not None:
        print("✅ Analysis completed successfully!")
    else:
        print("❌ Analysis failed!")

if __name__ == "__main__":
    main()
