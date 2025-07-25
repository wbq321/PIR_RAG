#!/usr/bin/env python3
"""
Data Preprocessing Script for PIR-RAG

This script preprocesses the corpus by splitting documents into size-based groups
for more efficient experimentation.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def analyze_document_sizes(corpus_texts):
    """Analyze the distribution of document sizes in the corpus."""
    byte_lengths = [len(text.encode('utf-8')) for text in corpus_texts]
    
    print("Document Size Distribution:")
    print("=" * 40)
    print(f"Total documents: {len(byte_lengths)}")
    print(f"Mean size: {np.mean(byte_lengths):.1f} bytes")
    print(f"Median size: {np.median(byte_lengths):.1f} bytes")
    print(f"Min size: {np.min(byte_lengths)} bytes")
    print(f"Max size: {np.max(byte_lengths)} bytes")
    print(f"Std dev: {np.std(byte_lengths):.1f} bytes")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(byte_lengths, p):.1f} bytes")
    
    return byte_lengths


def split_by_size_groups(corpus_df, embeddings, size_boundaries, output_dir):
    """
    Split corpus into size-based groups.
    
    Args:
        corpus_df: DataFrame with 'text' column
        embeddings: Corresponding embeddings array
        size_boundaries: List of size boundaries [1500, 2500, 3500, 4500]
        output_dir: Directory to save size groups
    """
    corpus_texts = corpus_df['text'].dropna().tolist()
    byte_lengths = [len(text.encode('utf-8')) for text in corpus_texts]
    
    # Define size groups
    size_groups = {
        'small': [],           # < 1500
        'medium_small': [],    # 1500-2500
        'medium': [],          # 2500-3500
        'medium_large': [],    # 3500-4500
        'large': []            # > 4500
    }
    
    group_names = ['small', 'medium_small', 'medium', 'medium_large', 'large']
    boundaries = [0] + size_boundaries + [float('inf')]
    
    # Classify documents into groups
    for i, (text, byte_len) in enumerate(zip(corpus_texts, byte_lengths)):
        # Find which group this document belongs to
        for j in range(len(boundaries) - 1):
            if boundaries[j] <= byte_len < boundaries[j + 1]:
                group_name = group_names[j]
                size_groups[group_name].append({
                    'original_index': i,
                    'text': text,
                    'byte_length': byte_len,
                    'embedding_index': i
                })
                break
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each group
    group_stats = {}
    for group_name, documents in size_groups.items():
        if documents:
            # Create DataFrame for this group
            group_df = pd.DataFrame(documents)
            
            # Save group data
            group_file = f"{group_name}_{'_'.join(map(str, [boundaries[group_names.index(group_name)], boundaries[group_names.index(group_name) + 1]]))}.csv"
            group_file = group_file.replace('_0_', '_lt').replace('_inf', '_gt4500')
            group_path = os.path.join(output_dir, group_file)
            
            group_df.to_csv(group_path, index=False)
            
            # Extract corresponding embeddings
            embedding_indices = group_df['embedding_index'].tolist()
            group_embeddings = embeddings[embedding_indices]
            
            # Save embeddings
            embedding_file = group_file.replace('.csv', '_embeddings.npy')
            embedding_path = os.path.join(output_dir, embedding_file)
            np.save(embedding_path, group_embeddings)
            
            # Collect statistics
            group_stats[group_name] = {
                'count': len(documents),
                'size_range': f"{boundaries[group_names.index(group_name)]}-{boundaries[group_names.index(group_name) + 1]}",
                'avg_size': np.mean([doc['byte_length'] for doc in documents]),
                'file': group_file
            }
            
            print(f"Saved {len(documents)} documents to {group_path}")
            print(f"Saved embeddings to {embedding_path}")
        else:
            group_stats[group_name] = {'count': 0, 'size_range': 'N/A'}
    
    # Print summary
    print("\nSize Group Summary:")
    print("=" * 50)
    for group_name, stats in group_stats.items():
        if stats['count'] > 0:
            print(f"{group_name:12}: {stats['count']:4d} docs, avg {stats['avg_size']:6.1f} bytes, range {stats['size_range']}")
        else:
            print(f"{group_name:12}: {stats['count']:4d} docs")
    
    return group_stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess corpus into size-based groups")
    parser.add_argument("--corpus_path", type=str, required=True,
                        help="Path to corpus CSV file")
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to embeddings NPY file")
    parser.add_argument("--output_dir", type=str, default="./data/size_groups",
                        help="Output directory for size groups")
    parser.add_argument("--size_boundaries", nargs='+', type=int, 
                        default=[1500, 2500, 3500, 4500],
                        help="Size boundaries in bytes")
    parser.add_argument("--analyze_only", action='store_true',
                        help="Only analyze size distribution, don't split")
    
    args = parser.parse_args()
    
    print("Loading data...")
    corpus_df = pd.read_csv(args.corpus_path)
    embeddings = np.load(args.embeddings_path)
    
    print(f"Loaded {len(corpus_df)} documents and {len(embeddings)} embeddings")
    
    if len(corpus_df) != len(embeddings):
        raise ValueError("Corpus and embeddings length mismatch")
    
    # Analyze document sizes
    corpus_texts = corpus_df['text'].dropna().tolist()
    byte_lengths = analyze_document_sizes(corpus_texts)
    
    if not args.analyze_only:
        print(f"\nSplitting corpus using boundaries: {args.size_boundaries}")
        group_stats = split_by_size_groups(
            corpus_df, embeddings, args.size_boundaries, args.output_dir
        )
        
        # Save group statistics
        stats_file = os.path.join(args.output_dir, "group_statistics.txt")
        with open(stats_file, 'w') as f:
            f.write("Size Group Statistics\n")
            f.write("=" * 30 + "\n")
            for group_name, stats in group_stats.items():
                if stats['count'] > 0:
                    f.write(f"{group_name}: {stats['count']} docs, avg {stats['avg_size']:.1f} bytes\n")
                else:
                    f.write(f"{group_name}: {stats['count']} docs\n")
        
        print(f"\nSize group preprocessing complete!")
        print(f"Groups saved to: {args.output_dir}")
        print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
