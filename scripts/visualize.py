#!/usr/bin/env python3
"""
Visualization script for Ollama benchmark results.
This script analyzes and displays performance metrics from the benchmark tests.
"""

import pandas as pd
import sys
import os

def load_and_analyze_data(file_path):
    """Load CSV data and perform basic analysis"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None
    
    df = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())

    if "device_model" in df.columns and "chip" in df.columns:
        row = df.iloc[0]
        print("\n=== Hardware ===")
        print(f"  Host:   {row.get('hostname', '?')}")
        print(f"  Device: {row.get('device_model', '?')} ({row.get('chip', '?')})")
        print(f"  CPU:    {row.get('cpu_count', '?')} cores")
        print(f"  Memory: {row.get('memory_gb', '?')} GB")
        print(f"  OS:     {row.get('os', '?')} {row.get('os_version', '')}")
    
    # Show basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Show models and categories
    print("\n=== Models ===")
    print(df['model'].unique().tolist())
    
    print("\n=== Categories ===")
    print(df['category'].unique().tolist())
    
    # Show avg output tokens per model
    print("\n=== Avg Output Tokens per Model ===")
    avg_tokens = df.groupby('model')['avg_output_tokens'].mean().sort_values(ascending=False)
    for model, tokens in avg_tokens.items():
        print(f"  {model}: {tokens:.1f}")
    
    # Show avg output tokens/sec per model
    print("\n=== Avg Output Tokens/sec per Model ===")
    avg_tps = df.groupby('model')['avg_output_tps'].mean().sort_values(ascending=False)
    for model, tps in avg_tps.items():
        print(f"  {model}: {tps:.1f}")
    
    return df

def analyze_by_category(df):
    """Analyze performance by category"""
    print("\n=== Performance by Category ===")
    
    # Group by category and show average TPS
    category_stats = df.groupby('category').agg({
        'avg_output_tps': ['mean', 'min', 'max'],
        'avg_total_duration_sec': 'mean'
    })
    
    print(category_stats)

def main():
    """Main function to run visual analysis"""
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <csv_file>")
        print("Example: python visualize.py outputs/ollama_benchmark_summary.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    df = load_and_analyze_data(csv_file)
    
    if df is not None:
        analyze_by_category(df)

if __name__ == "__main__":
    main()