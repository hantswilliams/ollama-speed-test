#!/usr/bin/env python3
"""
Enhanced visualization script for Ollama benchmark results.
This script demonstrates what visualizations could be created with matplotlib/seaborn.
"""

import pandas as pd
import sys
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_sample_charts(df):
    """Create sample charts showing what's possible with visualization libraries"""
    print("=== VISUALIZATION DEMONSTRATION ===")
    print()
    print("This script demonstrates what visualizations could be created with:")
    print("- matplotlib")
    print("- seaborn") 
    print("- plotly")
    print()
    print("Sample chart types that could be created:")
    print()
    
    # Show data structure
    print("1. Data Overview:")
    print(f"   Models: {df['model'].unique().tolist()}")
    print(f"   Categories: {df['category'].unique().tolist()}")
    print(f"   Records: {len(df)}")
    print()
    
    # Show performance metrics
    print("2. Key Performance Metrics:")
    performance_data = df.groupby('model')['avg_output_tps'].mean().sort_values(ascending=False)
    for i, (model, tps) in enumerate(performance_data.items()):
        print(f"   {i+1}. {model}: {tps:.1f} tokens/sec")
    print()
    
    # Show chart types that could be created
    print("3. Chart Types That Could Be Generated:")
    print("   - Bar chart: Average output tokens/sec by model and category")
    print("   - Line chart: Performance trends across different prompt types") 
    print("   - Scatter plot: TPS vs token count correlation")
    print("   - Heatmap: Performance matrix across models and categories")
    print("   - Box plot: Distribution of performance metrics")
    print("   - Dashboard: Interactive comparison of all metrics")
    print()
    
    print("4. Installation Required:")
    print("   pip install matplotlib seaborn plotly")
    print()
    
    # Show some summary statistics
    print("5. Summary Statistics:")
    print(df.describe().to_string())

def analyze_and_report(df):
    """Provide analysis and recommendations based on benchmark data"""
    print("=== BENCHMARK ANALYSIS REPORT ===")
    print()

    if "device_model" in df.columns and "chip" in df.columns:
        hosts = df["hostname"].dropna().unique() if "hostname" in df.columns else []
        if len(hosts) <= 1:
            row = df.iloc[0]
            print(f"Hardware: {row.get('device_model', '?')} | {row.get('chip', '?')} | "
                  f"{row.get('cpu_count', '?')} cores | {row.get('memory_gb', '?')} GB | "
                  f"{row.get('os', '?')} {row.get('os_version', '')}")
        else:
            print(f"Hosts in this dataset: {', '.join(hosts)}")
        print()
    
    # Performance by model
    print("Performance Ranking by Model (Avg Tokens/sec):")
    model_performance = df.groupby('model')['avg_output_tps'].mean().sort_values(ascending=False)
    for i, (model, tps) in enumerate(model_performance.items()):
        print(f"  {i+1}. {model}: {tps:.1f} tokens/sec")
    print()
    
    # Performance by category
    print("Performance by Category:")
    category_performance = df.groupby('category')['avg_output_tps'].mean().sort_values(ascending=False)
    for cat, tps in category_performance.items():
        print(f"  {cat}: {tps:.1f} tokens/sec")
    print()
    
    # Best model for each category
    print("Best Model per Category:")
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat]
        best_model = cat_data.loc[cat_data['avg_output_tps'].idxmax(), 'model']
        best_tps = cat_data['avg_output_tps'].max()
        print(f"  {cat}: {best_model} ({best_tps:.1f} tokens/sec)")

def generate_visualization_guide():
    """Generate guidance on how to create visualizations"""
    print("=== VISUALIZATION GUIDE ===")
    print()
    print("To create actual visualizations, follow these steps:")
    print()
    print("1. Install packages:")
    print("   pip install matplotlib seaborn plotly")
    print()
    print("2. Basic bar chart example:")
    print("   import matplotlib.pyplot as plt")
    print("   import seaborn as sns")
    print("   df = pd.read_csv('outputs/ollama_benchmark_summary.csv')")
    print("   sns.barplot(data=df, x='model', y='avg_output_tps', hue='category')")
    print("   plt.title('Model Performance Comparison')")
    print("   plt.xticks(rotation=45)")
    print("   plt.tight_layout()")
    print("   plt.savefig('performance_comparison.png')")
    print()
    print("3. Interactive dashboard example:")
    print("   import plotly.express as px")
    print("   fig = px.scatter(df, x='avg_output_tokens', y='avg_output_tps',")
    print("                  color='model', size='avg_total_duration_sec')")
    print("   fig.show()")

def main():
    """Main function to demonstrate visualization capabilities"""
    if len(sys.argv) < 2:
        print("Usage: python3 enhanced_visualize.py <csv_file>")
        print("Example: python3 enhanced_visualize.py outputs/ollama_benchmark_summary.csv")
        print()
        print("Using default sample data for demonstration...")
        # Create sample data for demonstration
        sample_data = {
            'model': ['qwen3-coder:30b-a3b-q4_K_M', 'qwen3.5:35b-a3b', 'qwen3-coder-next:q4_K_M'] * 2,
            'category': ['coding', 'coding', 'coding', 'general', 'general', 'general'],
            'avg_output_tps': [50.3, 26.7, 20.6, 50.3, 26.7, 20.6],
            'avg_output_tokens': [281.1, 300.0, 273.0, 300.0, 300.0, 300.0]
        }
        df = pd.DataFrame(sample_data)
        
    else:
        csv_file = sys.argv[1]
        if not os.path.exists(csv_file):
            print(f"Error: File {csv_file} not found")
            sys.exit(1)
        df = pd.read_csv(csv_file)
    
    create_sample_charts(df)
    print()
    analyze_and_report(df)
    print()
    generate_visualization_guide()

if __name__ == "__main__":
    main()