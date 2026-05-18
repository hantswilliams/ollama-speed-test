#!/usr/bin/env python3
"""
Simple visualization script that creates charts from benchmark data
"""

import pandas as pd
import sys
import os
from pathlib import Path

def create_basic_charts(csv_file):
    """Create basic charts from benchmark data"""
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found")
        return
    
    df = pd.read_csv(csv_file)
    
    # Try to import visualization libraries
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("Creating visualizations...")
        sns.set_style("whitegrid")
        
        # Create a simple bar chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='model', y='avg_output_tps', hue='category')
        plt.title('Average Output Tokens/Second by Model and Category')
        plt.xlabel('Model')
        plt.ylabel('Average Output Tokens/Second')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the chart
        output_path = Path(csv_file).parent / f"performance_chart_{Path(csv_file).stem}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved chart: {output_path}")
        
        # Create another simple chart
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='avg_output_tokens', y='avg_output_tps', hue='model')
        plt.title('Performance vs Token Count')
        plt.xlabel('Average Output Tokens')
        plt.ylabel('Average Output Tokens/Second')
        plt.tight_layout()
        
        # Save the second chart
        scatter_path = Path(csv_file).parent / f"performance_scatter_{Path(csv_file).stem}.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter chart: {scatter_path}")
        
        print("Visualizations created successfully!")
        
    except ImportError:
        print("Visualization libraries not available. Install with: uv pip install matplotlib seaborn")
        print("Or run this script directly with installed packages.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 create_charts.py <csv_file>")
        print("Example: python3 create_charts.py outputs/ollama_benchmark_summary.csv")
        return
    
    csv_file = sys.argv[1]
    create_basic_charts(csv_file)

if __name__ == "__main__":
    main()