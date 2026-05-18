# Ollama Speed Test

This repository contains scripts for benchmarking Ollama models across different coding and general prompts.

## Repository Structure

- `test.py`: Main benchmarking script
- `outputs/`: Directory for storing benchmark results (CSV files)
- `scripts/`: Directory for accessory tools
  - `scripts/visualize.py`: Analysis script for benchmark results  
  - `scripts/enhanced_visualize.py`: Enhanced visualization and analysis tool

## How to Run Tests

```bash
# Run with default models
python3 test.py

# Run with custom models
python3 test.py --models "qwen3-coder:30b-a3b-q4_K_M" "gemma4:e4b"

# Run with suffix for output files
python3 test.py --suffix "mytest"
```

## Benchmark Results

The test runs three repetitions per prompt for each model and outputs:
1. Raw results (detailed metrics per run)
2. Summary statistics (averages, min/max values)

Results are saved in CSV format in the `outputs/` directory.

## Visualization Capabilities

While the core benchmark script generates data files, this repository is designed to support visual analysis of performance metrics.

### Analysis Tools

- `scripts/visualize.py`: Provides command-line analysis of benchmark results
- `scripts/enhanced_visualize.py`: Enhanced analysis with visualization guidance

### Potential Visualizations

The following visualizations can be created with additional packages (matplotlib, seaborn, plotly):

1. **Performance Comparison Bar Charts**
   - Average output tokens/sec per model (grouped by coding vs general)
   - Performance distribution across models

2. **Interactive Dashboard Components** 
   - Model comparison heatmaps
   - Performance metrics correlation plots
   - Duration breakdown charts

3. **Key Metrics Visualizations**
   - TPS vs token count scatter plots
   - Wall time vs model performance
   - Load time vs model performance

### To Create Visualizations

1. Install required packages:
```bash
pip install matplotlib seaborn plotly
```

2. Run the analysis:
```bash
python3 scripts/visualize.py outputs/ollama_benchmark_summary.csv
```

3. Use the visualization templates to create charts of your choice