# Ollama LLM Model Evaluation & Benchmarking Tool

A tool for benchmarking LLM models available in Ollama, comparing their performance across different tasks.

## Features

- Benchmark multiple LLM models available in Ollama
- Test models on different categories of tasks (coding, general text, summarization)
- Measure response time and resource usage
- Capture detailed Ollama statistics (total duration, load duration, eval count, etc.)
- Save all results in a single CSV file for easy analysis

## Requirements

- Python 3.6+
- Ollama installed and running
- Required Python packages:
  - requests
  - psutil

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ollama-eval.git
   cd ollama-eval
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run a benchmark with default settings (gemma3:4b model, all categories):

```
python benchmark_llm.py
```

### Command-line Options

- `--models`: Specify which models to benchmark (space-separated list)
  ```
  python benchmark_llm.py --models gemma3:4b llama3.2
  ```

- `--category`: Benchmark only a specific category (coding, general_text, summarization)
  ```
  python benchmark_llm.py --category coding
  ```

### Examples

Benchmark multiple models on coding tasks:
```
python benchmark_llm.py --models gemma3:4b llama3.2 --category coding
```

Benchmark a single model on all tasks:
```
python benchmark_llm.py --models mistral
```

## Prompt Categories

The benchmark includes prompts from three categories:

1. **Coding**: Programming-related tasks
2. **General Text**: General knowledge and creative writing tasks
3. **Summarization**: Text summarization tasks

You can add or modify prompts by editing the JSON files in the `benchmarks` directory.

## Results

All benchmark results are appended to a single CSV file (`results/all_benchmarks.csv`) for easy analysis and comparison. This file includes:

- Model information
- Prompt and response
- Timing data
- Ollama statistics
- System information

This approach makes it easy to analyze and compare results over time using spreadsheet software or data analysis tools.

### Captured Metrics

The benchmark captures the following metrics for each model and prompt:

- **Duration**: Total time taken for the request (in seconds)
- **Ollama Statistics**:
  - **total_duration**: Total time taken by Ollama to process the request (in nanoseconds, converted to seconds in CSV)
  - **load_duration**: Time taken to load the model (in nanoseconds, converted to seconds in CSV)
  - **prompt_eval_count**: Number of tokens in the prompt
  - **prompt_eval_duration**: Time taken to evaluate the prompt (in nanoseconds, converted to seconds in CSV)
  - **prompt_eval_rate**: Rate of prompt evaluation (tokens per second)
  - **eval_count**: Number of tokens in the response
  - **eval_duration**: Time taken to generate the response (in nanoseconds, converted to seconds in CSV)
  - **eval_rate**: Rate of response generation (tokens per second)
- **System Information**: Details about the hardware used for the benchmark (CPU, RAM, GPU)

### Analyzing Results

The CSV format makes it easy to analyze results using spreadsheet software or data analysis tools. You can:

- Filter results by model, category, or date
- Calculate average performance metrics across different prompts
- Create charts and visualizations to compare model performance
- Export subsets of data for further analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
