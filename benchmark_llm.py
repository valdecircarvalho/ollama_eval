import logging
import json
import os
import time
import platform
import psutil
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_system_info():
    '''Collects system information (CPU, RAM, GPU).'''
    try:
        system_info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu": platform.processor(),
            "ram_total": psutil.virtual_memory().total / (1024.0 ** 3),  # in GB
            "gpu": get_gpu_info()
        }
        logging.info(f"System info: {system_info}")
        return system_info
    except Exception as e:
        logging.error(f"Error collecting system info: {e}")
        return {}

def get_gpu_info():
    '''Attempts to retrieve GPU information using nvidia-smi.'''
    try:
        # Run nvidia-smi command and capture output
        command = "nvidia-smi --query-gpu=name --format=csv,noheader"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        # Decode output and extract GPU name
        if output:
            gpu_name = output.decode("utf-8").strip()
            return gpu_name
        else:
            logging.warning(f"nvidia-smi returned an error: {error.decode('utf-8')}")
            return "N/A"
    except FileNotFoundError:
        logging.warning("nvidia-smi not found. NVIDIA drivers may not be installed.")
        return "N/A"
    except Exception as e:
        logging.error(f"Error getting GPU info: {e}")
        return "N/A"

def load_prompts(category=None):
    '''
    Loads prompts from JSON files in the benchmarks directory.
    If category is specified, loads prompts from that category only.
    Otherwise, loads prompts from all categories.
    '''
    prompts = {}
    
    # Define the categories to load
    categories = ["coding", "general_text", "summarization"]
    if category:
        categories = [category]
    
    # Load prompts from each category
    for cat in categories:
        prompts_file = f"benchmarks/{cat}.json"
        try:
            with open(prompts_file, 'r') as f:
                cat_prompts = json.load(f)
            logging.info(f"Loaded prompts from {prompts_file}")
            prompts[cat] = cat_prompts.get(cat, [])
        except FileNotFoundError:
            logging.error(f"Prompts file not found: {prompts_file}")
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {prompts_file}")
    
    return prompts

import requests

def list_models():
    '''Lists the models available in Ollama.'''
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            # Don't log the full model list
            logging.info(f"Successfully retrieved Ollama models")
            return models
        else:
            logging.error(f"Failed to list Ollama models. Status code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Failed to list Ollama models: {e}")
        return None

def format_duration(nanoseconds):
    '''Format duration in nanoseconds to a human-readable string.'''
    if nanoseconds < 1_000_000:  # Less than 1 millisecond
        return f"{nanoseconds} ns"
    elif nanoseconds < 1_000_000_000:  # Less than 1 second
        return f"{nanoseconds / 1_000_000:.2f} ms"
    elif nanoseconds < 60_000_000_000:  # Less than 1 minute
        return f"{nanoseconds / 1_000_000_000:.2f} s"
    else:  # 1 minute or more
        minutes = nanoseconds / 60_000_000_000
        return f"{minutes:.2f} min"

def run_benchmark(model, prompt):
    '''Runs a benchmark on a given model with a given prompt.'''
    start_time = time.time()
    try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False  # Explicitly set stream to False
            }
            logging.info(f"Sending request to Ollama API: {payload}")
            response = requests.post("http://localhost:11434/api/chat", json=payload)
            if response.status_code != 200:
                logging.error(f"Failed to chat with Ollama. Status code: {response.status_code}")
                return None, None, {}
            
            # Get the raw response text
            response_text = response.text
            logging.info(f"Raw response from Ollama API: {response_text[:200]}...")
            
            # The response is a stream of JSON objects, one per line
            # We need to parse the last line to get the final response
            lines = response_text.strip().split('\n')
            last_line = lines[-1]
            logging.info(f"Last line of response: {last_line}")
            
            # Parse the last line as JSON
            try:
                response_data = json.loads(last_line)
            except Exception as e:
                logging.error(f"Failed to parse JSON response: {e}")
                return None, None, {}
        except Exception as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            return None, None, {}
        
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Model: {model}, Prompt: {prompt}, Duration: {duration:.2f}s")
        
        # Extract the message content
        message_content = response_data.get('message', {}).get('content', '')
        
        # Extract Ollama statistics
        stats = {
            'total_duration': response_data.get('total_duration', 0),
            'load_duration': response_data.get('load_duration', 0),
            'prompt_eval_count': response_data.get('prompt_eval_count', 0),
            'prompt_eval_duration': response_data.get('prompt_eval_duration', 0),
            'eval_count': response_data.get('eval_count', 0),
            'eval_duration': response_data.get('eval_duration', 0),
        }
        
        # Calculate derived statistics
        if stats['prompt_eval_count'] > 0 and stats['prompt_eval_duration'] > 0:
            stats['prompt_eval_rate'] = stats['prompt_eval_count'] / (stats['prompt_eval_duration'] / 1_000_000_000)
        else:
            stats['prompt_eval_rate'] = 0
            
        if stats['eval_count'] > 0 and stats['eval_duration'] > 0:
            stats['eval_rate'] = stats['eval_count'] / (stats['eval_duration'] / 1_000_000_000)
        else:
            stats['eval_rate'] = 0
        
        # Format durations for human readability
        formatted_stats = {
            'total_duration': format_duration(stats['total_duration']),
            'load_duration': format_duration(stats['load_duration']),
            'prompt_eval_count': stats['prompt_eval_count'],
            'prompt_eval_duration': format_duration(stats['prompt_eval_duration']),
            'prompt_eval_rate': f"{stats['prompt_eval_rate']:.2f} tokens/s",
            'eval_count': stats['eval_count'],
            'eval_duration': format_duration(stats['eval_duration']),
            'eval_rate': f"{stats['eval_rate']:.2f} tokens/s",
        }
        
        logging.info(f"Response message: {message_content[:100]}...")
        logging.info(f"Ollama stats: {formatted_stats}")
        
        return message_content, duration, stats
    except Exception as e:
        logging.error(f"Error running benchmark: {e}")
        return None, None, {}

import csv

def save_result(result, system_info):
    '''Saves a single benchmark result to the CSV file.'''
    try:
        # Append to a single CSV file
        all_results_csv = "results/all_benchmarks.csv"
        os.makedirs(os.path.dirname(all_results_csv), exist_ok=True)
        
        file_exists = os.path.isfile(all_results_csv)
        
        fieldnames = [
            'model', 'category', 'prompt', 'response', 'duration',
            'total_duration', 'load_duration', 'prompt_eval_count',
            'prompt_eval_duration', 'prompt_eval_rate', 'eval_count',
            'eval_duration', 'eval_rate', 'timestamp', 
            'os', 'os_version', 'python_version', 'cpu', 'ram_total', 'gpu'
        ]
        
        with open(all_results_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            
            # Write header only if the file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format durations for CSV
            usage = result.get('usage', {})
            
            # Format durations for better readability
            total_duration_sec = usage.get('total_duration', 0) / 1_000_000_000
            load_duration_sec = usage.get('load_duration', 0) / 1_000_000_000
            prompt_eval_duration_sec = usage.get('prompt_eval_duration', 0) / 1_000_000_000
            eval_duration_sec = usage.get('eval_duration', 0) / 1_000_000_000
            
            # Comment out the actual prompt and response to avoid CSV formatting issues
            # prompt = result.get('prompt', '')
            # response = result.get('response', '')
            
            row = {
                'model': result.get('model', ''),
                'category': result.get('category', ''),
                'prompt': "prompt-placeholder",  # Replace with placeholder
                'response': "result-placeholder",  # Replace with placeholder
                'duration': f"{result.get('duration', 0):.2f}",
                'total_duration': f"{total_duration_sec:.2f}",
                'load_duration': f"{load_duration_sec:.2f}",
                'prompt_eval_count': usage.get('prompt_eval_count', 0),
                'prompt_eval_duration': f"{prompt_eval_duration_sec:.2f}",
                'prompt_eval_rate': f"{usage.get('prompt_eval_rate', 0):.2f}",
                'eval_count': usage.get('eval_count', 0),
                'eval_duration': f"{eval_duration_sec:.2f}",
                'eval_rate': f"{usage.get('eval_rate', 0):.2f}",
                'timestamp': timestamp,
                'os': system_info.get('os', ''),
                'os_version': system_info.get('os_version', ''),
                'python_version': system_info.get('python_version', ''),
                'cpu': system_info.get('cpu', ''),
                'ram_total': system_info.get('ram_total', 0),
                'gpu': system_info.get('gpu', '')
            }
            writer.writerow(row)
        logging.info(f"Appended result to {all_results_csv}")

    except Exception as e:
        logging.error(f"Error saving result: {e}")

import argparse

def main():
    '''Main function to orchestrate the benchmarking process.'''
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark LLM models using Ollama.')
    parser.add_argument('--models', nargs='+', default=["gemma3:4b"], help='List of models to benchmark (e.g., --models gemma3:4b llama3.2)')
    parser.add_argument('--category', help='Specific category to benchmark (coding, general_text, summarization)')
    args = parser.parse_args()

    logging.info("Starting LLM benchmarking...")

    # 1. Load system information
    system_info = get_system_info()

    # 2. List Ollama models
    list_models()

    # 3. Load prompts
    prompts = load_prompts(args.category)

    # 4. Select models to benchmark (from command line arguments)
    models = args.models
    logging.info(f"Benchmarking models: {models}")

    # 4. Run benchmarks for each model and prompt
    for model in models:
        for category, prompt_list in prompts.items():
            for prompt in prompt_list:
                logging.info(f"Running benchmark for model: {model}, category: {category}")
                response, duration, usage = run_benchmark(model, prompt)

                if response and duration:
                    result = {
                        "model": model,
                        "category": category,
                        "prompt": prompt,
                        "response": response,
                        "duration": duration,
                        "usage": usage
                    }
                    # Save the result immediately after each benchmark
                    save_result(result, system_info)

    logging.info("LLM benchmarking completed.")

if __name__ == "__main__":
    main()
