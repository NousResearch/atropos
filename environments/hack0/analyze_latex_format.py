#!/usr/bin/env python3
"""
Script to analyze LaTeX formats in the OpenMathReasoning dataset.

This script examines the expected_answer field in the OpenMathReasoning dataset
to identify and count LaTeX formats. It samples 1% of the dataset for analysis
and stores the results in a simple JSON file.
"""

import re
import json
import os
import random
import concurrent.futures
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from functools import partial

def extract_latex(text):
    """
    Extract all LaTeX code from the text.
    
    We look for common LaTeX patterns and delimiters:
    - $...$
    - $$...$$
    - \(...\)
    - \[...\]
    - \begin{...}...\end{...}
    - \command{...}
    """
    latex_patterns = [
        # Math delimiters
        (r'\$\$(.*?)\$\$', 'display_math'),
        (r'\$(.*?)\$', 'inline_math'),
        (r'\\[\(](.*?)\\[\)]', 'inline_math'),
        (r'\\[\[](.*?)\\[\]]', 'display_math'),
        
        # LaTeX environments
        (r'\\begin\{(.*?)\}(.*?)\\end\{\1\}', 'environment'),
        
        # Common LaTeX commands
        (r'\\boxed\{(.*?)\}', 'boxed'),
        (r'\\frac\{(.*?)\}\{(.*?)\}', 'fraction'),
        (r'\\sqrt\{(.*?)\}', 'square_root'),
        (r'\\mathbb\{(.*?)\}', 'mathbb'),
        (r'\\text\{(.*?)\}', 'text'),
    ]
    
    latex_formats = []
    
    for pattern, format_type in latex_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    # For patterns with multiple capture groups
                    latex_formats.append(f"{format_type}:{pattern}")
                else:
                    latex_formats.append(f"{format_type}:{pattern}")
    
    # If no specific patterns matched but has LaTeX commands, capture those
    if not latex_formats and re.search(r'\\[a-zA-Z]+', text):
        commands = re.findall(r'\\[a-zA-Z]+', text)
        for cmd in commands:
            latex_formats.append(f"command:{cmd}")
    
    return latex_formats

def process_batch(indices, dataset, batch_id):
    """Process a batch of dataset indices in parallel."""
    # Initialize counters for this batch
    no_latex_count = 0
    latex_format_counts = Counter()
    answers_without_latex = []
    answers_with_latex = {}
    
    for idx in indices:
        answer = dataset[idx]["expected_answer"]
        
        # Extract LaTeX formats
        latex_formats = extract_latex(answer)
        
        if not latex_formats:
            no_latex_count += 1
            # Save every alternate answer without LaTeX
            if no_latex_count % 2 == 1:
                answers_without_latex.append(answer)
        else:
            # Count each format
            for format_str in latex_formats:
                latex_format_counts[format_str] += 1
                
                # Save every alternate answer with this LaTeX format
                if format_str not in answers_with_latex:
                    answers_with_latex[format_str] = []
                
                # Check if we should save this answer (every alternate one)
                current_count = latex_format_counts[format_str]
                if current_count % 2 == 1:
                    if len(answers_with_latex[format_str]) < 5:  # Limit to 5 examples per format
                        answers_with_latex[format_str].append(answer)
    
    return {
        'no_latex_count': no_latex_count, 
        'latex_format_counts': latex_format_counts,
        'answers_without_latex': answers_without_latex,
        'answers_with_latex': answers_with_latex,
        'batch_id': batch_id
    }

def main():
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    output_file = "output/latex_format_counts.json"
    no_latex_output = "output/ans_without_latex.txt"
    with_latex_output = "output/ans_with_latex.json"
    
    print(f"Loading OpenMathReasoning dataset...")
    dataset = load_dataset("nvidia/OpenMathReasoning", split="cot")
    
    # Sample 1% of the dataset
    total_samples = len(dataset)
    sample_size = max(1, int(total_samples * 0.01))  # At least 1 sample
    
    print(f"Sampling {sample_size} out of {total_samples} samples (1%)...")
    sample_indices = random.sample(range(total_samples), sample_size)
    
    # Determine the number of cores to use for parallel processing
    num_cores = os.cpu_count()
    # Use 80% of available cores but at least 1
    num_workers = max(1, int(num_cores * 0.8))
    
    # Create smaller batches for more frequent progress updates
    # Use a fixed batch size to ensure frequent updates regardless of number of workers
    fixed_batch_size = 100  # Each batch will process 100 items
    batches = []
    for i in range(0, sample_size, fixed_batch_size):
        end = min(i + fixed_batch_size, sample_size)
        batches.append(sample_indices[i:end])
    
    print(f"Processing data using {num_workers} workers across {len(batches)} batches...")
    print(f"Each batch contains {fixed_batch_size} samples for more frequent progress updates")
    
    # Process batches in parallel
    results = []
    completed_batches = 0
    total_batches = len(batches)
    
    # Use a ProcessPoolExecutor with a progress bar
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            future_to_batch = {
                executor.submit(process_batch, batch, dataset, i): i 
                for i, batch in enumerate(batches)
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_batches += 1
                    # Update progress bar with percentage
                    pbar.update(1)
                    pbar.set_postfix({"progress": f"{completed_batches/total_batches*100:.1f}%"})
                except Exception as e:
                    print(f"\nBatch {batch_id} generated an exception: {e}")
    
    # Combine results from all batches
    no_latex_in_ans = sum(r['no_latex_count'] for r in results)
    
    # Combine all counters
    latex_format_counts = Counter()
    for r in results:
        latex_format_counts.update(r['latex_format_counts'])
    
    # Combine answers without LaTeX (taking a sample from each batch)
    answers_without_latex = []
    for r in results:
        answers_without_latex.extend(r['answers_without_latex'])
    
    # Combine answers with LaTeX, ensuring we don't exceed 5 examples per format
    answers_with_latex = {}
    for r in results:
        for format_str, answers in r['answers_with_latex'].items():
            if format_str not in answers_with_latex:
                answers_with_latex[format_str] = []
            
            remaining_slots = 5 - len(answers_with_latex[format_str])
            if remaining_slots > 0:
                answers_with_latex[format_str].extend(answers[:remaining_slots])
    
    # Prepare summary
    summary = {
        "summary": {
            "total_samples": sample_size,
            "no_latex_count": no_latex_in_ans,
            "with_latex_count": sample_size - no_latex_in_ans,
            "percent_without_latex": f"{no_latex_in_ans/sample_size*100:.1f}%",
            "percent_with_latex": f"{(sample_size - no_latex_in_ans)/sample_size*100:.1f}%",
            "num_workers_used": num_workers,
            "num_batches": len(batches),
            "batch_size": fixed_batch_size
        }
    }
    
    # Prepare results
    results = {
        **summary,
        "latex_format_counts": dict(latex_format_counts.most_common())
    }
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save answers without LaTeX
    with open(no_latex_output, "w") as f:
        for i, answer in enumerate(answers_without_latex):
            f.write(f"Answer #{i+1}:\n{answer}\n\n{'='*50}\n\n")
    
    # Save answers with LaTeX
    with open(with_latex_output, "w") as f:
        json.dump(answers_with_latex, f, indent=2)
    
    print(f"Analysis results saved to {output_file}")
    print(f"Answers without LaTeX saved to {no_latex_output}")
    print(f"Answers with LaTeX saved to {with_latex_output}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total answers analyzed: {sample_size}")
    print(f"Answers without LaTeX: {no_latex_in_ans} ({no_latex_in_ans/sample_size*100:.1f}%)")
    print(f"Answers with LaTeX: {sample_size - no_latex_in_ans} ({(sample_size - no_latex_in_ans)/sample_size*100:.1f}%)")
    print(f"Processed using {num_workers} parallel workers across {len(batches)} batches of {fixed_batch_size} samples each")
    
    print("\nTop 10 LaTeX formats:")
    for format_str, count in latex_format_counts.most_common(10):
        print(f"  {format_str}: {count} ({count/(sample_size - no_latex_in_ans)*100:.1f}% of answers with LaTeX)")

if __name__ == "__main__":
    main() 