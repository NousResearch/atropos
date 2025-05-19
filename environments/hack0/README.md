# Open Math Reasoning Environment

## Overview
The Open Math Reasoning Environment is designed to train and evaluate LLMs on advanced mathematical reasoning tasks. This environment leverages the NVIDIA OpenMathReasoning dataset to provide a robust training platform for mathematical problem-solving capabilities. The "hack0" directory contains experimental implementations and utilities for this environment, with the name indicating a development/experimental area where new approaches for mathematical reasoning are being tested.

## Features
- Integration with NVIDIA's OpenMathReasoning dataset
- Support for LaTeX parsing and verification of mathematical expressions
- Configurable system prompts to encourage deep reasoning chains
- Specialized evaluation metrics for mathematical accuracy
- Thinking-tag based formatting and evaluation

## Components

### Server Implementation
- `open_math_reasoning_server.py` - The main server implementation that handles interactions with the OpenMathReasoning dataset, processes model responses, and evaluates mathematical reasoning capabilities.

### Artifacts
- `data/` 

### Utilities
- `analyze_latex_format.py` - A utility script for analyzing LaTeX formatting in model outputs. This script:
  - Examines LaTeX usage in the OpenMathReasoning dataset
  - Extracts and classifies different LaTeX patterns
  - Generates statistics on LaTeX usage across the dataset
  - Outputs analysis files to help with prompt engineering

## Setup and Configuration

### Prerequisites
- Python 3.10+

### Configuration
The environment can be configured through a YAML file located at:
```
environments/dataset_environment/configs/open_math_reasoning.yaml
```

Important configuration options include:
- `group_size`: Number of generations to sample per problem (default: 8)
- `max_token_length`: Maximum token length for model outputs (default: 16384)
- `batch_size`: Number of problems to process in parallel (default: 12)
- `reward_functions`: Configurable reward functions with weights

## Usage

### Running the Server
To run the Open Math Reasoning server:

```bash
# Start the API server
run-api

# In a separate terminal, start the environment
python environments/hack0/open_math_reasoning_server.py serve --openai.model_name <your-model-name>
```

### LaTeX Format Analysis
To analyze LaTeX formats in the dataset:

```bash
python environments/hack0/analyze_latex_format.py
```

This will generate several output files in the `output/` directory:
- `latex_format_counts.json` - Statistics on LaTeX format usage
- `ans_without_latex.txt` - Examples of answers without LaTeX formatting
- `ans_with_latex.json` - Examples of answers with various LaTeX formats

## Implementation Details

The implementation focuses on several key aspects:

### System Prompt
```
You are a deep thinking AI specializing in advanced mathematics. 
You may use extremely long chains of thought to deeply consider mathematical problems and deliberate with yourself via systematic 
reasoning processes to help come to a correct solution prior to answering. 
You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem.

You are allocated a maximum of 2048 tokens, please strive to use less.

You should use proper LaTeX notation when writing mathematical expressions and formulas.
For example, use \frac{a}{b} for fractions, \sqrt{x} for square roots, and ^ for exponents.

You will then provide your final answer like this: \boxed{your answer here}
It is important that you provide your answer in the correct LaTeX format.
If you do not, you will not receive credit for your answer.
So please end your answer with \boxed{your answer here}
```

### Input Format
- Mathematical problems from the NVIDIA OpenMathReasoning dataset
- Each item contains:
  - `problem`: The mathematical problem
  - `expected_answer`: The correct answer, typically in LaTeX format

### Reward Function
- Score of 1.0 if the model's answer matches the ground truth (using specialized LaTeX verification)
- Score of 0.0 if incorrect or if LaTeX cannot be parsed properly
- Includes support for various LaTeX formats and mathematical notation verification

### Key Technical Components
1. **LaTeX Handling**: Special handling for mathematical expressions using LaTeX2Sympy for parsing and verification.

2. **Thinking Tags**: The server uses `<think>` tags to separate the model's reasoning process from its final answers.

3. **Answer Extraction**: The server extracts answers from model responses, focusing on boxed content as per mathematical convention.

4. **Trajectory Collection**: The server has been updated to properly collect and process trajectories for reinforcement learning.