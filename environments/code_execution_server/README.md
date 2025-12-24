# Code Execution Environment

A comprehensive environment for training language models to solve coding problems through **code generation and execution**. This environment evaluates models on their ability to generate correct Python code that passes test cases using a Modal endpoint to validate LLM-generated code.

## 🎯 Overview

The Code Execution Environment evaluates models on:
- Generating correct Python code solutions for coding problems
- Passing test cases through actual code execution
- Handling both function-based and stdin/stdout problem types
- Supporting multiple datasets (RLVR, DeepMind Code Contests, LCB Test)
- Providing comprehensive evaluation metrics (pass@1, pass@k, difficulty breakdowns)

**Key Philosophy**: This environment scores based on **code correctness** (1.0 for passing all tests, -1.0 for failing). Models must generate syntactically valid, executable Python code that produces the correct outputs for given test cases.

## ✨ Key Features

### 🔧 **Code Execution & Testing**
- Real code execution via Modal endpoint (`lcb_modal_endpoint.py`)
- Support for function-based problems (LeetCode-style)
- Support for stdin/stdout problems (competitive programming style)
- Automatic test case validation
- Error handling (timeouts, runtime errors, wrong answers)

### 📊 **Comprehensive Evaluation**
- Pass@1 metric calculation using combinatorial estimation
- Pass@group_size evaluation
- Difficulty-level breakdowns (easy/medium/hard)
- Completion length tracking (correct vs incorrect solutions)
- Overlong ratio tracking (solutions exceeding token limits)

### 📚 **Dataset Support**
- **RLVR_Coding_Problems**: Training dataset for reinforcement learning
- **DeepMind Code Contests**: Alternative training dataset
- **LCB Test**: Evaluation benchmark from LiveCodeBench
- Automatic dataset format handling and conversion

### ⚙️ **Safety & Reliability**
- Sandboxed code execution via Modal
- Timeout protection
- Resource limits (memory, CPU)
- Reliability guards preventing destructive operations
- Segmentation fault handling

## 🚀 Quick Start

### Basic Configuration

```python
from atropos.environments.code_execution_server import CodingEnv, CodeConfig

# Configuration
config = CodeConfig(
    dataset_name="normal",  # or "deepmind"
    temperature=1.0,
    eval_temperature=0.6,
    top_p=1.0,
    eval_top_p=0.95,
    start_idx=0,
    max_eval_token_length=40960,
)

# Initialize environment
env = CodingEnv(config, server_configs)
```

### Problem Types

#### **Function-Based Problems**
Problems where code defines a function that is called with test inputs:
```python
# Problem specification includes function signature
def solve(nums: List[int]) -> int:
    # Model generates function body
    return max(nums)

# Tests call the function directly
tests = {
    "fn_name": "solve",
    "input": [[1, 2, 3], [5, 4, 3]],
    "output": [3, 5]
}
```

#### **Standard Input/Output Problems**
Problems where code reads from stdin and writes to stdout:
```python
# Problem: Read integers, output their sum
# Model generates:
a = int(input())
b = int(input())
print(a + b)

# Tests provide stdin inputs and expected stdout outputs
tests = {
    "fn_name": "none",
    "input": ["5\n3", "10\n20"],
    "output": ["8", "30"]
}
```

## 📊 Evaluation Metrics

### **Pass@1 (Estimated)**
Uses combinatorial estimation to calculate the probability of at least one correct solution in a single attempt:
```
pass@1 = mean(1 - C(n-c, 1) / C(n, 1)) across all problems
where n = group_size, c = number of correct solutions
```

### **Pass@group_size**
Fraction of problems where at least one solution is correct:
```
pass@group_size = mean(num_correct > 0)
```

### **Difficulty Breakdowns**
Separate metrics for easy, medium, and hard problems:
- `eval/easy_pass_1`
- `eval/medium_pass_1`
- `eval/hard_pass_1`

### **Completion Analysis**
- `eval/completion_length`: Average completion length
- `eval/correct_completion_length`: Average length of correct solutions
- `eval/incorrect_completion_length`: Average length of incorrect solutions
- `eval/overlong_ratio`: Fraction of solutions exceeding token limits

### **Training Metrics**
- `train_rewards/rewards`: Average reward (correctness rate)
- `train_rewards/pass@group_size`: Training pass rate
- `train_rewards/overlong_ratio`: Training overlong ratio
- `train/completion_lengths`: Training completion statistics


## 🔍 Code Extraction & Scoring

### **Code Extraction**
The environment extracts Python code from model responses using regex:
- Looks for code blocks in markdown format: ` ```python ... ``` `
- Takes the last code block if multiple are present
- Returns `None` if no code block is found (scores -1.0)

### **Scoring Logic**
```python
if code is None:
    score = -1.0  # No code extracted
elif all_tests_pass(code, test_cases):
    score = 1.0   # All tests pass
else:
    score = -1.0  # Tests fail or error occurs
```

### **Test Execution**
Code is executed via Modal endpoint with:
- Timeout protection (default 15 seconds per test case)
- Memory limits (5GB)
- Reliability guards (disables file system, network, process operations)
- Error categorization (Wrong Answer, Time Limit Exceeded, Runtime Error)

## 🛠️ Advanced Features

### **Offline Filtering**
Utility to identify problems where the model achieves perfect scores:
```python
await env.offline_filter()
# Saves perfect problem indices to perfect_indices.txt
```

### **Blacklist System**
Problems in `perfect_indices.txt` are automatically blacklisted during training to focus on harder problems.

### **Data Logging**
Comprehensive logging to files:
- **Short logs**: `qwen_data_dump_{timestamp}.txt` - Basic stats per problem
- **Long logs**: `qwen_data_dump_long_{timestamp}.txt` - Includes code, errors, full outputs
- Separate directories: `train_logs/` and `eval_logs/`

### **Example Log Entry**
```json
{
    "cur_id": 42,
    "num_correct": 5,
    "total": 8,
    "scores": [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
    "lengths": [1200, 1180, 1220, 1190, 1210, 800, 750, 820],
    "errors": [...],
    "codes": [...],
    "gen": "assistant message content"
}
```


## 📝 Response Format Requirements

### **Expected Format**
Model responses should contain Python code in markdown code blocks:
````
Here's my solution:

```python
def solve(nums):
    return max(nums)
```
````

## 🔐 Code Execution Endpoint (`lcb_modal_endpoint.py`)

The Modal endpoint handles safe execution of generated Python code with comprehensive sandboxing and test validation. Most of the code is adapted from the [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) and [rllm](https://github.com/rllm-org/rllm) repositories.

#### **Problem Type Support**
- **Call-Based**: Function calls with JSON-serialized inputs/outputs (LeetCode-style)
- **Standard I/O**: stdin/stdout problems (competitive programming style)

The code execution endpoint (`lcb_modal_endpoint.py`) implements extensive safety measures:

### **Reliability Guards**
- Disables file system operations (`os.remove`, `os.chdir`, etc.)
- Blocks process operations (`os.fork`, `subprocess.Popen`)
- Prevents network access
- Limits memory usage (5GB default)
- Sets recursion limit to prevent stack overflow

### **Error Handling**
- Timeout exceptions (SIGALRM)
- Segmentation fault detection (faulthandler)
- Runtime error capture
- Wrong answer detection with detailed comparison

### **Execution Environment**
- Isolated Modal containers
- Signal-based timeouts
- Resource limits (CPU, memory)
- Sandboxed imports (whitelisted standard library modules)


## 📄 License

This environment is part of the Atropos training framework. See the main repository for license information.

