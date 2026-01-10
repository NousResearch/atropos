# Code Agent Traces with Ollama

Pipeline for generating agent traces for code generation tasks using Ollama with logprobs support.

## Overview

This environment generates coding solutions using Ollama's native API with full logprobs tracking, suitable for training RL agents on code generation tasks.

The pipeline:
1. Load coding problems from dataset
2. Generate code solutions using Ollama with logprobs
3. Execute code in sandboxed environment
4. Score solutions based on test case results
5. Output agent traces with tokens, logprobs, and rewards

## Requirements

```bash
pip install aiohttp transformers datasets rich pydantic
```

For code execution, you need Modal setup:
```bash
pip install modal
modal token new
```

## Configuration

Set environment variables:

```bash
# For local Ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=deepseek-r1:7b

# For Ollama Cloud
export OLLAMA_BASE_URL=https://ollama.com
export OLLAMA_API_KEY=your_api_key_here
export OLLAMA_MODEL=deepseek-v3.1
```

## Usage

### Test the Pipeline

```bash
python test_ollama_pipeline.py
```

### Run the Environment

```bash
python agent_trace_env.py
```

### With Custom Configuration

```bash
python agent_trace_env.py \
    --temperature 0.7 \
    --group_size 8 \
    --max_code_tokens 4096
```

## Output Format

Agent traces are saved as JSON files:

```json
{
  "problem_idx": 0,
  "problem": "Write a function...",
  "problem_type": "func",
  "timestamp": "2024-01-10T12:00:00",
  "solutions": [
    {
      "index": 0,
      "score": 1.0,
      "code": "def solution(): ...",
      "content": "Here's my solution...",
      "finish_reason": "stop",
      "token_count": 256
    }
  ],
  "summary": {
    "total_solutions": 4,
    "correct_solutions": 3,
    "avg_token_count": 280
  }
}
```

## Ollama Logprobs API

This pipeline uses Ollama's native `/api/chat` endpoint for logprobs support.
The OpenAI-compatible endpoint (`/v1/chat/completions`) does not return logprobs.

Example native API call:
```python
from atroposlib.envs.server_handling.ollama_server import OllamaServer, OllamaServerConfig

config = OllamaServerConfig(
    base_url="http://localhost:11434",
    model_name="deepseek-r1:7b",
)

server = OllamaServer(config)

completion, logprobs = await server.chat_completion_with_logprobs(
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
    top_logprobs=5,
)

for token_info in logprobs[0]:
    print(f"Token: {token_info['token']}, logprob: {token_info['logprob']:.4f}")
```

## Supported Models

Cloud models (via ollama.com):
- `deepseek-v3.1` - 671B parameters
- `deepseek-v3.2` - High efficiency reasoning
- `gpt-oss:120b-cloud`

Local models:
- `deepseek-r1:7b`
- `qwen2.5-coder:7b`
- `codellama:13b`
- Any Ollama-compatible model
