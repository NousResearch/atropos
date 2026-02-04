# ManagedServer: Automatic Token and Logprob Tracking

## Overview

`ManagedServer` is a wrapper around `APIServer` that automatically tracks text sequences with aligned tokens and logprobs. It eliminates the need for manual token extraction, alignment, and masking in your environment code, making it **the recommended approach** for handling inference in Atropos environments.

**Server Compatibility:** ManagedServer works with `VLLMServer`, `SGLangServer`, and `TrlVllmServer`. Simply set the `server_type` field in your `APIServerConfig` to `"vllm"`, `"sglang"`, or `"trl"` to use the appropriate backend with automatic server class selection.

> **⚠️ OpenAI Endpoints:** OpenAI's API does not expose token IDs or detailed logprobs required for full ManagedServer functionality. See [OpenAI Endpoint Limitations](#openai-endpoint-limitations) for details and workarounds.

### Why Use ManagedServer?

**Before ManagedServer** (manual approach):
```python
# Manual token extraction
response = await self.server.completion(prompt=prompt, n=8)
# Manually tokenize and align
tokens = self.tokenizer.encode(prompt + response.text)
# Manually apply masking
prompt_len = len(self.tokenizer.encode(prompt))
masked_tokens = [-100] * prompt_len + tokens[prompt_len:]
# Manually extract and align logprobs
logprobs = extract_logprobs_somehow(response)
```

**With ManagedServer** (automatic):
```python
async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
    response = await managed.completion(prompt=prompt, n=8)
    state = managed.get_state()
    nodes = state["nodes"]
    # tokens, masked_tokens, and logprobs are already aligned and ready!
```

### Key Benefits

- ✅ **Automatic Tokenization**: No need to manually tokenize prompts and completions
- ✅ **Automatic Masking**: Prompt tokens automatically masked with -100, logprobs with 1.0
- ✅ **Perfect Alignment**: Tokens and logprobs are guaranteed to align correctly
- ✅ **Multi-turn Support**: Automatically handles conversation extensions
- ✅ **Branching Support**: Handles n>1 completions naturally
- ✅ **Clean API**: Simple context manager pattern
- ✅ **Less Error-Prone**: Eliminates common token alignment bugs

## Core Concepts

### SequenceNode Structure

Each completion tracked by ManagedServer is stored as a `SequenceNode`:

```python
class SequenceNode(BaseModel):
    full_text: str                    # Complete text (prompt + completion)
    tokens: List[int]                 # Full token sequence (unmasked)
    masked_tokens: List[int]          # Tokens for training (-100 for prompt, actual IDs for completion)
    logprobs: List[float]             # Logprobs for training (1.0 for prompt, actual values for completion)
    metadata: Optional[Dict[str, Any]]  # Contains finish_reason, etc.
```

### Masking Methodology

ManagedServer applies automatic masking to distinguish between prompt and completion:

| Field | Masked Positions | Completion Positions  | Purpose                        |
|-------|------------------|-----------------------|--------------------------------|
| `tokens` | Actual token IDs | Actual token IDs      | Full unmasked sequence         |
| `masked_tokens` | **-100**         | Actual token IDs      | Training input (mask prompts)  |
| `logprobs` | **1.0**          | Actual logprob values | Training target (mask prompts) |

**Why 1.0 for masked logprobs?**

The value 1.0 is used to indicate "obviously bad" logprobs for prompt positions:
- `e^1.0 ≈ 2.718`, which would represent a probability > 1.0 (invalid)
- This makes masked positions easy to identify and filter during training
- Trainers should ignore positions where `logprobs > 0.0` or where `masked_tokens == -100`

**Example:**

```python
# Prompt: "What is 2+2?"
# Completion: " 4"
# Tokenized: [1, 1867, 374, 220, 17, 10, 17, 30] + [220, 19]

node.tokens = [1, 1867, 374, 220, 17, 10, 17, 30, 220, 19]
node.masked_tokens = [-100, -100, -100, -100, -100, -100, -100, -100, 220, 19]
node.logprobs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.342, -0.156]
```

### Two Operating Modes

ManagedServer supports two modes for tracking sequences:

#### 1. Default Mode (track_tree=False)
- Maintains a simple list of current nodes
- When a new prompt **extends** an existing node's `full_text`, it **replaces** that node
- Best for most RL scenarios (GRPO, DPO, etc.)
- Accessed via `state["nodes"]`

```python
async with server.managed_server(tokenizer=tokenizer) as managed:
    # First completion
    await managed.completion(prompt="Hello", n=1)
    state = managed.get_state()
    len(state["nodes"])  # → 1

    # Extension (prompt starts with previous full_text)
    await managed.completion(prompt="Hello World", n=1)
    state = managed.get_state()
    len(state["nodes"])  # → 1 (replaced, not added)
```

#### 2. Tree Mode (track_tree=True)
- Maintains a dictionary of nodes keyed by `full_text`
- Every unique `full_text` creates a new entry
- Useful for multi-turn RL with per-step advantages
- Accessed via `state["sequences"]` or `state["tree"]`

```python
managed = ManagedServer(server, tokenizer=tokenizer, track_tree=True)
```

## Usage Patterns

### Pattern 1: Basic Single-Turn (Completion API)

Use with completion-style prompts (like math_server_zero.py):

```python
async def collect_trajectories(self, item):
    prompt = format_prompt(item)

    # Use managed server context
    async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
        completion = await managed.completion(
            prompt=prompt,
            n=self.config.group_size,  # e.g., 16
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
        )

        # Get tracked sequences
        state = managed.get_state()
        nodes = state["nodes"]

    # Process nodes for training
    to_score = []
    for choice, node in zip(completion.choices, nodes):
        to_score.append({
            "full_text": node.full_text,
            "tokens": node.tokens,
            "masked_tokens": node.masked_tokens,
            "logprobs": node.logprobs,
            "finish_reason": node.metadata["finish_reason"],
        })

    return await self.score(to_score)
```

### Pattern 2: Basic Single-Turn (Chat Completion API)

Use with chat messages (like math_server.py):

```python
async def collect_trajectories(self, item):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": item["question"]},
    ]

    # Use managed server context
    async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
        chat_completion = await managed.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=4096,
            temperature=1.0,
            top_p=0.95,
        )

        # Get tracked sequences
        state = managed.get_state()
        nodes = state["nodes"]

    # Process nodes
    to_score = []
    for choice, node in zip(chat_completion.choices, nodes):
        to_score.append({
            "content": choice.message.content,
            "tokens": node.tokens,
            "masked_tokens": node.masked_tokens,
            "logprobs": node.logprobs,
            "finish_reason": choice.finish_reason,
        })

    return await self.score(to_score)
```

### Pattern 3: Multi-Turn Conversations

ManagedServer automatically detects when a prompt extends a previous sequence:

```python
# Turn 1
async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
    await managed.completion(prompt="Hello", n=1)
    state = managed.get_state()
    # nodes[0].full_text = "Hello World"

    # Turn 2: Extends turn 1
    # This prompt starts with "Hello World" (turn 1's full_text)
    await managed.completion(prompt="Hello World! How are you?", n=1)
    state = managed.get_state()
    # nodes[0].full_text = "Hello World! How are you? I'm great!"
    # The node from turn 1 has been replaced with the extended version
```

**How Extension Detection Works:**
1. ManagedServer checks if the new prompt starts with any existing node's `full_text`
2. If yes, it reuses those tokens and only tokenizes the new suffix
3. The extended node replaces the original in the list

### Pattern 4: Multiple Contexts in One Method

You can use multiple managed_server contexts for complex workflows:

```python
async def collect_trajectories_rlaif(self, item):
    # First set of completions
    async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
        completions_fwd = await managed.chat_completion(
            messages=messages_fwd,
            n=3,
            temperature=1.0,
        )
        state_fwd = managed.get_state()

    # Second set of completions (independent context)
    async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
        completions_bwd = await managed.chat_completion(
            messages=messages_bwd,
            n=3,
            temperature=1.0,
        )
        state_bwd = managed.get_state()

    # Process both sets
    nodes_fwd = state_fwd["nodes"]
    nodes_bwd = state_bwd["nodes"]
```

### Pattern 5: Passing Tokens Through Backlog

For complex multi-step workflows, you can pass pre-computed tokens/masks/logprobs through backlog items:

```python
async def collect_trajectories_normal(self, item):
    # Generate initial completions
    async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
        response = await managed.chat_completion(messages=chat, n=16)
        state = managed.get_state()
        nodes = state["nodes"]

    # Find interesting pairs for RLAIF
    if should_do_rlaif:
        # Pass tokens/masks/logprobs to next stage
        backlog_item = (
            item["problem"],
            item["answer"],
            "rlaif",  # Type marker
            messages_1,
            messages_2,
            # Pre-computed data from managed_server
            nodes[idx1].tokens,          # Solution 1 tokens
            nodes[idx1].masked_tokens,   # Solution 1 masks
            nodes[idx1].logprobs,        # Solution 1 logprobs
            nodes[idx2].tokens,          # Solution 2 tokens
            nodes[idx2].masked_tokens,   # Solution 2 masks
            nodes[idx2].logprobs,        # Solution 2 logprobs
        )
        return None, [backlog_item]

async def collect_trajectories_rlaif(self, item):
    # Extract pre-computed data
    tokens_1 = item[5]
    masks_1 = item[6]
    logprobs_1 = item[7]
    tokens_2 = item[8]
    masks_2 = item[9]
    logprobs_2 = item[10]

    # Do RLAIF judgment...
    # Use pre-computed tokens/masks/logprobs directly
    return {
        "tokens": [tokens_1, tokens_2],
        "masks": [masks_1, masks_2],
        "inference_logprobs": [logprobs_1, logprobs_2],
        "scores": [score_1, score_2],
    }
```

## Complete Examples

### Example 1: Completion API (math_server_zero.py)

```python
async def collect_trajectories(self, item) -> Tuple[List, List]:
    # Format prompt
    user_prompt = prompt_format.format(
        prompt=problem_format.format(problem=item[0])
    )

    # Calculate max tokens
    thinking_len = self.config.max_token_length - len(
        self.tokenizer.encode(user_prompt)
    )

    # Use managed server for automatic token/logprob tracking
    async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
        completion = await managed.completion(
            prompt=user_prompt,
            n=self.config.group_size,
            max_tokens=thinking_len,
            temperature=1.0,
            top_p=1.0,
            stop=stop_list,
        )

        # Get tracked sequences with aligned tokens and logprobs
        state = managed.get_state()
        nodes = state["nodes"]

    # Extract data from SequenceNodes for scoring
    to_score = []
    for choice, node in zip(completion.choices, nodes):
        to_score.append((
            node.full_text,              # Complete text (prompt + completion)
            item[1],                     # Answer
            choice.finish_reason,        # Finish reason
            node.tokens,                 # All tokens (prompt + completion)
            node.masked_tokens,          # Masked tokens (-100 for prompt, IDs for completion)
            node.logprobs,               # Logprobs (1.0 for prompt, actual for completion)
        ))

    # Score and return
    to_postprocess = await self.score(to_score)
    return to_postprocess, []
```

### Example 2: Chat Completion API (math_server.py)

```python
async def collect_trajectories_normal(self, item) -> Tuple[List, List]:
    # Prepare chat messages
    user_prompt = problem_format.format(problem=item[0])
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Calculate max tokens
    thinking_len = self.config.max_token_length - len(
        self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    )

    # Use managed server for automatic token/logprob tracking
    async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
        chat_completions = await managed.chat_completion(
            messages=chat,
            n=self.config.group_size,
            max_tokens=thinking_len,
            temperature=1.0,
            top_p=0.95,
        )

        # Get tracked sequences with aligned tokens and logprobs
        state = managed.get_state()
        nodes = state["nodes"]

    # Extract data from SequenceNodes for scoring
    to_score = []
    for chat_completion, node in zip(chat_completions.choices, nodes):
        messages = (
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": chat_completion.message.content},
        )
        to_score.append((
            messages,                    # Full conversation
            item[1],                     # Answer
            chat_completion.finish_reason,  # Finish reason
            node.tokens,                 # All tokens
            node.masked_tokens,          # Masked tokens
            node.logprobs,               # Logprobs
        ))

    # Score and return
    to_postprocess = await self.score_normal(to_score)
    return to_postprocess, []
```

### Example 3: RLAIF with Multiple Contexts (math_server.py)

```python
async def collect_trajectories_rlaif(self, item) -> Tuple[List, List]:
    # Prepare forward and backward prompts
    user_prompt_fwd = rlaif_format.format(
        problem=item[0],
        solution1=solution1_text,
        solution2=solution2_text,
    )
    user_prompt_bwd = rlaif_format.format(
        problem=item[0],
        solution1=solution2_text,  # Swapped
        solution2=solution1_text,  # Swapped
    )

    # Generate both forward and backward judgments in parallel
    async def get_fwd_completion():
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            return await managed.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_fwd},
                ],
                n=3,
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=0.95,
            )

    async def get_bwd_completion():
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            return await managed.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_bwd},
                ],
                n=3,
                max_tokens=max_tokens,
                temperature=1.0,
                top_p=0.95,
            )

    # Gather both completions
    completions_fwd, completions_bwd = await asyncio.gather(
        get_fwd_completion(),
        get_bwd_completion()
    )

    # Extract pre-computed tokens/masks/logprobs from item
    # (These were stored when the original solutions were generated)
    tokens_1 = item[6]
    masks_1 = item[7]
    logprobs_1 = item[8]
    tokens_2 = item[9]
    masks_2 = item[10]
    logprobs_2 = item[11]

    # Score based on judgments...
    score_1, score_2 = calculate_scores(completions_fwd, completions_bwd)

    # Return using pre-computed tokens
    return {
        "tokens": [tokens_1, tokens_2],
        "masks": [masks_1, masks_2],
        "inference_logprobs": [logprobs_1, logprobs_2],
        "scores": [score_1, score_2],
        "messages": [messages_1, messages_2],
    }, []
```

## Migration from Manual Token Handling

### Before: Manual Approach

```python
async def collect_trajectories(self, item):
    prompt = format_prompt(item)

    # Call server
    completion = await self.server.completion(
        prompt=prompt,
        n=8,
        max_tokens=4096,
        logprobs=True,
    )

    # Manually handle tokens
    to_score = []
    for choice in completion.choices:
        # Manually tokenize full text
        full_text = prompt + choice.text
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Manually compute prompt length
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_tokens)

        # Manually apply masking
        masked_tokens = [-100] * prompt_len + tokens[prompt_len:]

        # Manually extract and align logprobs (error-prone!)
        logprobs = [1.0] * prompt_len
        if hasattr(choice, 'logprobs') and choice.logprobs:
            for logprob_obj in choice.logprobs:
                logprobs.append(logprob_obj.logprob)

        # Manually pad/truncate to match length
        while len(logprobs) < len(tokens):
            logprobs.append(1.0)

        to_score.append({
            "tokens": tokens,
            "masked_tokens": masked_tokens,
            "logprobs": logprobs,
        })
```

### After: ManagedServer Approach

```python
async def collect_trajectories(self, item):
    prompt = format_prompt(item)

    # Use managed server - everything automatic!
    async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
        completion = await managed.completion(
            prompt=prompt,
            n=8,
            max_tokens=4096,
        )

        state = managed.get_state()
        nodes = state["nodes"]

    # Extract pre-computed, guaranteed-aligned data
    to_score = []
    for node in nodes:
        to_score.append({
            "tokens": node.tokens,           # ✅ Automatically computed
            "masked_tokens": node.masked_tokens,  # ✅ Automatically masked
            "logprobs": node.logprobs,       # ✅ Automatically aligned
        })
```

**Benefits:**
- ❌ No manual tokenization needed
- ❌ No manual masking calculations
- ❌ No logprob extraction and alignment
- ❌ No off-by-one errors
- ✅ Clean, simple code
- ✅ Guaranteed correctness

## API Reference

### ManagedServer Class

```python
class ManagedServer:
    def __init__(
        self,
        server: APIServer,
        tokenizer: Optional[Any] = None,
        track_tree: bool = False,
    ):
        """
        Initialize the managed server.

        Args:
            server: The underlying APIServer instance to wrap
            tokenizer: Tokenizer for encoding/decoding. If not provided,
                      will attempt to extract from server or create from model name.
            track_tree: If True, maintains a tree structure with parent-child links.
                       If False (default), maintains a simple list that updates in-place.
        """
```

### Methods

#### `async def chat_completion(**kwargs) -> ChatCompletion`
Intercept chat completion call and track sequences.

**Args:**
- `messages`: List of message dicts with 'role' and 'content'
- `n`: Number of completions to generate
- `max_tokens`: Maximum tokens in completion
- Other standard chat completion parameters

**Returns:**
- `ChatCompletion` response (same as OpenAI API)

**Side Effects:**
- Tracks sequences in internal storage
- Updates `current_nodes` list (default mode) or `sequences` dict (tree mode)

#### `async def completion(**kwargs) -> Completion`
Intercept completion call and track sequences.

**Args:**
- `prompt`: The prompt string
- `n`: Number of completions to generate
- `max_tokens`: Maximum tokens in completion
- Other standard completion parameters

**Returns:**
- `Completion` response (same as OpenAI API)

**Side Effects:**
- Tracks sequences in internal storage

#### `def get_state() -> Dict[str, Any]`
Get the current state of tracked sequences.

**Returns:**
- For default mode (track_tree=False):
  ```python
  {
      "nodes": List[SequenceNode]  # List of tracked sequences
  }
  ```
- For tree mode (track_tree=True):
  ```python
  {
      "sequences": Dict[str, SequenceNode],  # Keyed by full_text
      "tree": Dict[str, SequenceNode],       # Alias for compatibility
  }
  ```

#### `def reset()`
Clear all tracked sequences.

### Context Manager (Recommended Usage)

```python
async with server_manager.managed_server(tokenizer=tokenizer) as managed:
    # Use managed.completion() or managed.chat_completion()
    ...
    # Get state before context exits
    state = managed.get_state()
```

The context manager:
- Creates a `ManagedServer` instance
- Returns it for use within the block
- Automatically cleans up when the block exits

## Best Practices

1. **Always use the context manager pattern** for automatic cleanup:
   ```python
   async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
       ...
   ```

2. **Get state before exiting the context**:
   ```python
   async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
       completion = await managed.completion(...)
       state = managed.get_state()  # ✅ Do this inside the context
   # ❌ Don't try to access state here - context has exited
   ```

3. **Use separate contexts for independent completions**:
   ```python
   # Context 1: Generate candidates
   async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
       candidates = await managed.completion(...)
       state1 = managed.get_state()

   # Context 2: Judge candidates (independent)
   async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
       judgments = await managed.completion(...)
       state2 = managed.get_state()
   ```

## Troubleshooting

### Issue: "Extension detection not working"

**Cause:** The new prompt doesn't exactly start with previous node's `full_text`.

**Solution:** Ensure prompt strings match exactly, including whitespace:
```python
# Turn 1 produces: "Hello World"
# Turn 2 prompt must be: "Hello World..." (exact prefix match)
```

## OpenAI Endpoint Limitations

OpenAI's API does not expose token IDs or detailed logprobs in the same way that vLLM, SGLang, and other self-hosted inference servers do. This means **ManagedServer cannot provide accurate token-level training data** when using OpenAI endpoints.

### Default Behavior

By default, attempting to use `managed_server()` with an `OpenAIServer` will raise a `NotImplementedError`:

```python
async with self.server.managed_server() as managed:
    # Raises NotImplementedError if server is OpenAIServer
    ...
```

The error message will explain the limitation and how to opt-in if you don't need real token data.

### DummyManagedServer (Opt-in)

If you're using OpenAI endpoints for **evaluation or testing** (not training) and don't need actual token IDs or logprobs, you can opt-in to use `DummyManagedServer` by setting an environment variable:

```bash
export ATROPOS_ALLOW_DUMMY_MANAGED_SERVER=1
```

With this flag set, `managed_server()` will return a `DummyManagedServer` that:
- Provides the same interface as `ManagedServer`
- Returns **fixed placeholder values** for tokens and logprobs:
  - `tokens`: `[1, 2, 3]`
  - `masked_tokens`: `[-100, 2, 3]`
  - `logprobs`: `[-0.5, -0.5, -0.5]`
- Uses simple text formatting for `full_text`: `role:content` joined by `\n\n`

### When to Use DummyManagedServer

✅ **Appropriate uses:**
- Testing environment logic without needing real token data
- Evaluation workflows where you only need completion text
- Prototyping before switching to a self-hosted inference server

❌ **Not appropriate for:**
- Training (tokens and logprobs are meaningless placeholders)
- Any workflow that depends on accurate token-level information

### Example

```python
import os

# Opt-in to dummy managed server for OpenAI
os.environ["ATROPOS_ALLOW_DUMMY_MANAGED_SERVER"] = "1"

# Now this works with OpenAI endpoints
async with self.server.managed_server() as managed:
    response = await managed.chat_completion(messages=messages, n=4)
    state = managed.get_state()
    nodes = state["nodes"]

    # nodes contain placeholder token data - DO NOT use for training
    for node in nodes:
        print(node.full_text)  # Real completion text
        print(node.tokens)     # [1, 2, 3] - placeholder!
        print(node.logprobs)   # [-0.5, -0.5, -0.5] - placeholder!
```

### Recommendation

For training workloads, use a self-hosted inference server (`VLLMServer`, `SGLangServer`, or `TrlVllmServer`) that provides full token and logprob access. OpenAI endpoints are best suited for evaluation, testing, or workflows that only need completion text.

## Additional Resources

- [ManagedServer Source Code](managed_server.py)
- [ManagedServer Tests](../../tests/test_managed_server.py)
- [Example: math_server_zero.py](../../../environments/math_server_zero.py#L320-L332)
- [Example: math_server.py](../../../environments/math_server.py#L377-L387)
- [BaseEnv Documentation](../README.md)
- [API Server Documentation](../../api/README.md)
