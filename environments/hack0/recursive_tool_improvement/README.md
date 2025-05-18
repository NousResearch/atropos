# Recursive Tool Improvement Environment

This environment trains language models to create, improve, and recursively refine tool compositions to solve complex problems. It addresses a novel research direction focused on higher-order tool composition, longitudinal improvement patterns, and meta-cognitive evaluation development.

## Core Concept

The Recursive Tool Improvement Environment enables models to:

1. **Create** initial tool compositions for a given problem
2. **Execute** the compositions and observe results
3. **Critique** their approach
4. **Improve** the compositions based on self-critique
5. **Repeat** iteratively, creating a feedback loop of continuous improvement

This environment is designed to study how language models can learn to compose primitive tools into higher-order, reusable abstractions through a process of self-improvement.

## Key Research Questions

This environment explores several novel research directions:

1. **Tool Composition** (Highly Novel): How can LLMs learn to compose primitive tools into higher-order, reusable compositions rather than just sequencing individual tools?

2. **Optimization Patterns** (Highly Novel): What patterns emerge in the optimization process as models refine their tool compositions across multiple iterations?

3. **Meta-cognitive Evaluation** (Highly Novel): Can models develop their own principles for evaluating and improving tool compositions without human-defined metrics?

4. **Scaling Effects**: How does model scale impact the ability to create and recursively improve tool compositions?

## Features

- **Tool Registry**: A flexible registry for defining primitive tools that can be composed
- **Execution Engine**: Sandboxed execution environment for safely running code
- **Binary Verification**: Simple 0/1 rewards based on functional correctness
- **Multi-Turn Conversations**: Support for iterative improvement cycles
- **Multiple Reasoning Modes**: Support for deductive, abductive, and inductive reasoning

## Getting Started

### Installation

```bash
cd atropos/environments/hack0/recursive_tool_improvement
pip install -r requirements.txt
```

### Running Locally

You can run the environment using the Atropos command-line interface:

```bash
python recursive_tool_improvement.py process --env.data_path_to_save_groups output.jsonl
```

This will:
1. Load or create test problems
2. Generate rollout data for each problem
3. Save the results to output.jsonl
4. Create an HTML visualization (output.html)

### Configuration

Key configuration parameters:

- `max_iterations`: Number of improvement iterations per trajectory (default: 3)
- `tool_set`: Which tool set to use (default: "basic")
- `reasoning_mode`: Reasoning mode to use (deduction, abduction, induction, all)
- `verification_type`: Verification type (binary or graduated)

## Components

### Tool Registry

The environment includes a registry of basic text processing tools:

- String manipulation (split, join, replace, case conversion)
- Regular expression operations (extraction, replacement)
- List operations (filtering, sorting, deduplication)
- JSON processing (parsing, formatting)

### Execution Engine

Features a secure sandbox for executing tool compositions:

- Timeout limits
- Memory restrictions
- Module import controls
- Function call limits
- Standard output/error capture

### Reward Functions

- **Binary Verification**: Clean 0/1 reward signal based on functional correctness
- **Improvement Reward**: Measures progress between iterations

## Research Insights

This environment is informed by three key research projects:

1. **Absolute-Zero-Reasoner**:
   - Self-proposal mechanism
   - Three reasoning modes (deduction, abduction, induction)
   - Execution-based verification

2. **Tool-N1**:
   - Binary reward mechanism
   - Semantic equivalence in validation
   - RL-first approach (vs supervised fine-tuning)

3. **BespokeLabs Multi-Turn Tool Use**:
   - "Less Is More" reward design approach
   - Training stability techniques
   - Multi-turn interaction handling

## Demo

For a demonstration of the environment in action, run:

```bash
python recursive_tool_improvement.py process --env.total_steps 5 --env.group_size 1
```

This will generate JSON and HTML visualizations showing the iterative improvement process across multiple problems.

## Evaluation Metrics

The environment tracks several key metrics:

- **Binary Success Rate**: Percentage of problems solved correctly
- **Improvement Rate**: Average score improvement across iterations
- **Convergence Speed**: Number of iterations needed to reach optimal solution
- **Tool Efficiency**: Reduction in tool usage count while maintaining/improving results

## WandB Integration

Metrics are automatically logged to Weights & Biases, including:
- Success rates
- Improvement trends
- Convergence speeds
- Example trajectories

## License

This project is licensed under the MIT License.