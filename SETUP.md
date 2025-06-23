# Society Simulator - Setup Guide

## Quick Start ⚡

```bash
# 1. Install minimal dependencies
pip install numpy matplotlib pandas tqdm

# 2. Run basic simulation
python run_simulation.py --agents 50 --steps 200

# 3. Run with visualization (closes window to exit)
python run_simulation.py --agents 25 --steps 500 --visualize

# 4. Save results
python run_simulation.py --agents 100 --save my_experiment.json
```

## Installation Options

### Option 1: Minimal Setup (No LLM)
```bash
pip install -r requirements-minimal.txt
python run_simulation.py --test
```

### Option 2: With LLM Support
```bash
pip install -r requirements-minimal.txt
pip install openai anthropic

# Set API key
export OPENAI_API_KEY="your-key-here"
python run_simulation.py --agents 25 --llm openai
```

### Option 3: Full Development Setup
```bash
pip install -r requirements.txt  # Full dependencies (may have issues)
# OR
pip install -r requirements-minimal.txt  # Recommended
```

## Command Line Usage

### Basic Commands
```bash
# Default simulation
python run_simulation.py

# Specify parameters
python run_simulation.py --agents 100 --steps 1000

# With visualization
python run_simulation.py --agents 50 --visualize

# Save results
python run_simulation.py --save experiment1.json

# Load and continue
python run_simulation.py --load experiment1.json --continue 500
```

### Configuration Files
```bash
# Use predefined configurations
python run_simulation.py --config config_templates/basic_experiment.json
python run_simulation.py --config config_templates/llm_experiment.json
python run_simulation.py --config config_templates/performance_test.json
```

### Testing and Benchmarks
```bash
# Quick functionality test
python run_simulation.py --test

# Performance benchmark
python run_simulation.py --benchmark

# Stress test
python run_simulation.py --agents 200 --steps 100
```

## LLM Integration

### OpenAI Setup
```bash
pip install openai
export OPENAI_API_KEY="sk-your-key-here"
python run_simulation.py --llm openai --model gpt-3.5-turbo
```

### Anthropic Setup
```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
python run_simulation.py --llm anthropic --model claude-3-haiku-20240307
```

### Fallback Mode
If no API key is provided, the system automatically falls back to rule-based agent behavior.

## Configuration

### Example Configuration File
```json
{
  "simulation": {
    "agents": 50,
    "steps": 500,
    "world_size": [100, 100],
    "seed": 42
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "temperature": 0.7
  },
  "output": {
    "save_results": true,
    "visualization": true,
    "verbose": true
  }
}
```

### Available Parameters
- **agents**: Number of agents (1-1000+)
- **steps**: Simulation duration (1-10000+)
- **world_size**: World dimensions [width, height]
- **llm**: LLM provider (none, openai, anthropic, mock)
- **visualize**: Show real-time 2D visualization
- **save**: Save results to JSON file

## Output Formats

### Console Output
```
🚀 Starting Society Simulation
   Agents: 50
   Steps: 200
   Intelligence: none

📊 Final Results:
   Average Energy: 0.37
   Average Happiness: 1.55
   Social Connections: 58
   Economic Activity: 16915 total currency
   Cultural Diversity: 5 active groups
```

### JSON Results File
```json
{
  "metadata": {
    "agents": 50,
    "steps": 200,
    "timestamp": 1749067770.18
  },
  "statistics": {
    "avg_energy": 0.37,
    "avg_happiness": 1.55,
    "total_connections": 58
  },
  "agents": [
    {
      "agent_id": "agent_0",
      "type": "farmer",
      "position": {"x": 45.2, "y": 67.8},
      "energy": 0.43,
      "resources": {"food": 25, "currency": 340}
    }
  ]
}
```

## Visualization

### 2D Real-time Visualization
```bash
python run_simulation.py --agents 50 --visualize
```

Shows:
- Agent positions (colored by type)
- Agent states (different marker shapes)
- Social connections (gray lines)
- Live statistics (energy, happiness, connections)
- Real-time plots of metrics over time

Close the plot window to stop the simulation.

## Performance

### Expected Performance
- **25 agents**: 2,000+ SPS (steps per second)
- **50 agents**: 400-500 SPS
- **100 agents**: 100-200 SPS
- **200+ agents**: 50-100 SPS

### Performance Tips
1. Disable visualization for large simulations
2. Use `--quiet` mode to reduce output overhead
3. Disable LLM for maximum performance
4. Use rule-based agents for stress testing

## Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
pip install numpy matplotlib pandas tqdm
```

#### LLM not working
- Check API key: `echo $OPENAI_API_KEY`
- Install client: `pip install openai`
- Use fallback: `--llm none`

#### Performance issues
- Reduce agent count: `--agents 25`
- Disable visualization: Remove `--visualize`
- Use basic config: `--config config_templates/basic_experiment.json`

#### Visualization not showing
- Install matplotlib: `pip install matplotlib`
- Check display: Run `python -c "import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.show()"`
- Use non-interactive mode if needed

### System Requirements
- Python 3.8+
- 4GB+ RAM (for 100+ agents)
- Display for visualization
- Internet for LLM APIs

## Examples

### Research Experiment
```bash
# Run controlled experiment
python run_simulation.py \
  --agents 100 \
  --steps 1000 \
  --seed 42 \
  --save research_experiment_1.json \
  --verbose

# Analyze results
python -c "
import json
with open('research_experiment_1.json') as f:
    data = json.load(f)
print('Social density:', data['statistics']['total_connections'] / data['metadata']['agents'])
"
```

### LLM Comparison
```bash
# Rule-based agents
python run_simulation.py --agents 25 --llm none --save baseline.json

# OpenAI agents
python run_simulation.py --agents 25 --llm openai --save openai.json

# Compare results
python analyze_results.py baseline.json openai.json
```

### Performance Testing
```bash
# Test different scales
for agents in 25 50 100 200; do
  echo "Testing $agents agents"
  python run_simulation.py --agents $agents --steps 100 --quiet
done
```

## Next Steps

1. **Basic Usage**: Start with `python run_simulation.py --test`
2. **Experiment**: Try different configurations
3. **Visualize**: Use `--visualize` to see agent behavior
4. **Scale Up**: Test with more agents and longer simulations
5. **LLM Integration**: Add OpenAI/Anthropic API keys for intelligent agents
6. **Research**: Use saved results for analysis and comparison

## Support

- **Issues**: Check console output for error messages
- **Performance**: Use `--benchmark` to test your system
- **Documentation**: See code comments and docstrings
- **Examples**: Try the config templates in `config_templates/`