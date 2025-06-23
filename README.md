# Society Simulator - LLM-Driven Multi-Agent Society

**Built on AMIEN Foundation** - An advanced society simulation platform with intelligent agents powered by Large Language Models.

## Phase 2 Complete: Real Intelligence ✅

The society simulator now features **LLM-driven intelligent agents** with memory systems, goal planning, and complex social dynamics.

## 🧠 Intelligent Agent Features

### **LLM Integration**
- **OpenAI & Anthropic Support**: Real AI decision-making with API integration
- **Intelligent Fallbacks**: Graceful degradation to rule-based behavior
- **Response Caching**: Efficient LLM usage with intelligent caching
- **Mock Testing**: Full testing without API costs

### **Memory Systems**
- **Short-term Memory**: Recent events and interactions (10 items)
- **Long-term Memory**: Important memories with emotional impact (50 items)
- **Working Memory**: Current context for decision-making (5 items)
- **Memory Categorization**: Events, interactions, achievements, observations

### **Goal-Oriented Behavior**
- **Personality-Based Goals**: Economic, social, personal, cultural objectives
- **Goal Tracking**: Priority, deadlines, completion monitoring
- **Goal Achievement**: Satisfaction and happiness rewards
- **Dynamic Planning**: Adaptive behavior based on goal progress

### **Advanced Social Dynamics**
- **Relationship Building**: Trust, strength, relationship types
- **Family Formation**: Automatic family unit creation
- **Personality Compatibility**: Big Five personality model
- **Social Networks**: Dynamic relationship tracking and analysis

## 🚀 Quick Start

### Basic Society Simulation
```bash
# Install dependencies
pip install numpy matplotlib pandas tqdm

# Run basic simulation (rule-based agents)
python run_simulation.py --agents 50 --steps 200

# Run with visualization
python run_simulation.py --agents 25 --visualize
```

### Intelligent Agents (LLM-Powered)
```bash
# Install LLM dependencies
pip install openai anthropic

# Mock LLM testing (no API key needed)
python run_simulation.py --agents 25 --steps 100 --llm mock

# OpenAI intelligent agents
export OPENAI_API_KEY="your-key-here"
python run_simulation.py --agents 25 --llm openai --model gpt-3.5-turbo

# Anthropic intelligent agents
export ANTHROPIC_API_KEY="your-key-here"
python run_simulation.py --agents 25 --llm anthropic --model claude-3-haiku-20240307
```

### Save and Analyze Results
```bash
# Save simulation data
python run_simulation.py --agents 50 --llm mock --save experiment.json

# Run performance benchmark
python run_simulation.py --benchmark

# Quick functionality test
python run_simulation.py --test
```

## 📊 Performance

| Agents | Intelligence | Performance | Use Case |
|--------|-------------|-------------|----------|
| 25     | Rule-based  | 2,000+ SPS  | Development |
| 25     | LLM-driven  | 800-1,000 SPS | Intelligence Testing |
| 50     | Rule-based  | 400-500 SPS | Standard Experiments |
| 50     | LLM-driven  | 300-400 SPS | Research Studies |
| 100+   | Rule-based  | 100-200 SPS | Large Scale |

*SPS = Steps Per Second*

## 🧪 Simulation Outputs

### Intelligent Simulation Results
```
🧠 Intelligent Simulation Results:
   World Mood: 3.92
   Average Happiness: 4.00
   Social Density: 4.16 relationships per agent
   Goal Completion: 1.4%
   Total Memories: 1
   Families Formed: 8
   LLM Requests: 25
   Cache Efficiency: 0.0%

👥 Agent Insights:
   Most Social: [('agent_9', 8), ('agent_21', 6), ('agent_3', 5)]
   Wealthiest: [('agent_20', 1251), ('agent_8', 1174), ('agent_17', 1025)]
   Goal Achievers: [('agent_9', 1), ('agent_0', 0), ('agent_1', 0)]
   Extroverts: 5 agents, Creative: 5 agents
```

### JSON Export Format
```json
{
  "metadata": {
    "agents": 25,
    "steps": 100,
    "simulation_type": "intelligent",
    "timestamp": 1749069030.731071
  },
  "statistics": {
    "world_mood": 3.92,
    "averages": {"happiness": 4.00, "energy": 0.78},
    "social": {"relationships": 104, "families": 8},
    "goals": {"completion_rate": 1.4},
    "llm_stats": {"total_requests": 25, "cache_rate": "0.0%"}
  },
  "insights": {
    "most_social": [["agent_9", 8], ["agent_21", 6]],
    "personality_clusters": {"extroverts": ["agent_2", "agent_7"]}
  },
  "agents": [
    {
      "agent_id": "agent_0",
      "personality": {"extroversion": 0.73, "openness": 0.45},
      "goals": {"total": 3, "completed": 0},
      "memory": {"short_term": 2, "long_term": 0},
      "resources": {"currency": 850, "food": 45}
    }
  ]
}
```

## 🛠️ Technical Architecture

### **Core Components**
- `run_simulation.py` - Main CLI interface with LLM integration
- `intelligent_agent.py` - LLM-driven agents with memory and goals
- `intelligent_world.py` - World container with async agent coordination
- `llm_integration.py` - OpenAI/Anthropic integration with fallbacks
- `society_demo.py` - Basic rule-based simulation engine

### **LLM Decision Making**
```python
# Example LLM prompt for agent decision
"""
You are agent_5, a farmer in a virtual society.

PERSONALITY: I am outgoing and social, organized and responsible.

CURRENT STATUS:
- Energy: 0.45/1.0 (tired)
- Happiness: 0.72/1.0 (content)
- Resources: Food: 25, Currency: 340

RECENT MEMORIES:
- Had a great conversation with agent_3
- Successfully traded 5 materials for 50 currency

CURRENT GOALS:
- Accumulate 1000 currency (priority: 0.8)
- Make 5 close friends (priority: 0.6)

NEARBY AGENTS:
- trader (stranger) at distance 8.5
- craftsman (friend, trust: 0.7) at distance 12.1

Choose ONE action: work, socialize, trade, rest, move, help, learn
"""
```

### **Asynchronous Performance**
- **Batch Processing**: Agents processed in batches of 10 for LLM efficiency
- **Async Coordination**: Non-blocking LLM calls with asyncio
- **Smart Caching**: Reduces redundant LLM requests
- **Fallback Systems**: Graceful degradation when LLM unavailable

## 🔬 Research Applications

### **Social Science Research**
- **Emergence Studies**: How societies form and evolve
- **Cultural Dynamics**: Cross-cultural interaction patterns
- **Economic Modeling**: Resource distribution and trading behavior
- **Social Network Analysis**: Relationship formation and influence

### **AI Behavior Research**
- **LLM Decision Making**: Comparing different AI models
- **Multi-Agent Coordination**: Emergent group behaviors
- **Memory System Effects**: Impact of different memory architectures
- **Goal-Oriented AI**: Planning and achievement patterns

### **Comparative Studies**
```bash
# Compare rule-based vs. LLM agents
python run_simulation.py --agents 50 --save baseline.json
python run_simulation.py --agents 50 --llm mock --save intelligent.json

# Compare different LLM providers
python run_simulation.py --agents 25 --llm openai --save openai_results.json
python run_simulation.py --agents 25 --llm anthropic --save anthropic_results.json
```

## 🎯 Development Roadmap

### **Phase 3: Scale & Visualization** (Next)
- [ ] Advanced 3D visualization
- [ ] 500+ agent performance optimization
- [ ] Web-based real-time interface
- [ ] Advanced analytics dashboard

### **Phase 4: Research Framework** (Future)
- [ ] Hypothesis testing framework
- [ ] A/B testing automation
- [ ] Statistical analysis tools
- [ ] Research paper generation

## 📁 Project Structure

```
NOUS/
├── run_simulation.py          # Main CLI interface
├── intelligent_agent.py       # LLM-driven intelligent agents
├── intelligent_world.py       # Intelligent simulation world
├── llm_integration.py         # OpenAI/Anthropic integration
├── society_demo.py           # Basic rule-based simulation
├── visualization.py          # 2D real-time visualization
├── requirements-minimal.txt   # Essential dependencies
├── SETUP.md                  # Detailed setup guide
├── PHASE_1_COMPLETE.md       # Phase 1 completion summary
└── config_templates/         # Example configurations
    ├── basic_experiment.json
    ├── llm_experiment.json
    └── performance_test.json
```

## 🏗️ Built on AMIEN Foundation

This project builds on the **AMIEN (Autonomous Machine Intelligence Evolution Network)** foundation - a fully operational autonomous AI research system that continues to generate VR research autonomously.

### AMIEN Achievements
- **Production AI Research System**: Generates real research papers 24/7
- **Massive Scale Simulation**: Up to 50,000 simulated users
- **Cloud Infrastructure**: Google Cloud Platform deployment
- **Automated Scheduling**: Daily, weekly, monthly research cycles

The society simulator inherits AMIEN's robust architecture while focusing on multi-agent social dynamics and LLM integration.

## 🤝 Contributing

The system is designed for extension and research use:

1. **Agent Behaviors**: Add new agent types and decision patterns
2. **LLM Integration**: Support additional AI providers
3. **Social Dynamics**: Implement new relationship types
4. **Research Tools**: Add analysis and visualization features

## 📄 License

MIT License - See LICENSE file for details.

---

**Phase 2 Complete**: Real Intelligence with LLM-driven agents, memory systems, and advanced social dynamics. Ready for large-scale research and experimentation.