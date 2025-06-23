# 🧬 Population Dynamics System

## Overview

The Population Dynamics System brings **realistic population management** to the LLM Society Simulation, implementing natural growth, aging, death, disasters, and technological events that dynamically control the population around a target of 2,500 agents.

## Key Features

### 🌱 **Natural Population Growth**
- **Births**: Agents are naturally born based on population pressure and environmental conditions
- **Aging**: All agents age continuously and experience age-related effects
- **Immigration**: New agents can join the population when conditions are favorable

### ☠️ **Death System**
- **Natural Death**: Age-based mortality with increasing probability for elderly agents
- **Health-based Mortality**: Low energy/health affects survival chances
- **Event-based Deaths**: Disasters, diseases, and conflicts can cause casualties

### 🌪️ **Dynamic Events**
- **Natural Disasters**: Earthquakes, floods, storms, droughts, wildfires
- **Technological Events**: Industrial accidents, system failures, conflicts, innovation booms
- **Disease Outbreaks**: Flu, food poisoning, epidemics, plagues
- **Environmental Stress**: Affects birth/death rates and population capacity

### 📊 **Realistic Demographics**
- **Age Distribution**: Children, Young Adults, Adults, Seniors
- **Social Stability**: Affected by recent events and population stress
- **Resource Management**: Resource availability affects population carrying capacity
- **Technological Progress**: Advances improve disaster resilience and quality of life

## Configuration

### Population Settings

```yaml
population:
  enable_dynamics: true
  initial_population: 25          # Start small
  target_population: 2500         # Goal population
  max_population: 3000           # Hard limit

  # Birth/Death rates (per simulation step)
  base_birth_rate: 0.0008        # ~0.08% chance per agent per step
  base_death_rate: 0.0003        # ~0.03% chance per agent per step
  aging_rate: 0.005              # Agents age 0.005 years per step

  # Event probabilities (per step)
  disaster_probability: 0.0001   # ~1 every 10,000 steps
  tech_event_probability: 0.00008 # ~1 every 12,500 steps
  disease_probability: 0.00006   # ~1 every 16,667 steps
```

### Growth Dynamics

The system implements realistic population growth curves:

1. **Initial Growth Phase**: High birth rates when population is below target
2. **Stabilization Phase**: Birth/death rates balance as population approaches target
3. **Population Pressure**: Death rates increase when population exceeds target
4. **Environmental Limits**: Resource scarcity and stress limit carrying capacity

## Event Types

### 🌪️ Natural Disasters
- **Earthquake**: High damage, localized impact
- **Flood**: Medium damage, affects low-lying areas
- **Storm**: Lower damage, widespread impact
- **Drought**: Affects resources and long-term stability
- **Wildfire**: High damage, can spread

### ⚡ Technological Events
- **Industrial Accident**: Localized casualties
- **System Failure**: Society-wide disruption
- **Social Conflict**: Resource-based conflicts
- **Innovation Boom**: Positive technological advancement ✨
- **Resource Discovery**: Increases carrying capacity ✨

### 🦠 Disease Outbreaks
- **Flu Outbreak**: Seasonal, moderate spread
- **Food Poisoning**: Localized, low mortality
- **Epidemic**: Serious outbreak requiring intervention
- **Plague**: Devastating, high mortality

*Note: ✨ indicates positive events that improve conditions*

## How It Works

### Population Control Loop

```python
# Each simulation step:
1. Age all agents
2. Calculate dynamic birth/death rates based on:
   - Current population vs target
   - Environmental stress
   - Resource availability
3. Process births and natural deaths
4. Check for random events
5. Update demographics and statistics
```

### Dynamic Rate Calculation

```python
# Birth rate decreases as population approaches target
population_ratio = current_pop / target_population
birth_modifier = max(0.1, 2.0 - population_ratio)
birth_rate = base_birth_rate * birth_modifier * resource_factor / stress_factor

# Death rate increases with overpopulation
death_modifier = max(0.5, population_ratio * 0.8)
death_rate = base_death_rate * death_modifier * stress_factor
```

### Event Impact System

Events have realistic consequences:
- **Immediate Effects**: Casualties, resource loss, happiness/energy changes
- **Environmental Stress**: Increases death rates and decreases birth rates
- **Recovery Period**: Gradual return to normal conditions
- **Technological Progress**: Permanent improvements from positive events

## Usage Examples

### Basic Setup

```python
from src.utils.config import Config
from src.simulation.society_simulator import SocietySimulator

# Enable population dynamics
config = Config()
config.population.enable_dynamics = True
config.population.initial_population = 50
config.population.target_population = 2500

# Run simulation
simulator = SocietySimulator(config)
await simulator.run()
```

### Testing with Accelerated Events

```python
# Increase event rates for testing
config.population.disaster_probability = 0.002    # More frequent disasters
config.population.base_birth_rate = 0.001         # Higher birth rate
config.simulation.max_steps = 5000                # Longer simulation
```

### Load from Configuration File

```bash
# Use the provided test configuration
python src/main.py --config config_population_test.yaml --agents 25 --steps 5000
```

## Monitoring and Statistics

### Real-time Statistics

The system provides comprehensive real-time monitoring:

```
📊 Step 1590 | ⚡ 52.86 SPS | 👥 118 agents | 🎯 Target: 100 (118.0%) |
👴 Avg Age: 9.3 | ⚖️ Stability: 0.97 | ☠️ Deaths: 30 | 📰 Events: 3
```

### Final Demographics Report

```
👥 Population Dynamics:
   Target Population: 2500
   Final Population: 2456
   Population Ratio: 98.2%
   Total Deaths: 1247
   Average Age: 34.2 years
   Social Stability: 0.89
   Environmental Stress: 0.12
   Tech Level: 1.45
   Resource Availability: 1.18

   Age Distribution:
     Children: 612
     Young Adults: 1024
     Adults: 689
     Seniors: 131
```

## Performance Impact

The population dynamics system is designed for efficiency:

- **Minimal Overhead**: ~0.1ms additional processing per step
- **Scalable**: Handles up to 3,000 agents efficiently
- **Memory Efficient**: Lightweight tracking structures
- **Configurable**: Can be disabled for pure agent simulation

## Simulation Scenarios

### Scenario 1: Natural Growth to 2,500

Starting with 25 agents, the population naturally grows to 2,500 over several thousand simulation steps, experiencing various events along the way.

### Scenario 2: Disaster Recovery

A major plague reduces population by 30%, triggering:
1. **Immediate Response**: Lower birth rates due to stress
2. **Recovery Phase**: Gradual population rebuilding
3. **Adaptation**: Improved disease resistance

### Scenario 3: Technological Boom

Innovation events create a positive feedback loop:
1. **Resource Discovery**: Increases carrying capacity
2. **Tech Advancement**: Improves disaster resilience
3. **Population Growth**: Sustained growth beyond normal limits

## Integration with Existing Systems

### LLM Agent Integration
- Agents respond to population events in their decision-making
- Age affects agent behavior (energy, happiness, social connections)
- Environmental stress influences agent mood and actions

### 3D Asset Generation
- Population events can trigger asset generation (monuments, disaster markers)
- Demographic changes affect the types of objects agents create
- Cultural evolution through asset preferences

### Metrics and Monitoring
- Population statistics integrated into real-time monitoring
- Event history tracked for analysis
- Demographics exported for research purposes

## Future Enhancements

### Planned Features
- **Family Systems**: Agents form families and have children together
- **Migration Patterns**: Agents move between regions based on conditions
- **Cultural Evolution**: Population groups develop distinct cultures
- **Economic Systems**: Resource-based economy affecting population dynamics
- **Genetic Algorithms**: Agent traits evolve over generations

### Research Applications
- **Social Science**: Study population dynamics in virtual societies
- **Disaster Planning**: Model disaster response and recovery
- **Policy Testing**: Evaluate interventions on population health
- **Emergent Behavior**: Observe complex social phenomena

## Testing

Run the test suite:

```bash
# Basic functionality test
python test_population_dynamics.py

# Extended test with configuration
python src/main.py --config config_population_test.yaml
```

## Conclusion

The Population Dynamics System transforms the LLM Society Simulation from a static multi-agent system into a **living, breathing virtual world** where population naturally grows, adapts to challenges, and evolves over time.

This system brings us significantly closer to the ultimate goal of a 2,500-agent society that maintains itself through realistic demographic processes, making the simulation more engaging, scientifically valuable, and true to real-world population dynamics.

---

*Ready to watch your virtual society grow, adapt, and thrive! 🌱*
