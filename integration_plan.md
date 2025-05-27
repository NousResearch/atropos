# AMIEN + AI Research Tools Integration Plan

## Executive Summary

Based on comprehensive research, we can integrate two proven AI research systems into AMIEN:

1. **AI Scientist v1/v2** (Sakana AI) - Autonomous paper generation with peer review
2. **FunSearch** (DeepMind) - Mathematical function discovery through evolutionary algorithms

Both systems are open-source, proven to work, and can be integrated with your existing GCP infrastructure.

## Phase 1: Foundation Integration (Week 1)

### AI Scientist Integration

**Repository Setup:**
```bash
# Clone both versions
git clone https://github.com/SakanaAI/AI-Scientist.git
git clone https://github.com/SakanaAI/AI-Scientist-v2.git
```

**Key Integration Points:**
- **Templates**: AI Scientist v1 uses templates (NanoGPT, 2D Diffusion, Grokking)
- **Template-Free**: AI Scientist v2 uses agentic tree search (more flexible)
- **Cost**: ~$15-20 per paper with Claude 3.5 Sonnet
- **Success Rate**: v1 has higher success rates with good templates, v2 is more exploratory

**AMIEN Integration Strategy:**
```python
# Add to production_research_pipeline.py
class AIScientistManager:
    def __init__(self):
        self.v1_templates = ["spatial_reasoning", "vr_affordances", "visual_cues"]
        self.v2_agentic = True
        
    def generate_research_paper(self, experiment_results):
        # Use v1 for well-defined spatial reasoning problems
        # Use v2 for open-ended VR exploration
        pass
```

### FunSearch Integration

**Repository Setup:**
```bash
git clone https://github.com/google-deepmind/funsearch.git
```

**Key Capabilities:**
- **Mathematical Discovery**: Proven to discover new mathematical constructions
- **Function Evolution**: Evolves priority functions using LLMs + evolutionary algorithms
- **Distributed System**: 15 samplers + 150 CPU evaluators
- **Cost**: Much lower than AI Scientist (~$1-5 per discovery)

**AMIEN Integration Strategy:**
```python
# Add to enhanced_padres_perplexity.py
class FunSearchManager:
    def __init__(self):
        self.evaluator_nodes = 150
        self.sampler_nodes = 15
        
    def evolve_spatial_functions(self, vr_environment_data):
        # Evolve priority functions for spatial reasoning
        # Discover new VR affordance algorithms
        pass
```

## Phase 2: Discovery Engines (Week 2)

### Spatial Reasoning Function Discovery

**Objective**: Use FunSearch to discover new spatial reasoning algorithms

```python
# New file: spatial_function_discovery.py
class SpatialFunctionEvolver:
    def __init__(self):
        self.problem_skeleton = """
        def spatial_priority(vr_object, user_context, environment):
            # FunSearch will evolve this function
            pass
        """
        
    def evaluate_spatial_function(self, function, test_cases):
        # Evaluate how well the function performs spatial reasoning
        # Use Padres API results as ground truth
        pass
```

**Integration with Padres API:**
- Use existing Padres experiments as evaluation data
- Evolve functions that predict spatial affordances
- Discover new visual cue patterns

### Synthetic VR User Generation

**Objective**: Create 10,000 diverse synthetic users for testing

```python
# New file: synthetic_user_generator.py
class SyntheticUserManager:
    def __init__(self):
        self.cultural_profiles = self.load_cultural_data()
        self.neurological_profiles = self.load_neurological_data()
        
    def generate_diverse_users(self, count=10000):
        # Generate users with different:
        # - Cultural backgrounds (50+ countries)
        # - Age groups (5-95 years)
        # - Neurological variations (ADHD, autism, etc.)
        # - Physical abilities
        pass
```

## Phase 3: Advanced Integration (Week 3-4)

### Multi-Environment VR Testing

**Objective**: Test discoveries across 1,000 parallel VR environments

```python
# New file: vr_environment_manager.py
class VREnvironmentManager:
    def __init__(self):
        self.environments = {
            'physics_variants': ['earth_gravity', 'moon_gravity', 'zero_gravity'],
            'contexts': ['office', 'home', 'outdoor', 'industrial'],
            'lighting': ['bright', 'dim', 'colored', 'dynamic']
        }
        
    def create_parallel_environments(self, count=1000):
        # Create diverse VR testing environments
        # Each with different physics, contexts, lighting
        pass
```

### Cross-Domain Inspiration Engine

**Objective**: Apply insights from fireflies, casino psychology, etc.

```python
# New file: inspiration_engine.py
class CrossDomainInspiration:
    def __init__(self):
        self.domains = {
            'fireflies': 'bioluminescent_patterns',
            'casinos': 'attention_capture_mechanisms',
            'nature': 'organic_movement_patterns'
        }
        
    def generate_inspired_hypotheses(self, domain):
        # Use AI Scientist to generate papers inspired by other domains
        # "Firefly-Inspired VR Visual Cues for Enhanced Spatial Navigation"
        pass
```

## Infrastructure Requirements

### Google Cloud Platform Setup

**Compute Resources:**
```yaml
# deployment_config.yaml
ai_scientist_cluster:
  instances: 10
  machine_type: "n1-standard-8"
  gpu: "nvidia-tesla-t4"
  
funsearch_cluster:
  samplers: 15
  evaluators: 150
  machine_type: "n1-standard-4"
  
vr_simulation_cluster:
  instances: 50
  machine_type: "n1-highmem-8"
  gpu: "nvidia-tesla-v100"
```

**Storage:**
```yaml
storage:
  research_papers: "gs://amien-research-papers"
  discovered_functions: "gs://amien-discovered-functions"
  vr_experiment_data: "gs://amien-vr-experiments"
  synthetic_users: "gs://amien-synthetic-users"
```

### API Keys Required

```bash
# Add to Secret Manager
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"  # For Claude 3.5 Sonnet
export GEMINI_API_KEY="your_gemini_key"
export S2_API_KEY="your_semantic_scholar_key"  # For literature search
```

## Expected Outcomes

### Week 1 Deliverables
- [ ] AI Scientist v1 & v2 integrated with AMIEN
- [ ] FunSearch integrated with Padres API
- [ ] First spatial reasoning function discovered
- [ ] First AI-generated research paper on VR affordances

### Week 2 Deliverables
- [ ] 1,000 synthetic VR users generated
- [ ] 100 parallel VR environments created
- [ ] Cross-domain inspiration engine operational
- [ ] 5 new spatial reasoning algorithms discovered

### Week 3-4 Deliverables
- [ ] 10,000 synthetic users across 1,000 environments
- [ ] 50 AI-generated research papers
- [ ] 20 novel VR affordance discovery algorithms
- [ ] Unity/Unreal plugin generation system

## Cost Estimates

**Monthly Operating Costs:**
- AI Scientist: $1,500 (100 papers × $15 each)
- FunSearch: $500 (100 function discoveries × $5 each)
- GCP Infrastructure: $5,000 (compute, storage, networking)
- **Total: ~$7,000/month**

**Expected ROI:**
- 100+ research papers per month
- 100+ novel algorithms per month
- Potential patent applications: 10-20 per month
- Academic citations and recognition: Significant

## Risk Mitigation

### Technical Risks
1. **Integration Complexity**: Start with simple templates, gradually increase complexity
2. **Resource Costs**: Implement auto-scaling and budget alerts
3. **Quality Control**: Implement automated validation for all discoveries

### Scientific Risks
1. **Reproducibility**: All experiments logged and version-controlled
2. **Validation**: Human expert review for critical discoveries
3. **Ethics**: Clear attribution for AI-generated content

## Success Metrics

### Quantitative
- Papers generated per week: Target 25
- Functions discovered per week: Target 25
- Successful VR experiments: Target 1,000
- Cost per discovery: Target <$15

### Qualitative
- Peer review acceptance rate: Target >50%
- Novel insights generated: Measured by expert review
- Cross-domain inspiration success: Measured by implementation rate

## Next Steps

1. **Immediate**: Clone repositories and begin integration
2. **Week 1**: Set up basic AI Scientist pipeline
3. **Week 2**: Integrate FunSearch with Padres API
4. **Week 3**: Scale to full multi-environment testing
5. **Week 4**: Launch continuous discovery pipeline

This integration plan transforms AMIEN from a research pipeline into a **discovery engine** capable of autonomous scientific breakthroughs in VR spatial reasoning. 