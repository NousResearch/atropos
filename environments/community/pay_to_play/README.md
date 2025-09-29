# Pay-to-Play Environment with Mixture of Judges

A reinforcement learning environment where an AI agent must strategically select and pay specialized agent cards before each evaluation, implementing economic constraints and strategic decision-making in AI training.

This environment builds upon recent advances in RLHF with AI feedback ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)) and mixture of judges approaches ([Xu et al., 2024](https://arxiv.org/abs/2409.20370)) to create a training paradigm that combines economic incentives with multi-agent evaluation.

## 🎯 Overview

This environment transforms the traditional AI evaluation process by introducing:

- **Economic Constraints**: Real USDC payments on Base blockchain (or simulated)
- **Strategic Agent Card Selection**: Agent chooses from multiple specialized agent cards with different expertise and prices
- **Budget Management**: Agent must balance cost vs. quality across training iterations
- **Performance Tracking**: Historical data informs future agent card selection decisions

## 🎯 Architecture

### Separated Configuration Design

The system uses a clean separation of concerns:

- **`agent_cards_config.py`**: Agent card definitions, pricing, specialties, and system prompts
- **`secrets.json`**: Wallet addresses and private keys
- **`pay_to_play_env.py`**: Main environment logic and orchestration

This design allows:
- Easy addition of new agent cards without touching wallet credentials
- Secure management of private keys separate from code
- Version control safety (secrets.json should never be committed)

### Agent Card System

The environment includes three specialized agent cards defined in `agent_cards_config.py`:

| Agent Card | Price | Specialties | Description |
|------------|-------|-------------|-------------|
| **Technical Expert** | $0.03 | Technical Accuracy, Reasoning Logic | Premium agent card for STEM questions, complex reasoning, factual correctness |
| **Communication Specialist** | $0.02 | Clarity Communication | Mid-tier agent card focusing on readability, structure, and clear explanations |
| **Creative Thinker** | $0.01 | Creative Thinking | Budget agent card for creativity, originality, and innovative solutions |

### Adding New Agent Cards

To add a new agent card:

1. **Add to `agent_cards_config.py`**:
```python
"new_agent_card": AgentCardConfig(
    name="New Agent Card Name",
    price_usd=Decimal("0.025"),
    specialties=[AgentCardSpecialty.FACTUAL_CORRECTNESS],
    description="Agent card description here",
    system_prompt="Your system prompt here..."
),
```

2. **Add wallet to `secrets.json`**:
```json
"new_agent_card": {
  "address": "0x...",
  "private_key": "0x..."
}
```

The environment will automatically load and initialize the new agent card.

### Agent Decision Process

1. **Question Analysis**: Agent directly analyzes the question and determines what expertise is needed
2. **Agent Card Evaluation**: Assess available agent cards based on:
   - Specialty matching to the question requirements
   - Historical performance
   - Budget constraints
   - Cost-effectiveness
3. **Strategic Selection**: Choose 1-3 agent cards based on configuration
4. **Payment Execution**: Make USDC payments to selected agent cards
5. **Evaluation**: Receive scores from multiple agent cards and aggregate results

### Budget Tracking

The `BudgetTracker` class monitors:
- Current balance and total spending
- Per-agent-card spending breakdown
- Average cost per evaluation
- Affordability checks

## 📊 Configuration

```python
class PayToPlayConfig(BaseEnvConfig):
    testing_mode: bool = False  # True for simulated payments
    initial_budget_usd: float = 1.0  # Starting budget
    min_judges_per_eval: int = 1  # Minimum agent cards required
    max_judges_per_eval: int = 3  # Maximum agent cards allowed
```

## 🚀 Usage

### Initial Setup

1. **Copy wallet template**:
```bash
cd src/environments/pay_to_play/
cp secrets.json.template secrets.json
```

2. **Configure your wallets** by editing `secrets.json` with real addresses and private keys

3. **Add to gitignore** (should already be there):
```bash
echo "src/environments/pay_to_play/secrets.json" >> .gitignore
```

### Basic Setup

```python
from environments.pay_to_play.pay_to_play_env import PayToPlayEnv, PayToPlayConfig
from atroposlib.envs.base import APIServerConfig

# Configuration
config = PayToPlayConfig(
    testing_mode=True,  # Set to False for real blockchain payments
    initial_budget_usd=0.50,
    min_judges_per_eval=1,
    max_judges_per_eval=2,
    tokenizer_name="gpt2",
    use_wandb=True
)

server_configs = [
    APIServerConfig(
        model_name="your-model",
        base_url="http://localhost:9001/v1",
        api_key="your-key",
        num_requests_for_eval=64,
    ),
]

# Initialize environment
env = PayToPlayEnv(config, server_configs, testing=True)
```

### Training Loop

```python
async def training_loop():
    await env.setup()

    for step in range(config.total_steps):
        # Get next question
        question = await env.get_next_item()

        # Agent selects agent cards and makes payments
        # Evaluates response and gets training signal
        scored_data, _ = await env.collect_trajectories(question)

        # Log metrics
        await env.wandb_log()
```

## 💰 Economic Model

### Pricing Strategy

- **Technical Expert ($0.03)**: Premium pricing reflects high accuracy and specialized knowledge
- **Communication Specialist ($0.02)**: Mid-tier pricing for clarity and accessibility focus
- **Creative Thinker ($0.01)**: Budget option encouraging creativity and innovation

### Budget Scenarios

| Budget | Strategy | Evaluations Possible |
|--------|----------|---------------------|
| $0.02 | Use only Creative Thinker | 2 evaluations |
| $0.05 | Mixed strategy possible | 3-5 evaluations |
| $0.10+ | Full flexibility | 10+ evaluations |

## 🧠 Agent Intelligence

The agent makes strategic decisions considering:

1. **Dynamic Question Analysis**: Agent analyzes each question to understand what type of expertise is required
2. **Agent Card Specialty Matching**: Matches question requirements to available agent card specialties without rigid categorization
3. **Budget Conservation**: Early training uses cheaper agent cards, later phases invest in quality
4. **Performance History**: Agent cards with better past performance get preference
5. **Cost-Effectiveness**: Balance between agent card quality and budget constraints for each specific question

### Intelligent Selection Algorithm

```python
async def _agent_select_judges(self, question: str) -> JudgeSelection:
    # 1. Agent analyzes question directly (no pre-categorization)
    # 2. Get agent card performance stats
    judge_stats = self._get_judge_performance_stats()

    # 3. AI agent makes strategic decision with full context
    selection_response = await self.server.chat_completion(
        messages=[
            {"role": "system", "content": "Analyze each question and match to agent card specialties..."},
            {"role": "user", "content": selection_prompt}
        ]
    )

    # 4. Validate and execute selection
    return validated_selection
```

The agent receives detailed information about each agent card's specialties, past performance, pricing, and makes intelligent decisions about which combination of agent cards will provide the best evaluation for each specific question.

## 📈 Metrics and Monitoring

### Weights & Biases Integration

The environment logs comprehensive metrics:

**Budget Metrics**:
- Current balance and total spent
- Budget utilization percentage
- Average cost per evaluation
- Per-agent-card spending breakdown

**Agent Card Performance**:
- Individual agent card scoring history
- Agent satisfaction ratings
- Selection frequency analysis
- Performance consistency tracking

**Agent Decisions**:
- Agent card selection reasoning
- Question analysis by the agent
- Strategic decision patterns

### Example Wandb Dashboard

```
budget/current_balance: 0.85
budget/total_spent: 0.15
budget/utilization_percent: 15.0
payments/success_rate: 1.0
judge_performance/technical_expert_avg_score: 0.85
selection_frequency/creative_thinker_percent: 60.0
```

## 🔧 Testing

The system includes comprehensive testing for:
- Agent card initialization and metadata
- Agent question analysis (no rigid categorization)
- Budget tracking functionality
- Strategic selection logic
- Pricing strategy analysis

## 🌐 Blockchain Integration

### Wallet Configuration

**⚠️ SECURITY IMPORTANT**: Never commit `secrets.json` with real private keys to version control!

1. **Copy the template**:
```bash
cp secrets.json.template secrets.json
```

2. **Update `secrets.json` with your wallet credentials**

3. **Add to `.gitignore`**:
```bash
echo "src/environments/pay_to_play/secrets.json" >> .gitignore
```

### Security Best Practices

- **Use separate wallets** for each agent card to maintain clear accounting
- **Test with small amounts** first in testing mode
- **Monitor wallet balances** regularly
- **Use hardware wallets** for production deployments
- **Never share private keys** or commit them to version control
- **Consider using environment variables** for production deployments

### Real Payments

Set `testing_mode=False` for real USDC payments on Base blockchain:
- Requires funded agent wallet with USDC
- Payments are permanent and irreversible
- Monitor gas costs for small transactions
- Ensure all agent card wallets are properly configured

## 🎖️ Key Features

✅ **Multiple Specialized Agent Cards**: Different expertise areas and pricing tiers
✅ **Intelligent Agent Selection**: AI-driven agent card selection with dynamic question analysis
✅ **Budget Awareness**: Real economic constraints drive efficient learning
✅ **Performance Tracking**: Historical data informs future decisions
✅ **Blockchain Integration**: Real USDC payments on Base network
✅ **Comprehensive Monitoring**: Detailed metrics and decision analysis
✅ **Fallback Mechanisms**: Robust handling of budget constraints
✅ **Testing Framework**: Simulation mode for development and testing

## 🚧 Future Enhancements

- **Dynamic Pricing**: Agent card prices adjust based on demand and performance
- **Agent Card Reputation System**: Community-driven agent card quality ratings
- **Multi-Round Evaluation**: Iterative feedback and improvement cycles
- **Agent Card Specialization**: More granular specialty categories
- **Economic Incentives**: Reward mechanisms for high-performing agent cards

## 📋 Requirements

- Python 3.11+
- Web3.py for blockchain interaction
- AtroposLib for base environment
- USDC on Base blockchain (for real payments)
- Weights & Biases account (optional, for monitoring)

## 🤝 Contributing

When adding new agent cards or features:

1. Update `AgentCardSpecialty` enum for new specialties
2. Add agent card configuration in `agent_cards_config.py`
3. Update wallet configuration in `secrets.json`
4. Add appropriate test cases
5. Update documentation

## 📚 References

This environment builds upon recent advances in reinforcement learning from AI feedback:

- **RLAIF vs. RLHF**: Lee, H., et al. (2023). "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." *arXiv preprint arXiv:2309.00267*. [https://arxiv.org/abs/2309.00267](https://arxiv.org/abs/2309.00267)

- **Mixture of Judges**: Xu, T., et al. (2024). "The Perfect Blend: Redefining RLHF with Mixture of Judges." *arXiv preprint arXiv:2409.20370*. [https://arxiv.org/abs/2409.20370](https://arxiv.org/abs/2409.20370)
