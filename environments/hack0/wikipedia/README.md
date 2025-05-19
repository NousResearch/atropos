# Wikipedia Article Creator Environment

This environment trains an LLM to research and create Wikipedia-style articles on arbitrary topics using web search and content extraction tools.

## Features

- Multi-step research process with web search and content extraction
- Factual accuracy evaluation against real Wikipedia articles
- Wandb logging for detailed analysis of article quality
- Configurable parameters for controlling article generation

## Article Evaluation

The environment now includes a factual accuracy evaluation system that:

1. Compares AI-generated articles against reference Wikipedia articles
2. Provides line-by-line assessment of factual accuracy
3. Categorizes statements as CORRECT, INCORRECT, or UNKNOWN
4. Calculates accuracy metrics and integrates them into the scoring function

## Setup and Configuration

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

The environment requires the following environment variables:

```bash
# Required for article research
export TAVILY_API_KEY="your_tavily_api_key"  # For web search and content extraction

# Required for LLM access
export OPENAI_API_KEY="your_openai_api_key"  # For OpenAI models

# Required for factual accuracy evaluation
export OPENAI_API_KEY="your_openai_api_key"  # Used for article evaluation
```

## Usage

### Training

To run training with this environment:

```bash
python -m atroposlib.cli.dpo --env-module "environments.hack0.wikipedia.wikipedia_article_creator" --wandb-mode online
```

### Evaluation

You can evaluate a model's ability to create accurate Wikipedia articles:

```bash
python -m atroposlib.cli.sft --eval-only --env-module "environments.hack0.wikipedia.wikipedia_article_creator" 
```

## Evaluation Metrics

The evaluation process generates the following metrics:

### Quality Metrics

- **Structure Score**: Quality of article organization and section structure (0-1)
- **Comprehensiveness**: Coverage of important aspects of the topic (0-1)
- **Fact Usage**: Effective incorporation of researched facts (0-1)
- **Overall Quality**: Combined score of the above metrics (0-1)

### Factual Accuracy Metrics

- **Correct Statements**: Percentage and count of statements verified as factually correct
- **Incorrect Statements**: Percentage and count of statements that contradict the reference
- **Unknown Statements**: Percentage and count of statements that can't be verified
- **Factual Accuracy Score**: Net accuracy score in range [-1, 1]

### Combined Score

- **Overall Article Score**: Comprehensive metric combining both quality and factual accuracy in range [-1, 1]

This combined metric provides the best representation of article quality, as it balances structural elements with factual correctness. [This example run](https://wandb.ai/niemerg-chicago/atropos-environments_hack0_wikipedia/runs/cddj4yyy) demonstrates how these metrics are tracked and visualized during training.

When evaluating models, the `train/overall_article_score` is the key metric to focus on, as it represents the total effectiveness of the model in producing high-quality, factually accurate articles.

## Configuration

Key configuration parameters:

- `max_steps`: Maximum research steps per article
- `temperature`: Sampling temperature for article generation
- `eval_topics`: Number of topics for evaluation
- `tool_timeout`: Timeout for web search/extraction tools
- `thinking_active`: Enable thinking tags for model reasoning
- `max_article_tokens`: Maximum length of final article

## Article Evaluation System

The factual accuracy evaluation system uses OpenAI models to:

1. Process AI-generated articles line by line
2. Compare each statement with reference Wikipedia content
3. Provide detailed analysis of factual accuracy
4. Calculate an accuracy score that influences the overall rating

The evaluation results are integrated with wandb logging, providing rich insights into article quality and accuracy.