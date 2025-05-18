# Wikipedia Article Creator Environment

This environment allows an LLM to research and create Wikipedia-style articles on arbitrary topics using web search and content extraction tools. The model goes through a multi-step research process, searching for information and extracting content from webpages, before finally producing a comprehensive article.

## Features

- Multi-step research process with tool-calling
- Web search and content extraction via Tavily API
- Support for both OpenAI models and local models
- Evaluation of article quality based on structure, comprehensiveness, and fact usage
- Environment variable configuration through .env files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NousResearch/atropos.git
cd atropos
```

2. Install dependencies:
```bash
pip install -e .  # Install Atropos
pip install -r environments/hack0/wikipedia/requirements.txt  # Install Wikipedia environment dependencies
```

3. Set up API keys in a `.env` file:
```bash
cp environments/hack0/wikipedia/.env.template .env
# Edit the .env file with your API keys
```

## Configuration

### Environment Variables

The following environment variables can be set in your `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key (required for using OpenAI models)
- `TAVILY_API_KEY`: Your Tavily API key (required for web search and content extraction)
- `MODEL_NAME`: Model to use (defaults to "gpt-4o")
- `MAX_STEPS`: Maximum research steps (defaults to 10)
- `TEMPERATURE`: Temperature for model generation (defaults to 0.7)

### API Keys

- **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get your API key from [Tavily](https://tavily.com/)

## Usage

### Running with OpenAI Models

You can use the included runner script to test the environment with OpenAI models:

```bash
# Run with default settings (topic: "Climate change in Antarctica", model: gpt-4o)
python environments/hack0/wikipedia/run_with_openai.py

# Specify a custom topic and model
python environments/hack0/wikipedia/run_with_openai.py --topic "History of quantum computing" --model "gpt-3.5-turbo" --max-steps 8
```

### Integration with Atropos Training

1. Start the API server:
```bash
run-api
```

2. Start the Wikipedia article creator environment:
```bash
python environments/hack0/wikipedia/wikipedia_article_creator.py serve
```

### Generating Dataset for Training

You can generate a dataset for SFT training:

```bash
run-api & # Start API server in background
python environments/hack0/wikipedia/wikipedia_article_creator.py serve & # Start environment in background
atropos-sft-gen path/to/output.jsonl --tokenizer gpt2  # For OpenAI models
```

## How It Works

1. The environment presents a topic to the model (e.g., "The history of artificial intelligence")
2. The model uses web_search to find information about the topic
3. The model can use visit_page to extract content from specific websites
4. After gathering sufficient information, the model produces a final Wikipedia-style article
5. The environment evaluates the article's quality based on structure, comprehensiveness, and fact usage

## Implementation Details

The environment consists of the following components:

- `WikipediaArticleCreatorEnv`: Main environment class
- `EpisodeState`: Tracks state across multiple interaction steps
- `TavilySearchTool`: Tool for web search
- `TavilyExtractTool`: Tool for extracting content from websites

The research process is managed by the `_next_step` method, which:
1. Gets the current model response
2. Checks if it contains the final article
3. Extracts and executes tool calls if not
4. Updates the episode state and adds tool results to the conversation

## Customization

You can customize the environment by:

- Modifying the system prompt in `SYSTEM_PROMPT`
- Adjusting the configuration parameters in `config_init`
- Implementing custom article evaluation metrics in `_assess_article_quality`

## References

- [Atropos Documentation](https://github.com/NousResearch/atropos/README.md)
- [Multi-Step Rollout Plan](multi_step_rollout_plan.md)
- [OpenAI Integration Plan](multi_step_rollout_plan_openai.md)