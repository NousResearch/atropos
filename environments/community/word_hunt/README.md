# Word Hunt Environment

This Atropos environment is designed to train language models to play **Word Hunt**, a game where the goal is to trace through a 4x4 grid of letters to create as many words as possible within the time limit.

Word Hunt combines multiple cognitive challenges: spatial reasoning (tracing paths through the grid), vocabulary knowledge (recognizing valid words), and strategic optimization (prioritizing longer, higher-scoring words within token output constraints).


## Game Rules and Sample Prompt

The model receives a 4x4 grid of letters and must find valid English words by tracing through adjacent letters (including diagonally). The key rules are:

- Words must be **at least 3 letters long**
- Letters must be **adjacent** (horizontally, vertically, or diagonally)
- Each letter can only be **used once per word**
- The board **does not wrap around** (edges are not connected)
- Only **valid English words** count toward the score
- **Duplicate words** don't count for extra points

**Sample Prompt:**
```
Find English words on this 4x4 letter grid to maximize your score.
Longer words are worth more points. You must adhere to the following rules:

• Words must be AT LEAST 3 letters long and have to be formed by connecting
  adjacent letters on the board (including diagonally).
• The board does not wrap around; letters on opposite edges are not considered adjacent.
• The whole word must have an adjacent path through it
• Each letter can only be used once per word.
• The word must be a valid word in the English language
• Making the same word in multiple ways does not count for extra points -
  each unique word only counts once.

Provide your answer as a comma-separated list, like this: WORD, ANOTHER, EXAMPLE

Scoring: 3-letter: 100pts, 4-letter: 400pts, 5-letter: 800pts

Board:
G O E L
M I I E
N G M C
B S D T

Found words:
```

## Features

- **High-Performance Solver:** Uses a Trie-based recursive backtracking algorithm to efficiently find all possible words on any given board. This allows us to measure model performance against the optimal solution and provide accurate scoring.

- **Scoring:** Points are awarded based on word length following the official Word Hunt scoring system:
  - 3-letter words: 100 points
  - 4-letter words: 400 points
  - 5-letter words: 800 points
  - 6+ letter words: 1400 + (400 × (length - 6)) points
  - These scoring rules are based on the version of Word Hunt in GamePidgeon, the iOS app

  The default maximum output tokens a response will produce is 100 tokens, which simulates the time pressure and encourages strategic resource allocation.


- **Reward Signal:** The environment provides normalized rewards (0-1 range) based on the model's score divided by the maximum possible score on each board, ensuring consistent training signals across different board difficulties.


## Setup and Dependencies

The environment relies on a dictionary file to validate words. You'll need to download the dictionary file before running the environment:

1. **Download the dictionary file:**
   Manually download `Dictionary.txt` from: https://github.com/Aboozle1/wordhuntsolver/blob/main/Dictionary.txt

2. **Place it in the correct location:**
   Save the file as: `environments/community/word_hunt/Dictionary.txt`

If you wish to use a different dictionary, you can change the `dictionary_path` in `word_hunt_config.py`.

## How to Run

You can run the environment and generate a small sample of rollouts using the `process` command. This is a quick way to test that the environment is working correctly. The results will be saved to a `.jsonl` file and a corresponding `.html` report for easy viewing.

```bash
python environments/community/word_hunt/word_hunt_env.py process \
  --env.total_steps 2 \
  --env.data_path_to_save_groups word_hunt_sample_rollouts.jsonl \
  --env.use_wandb false
```

This command will:
- Run the model for **2 steps** (i.e., process 2 groups of boards).
- Save the results to `word_hunt_sample_rollouts.jsonl` and `word_hunt_sample_rollouts.html`.

You can also override other parameters, like the model endpoint:
```bash
python environments/community/word_hunt/word_hunt_env.py process \
  --openai.base_url <YOUR_API_BASE_URL> \
  --openai.api_key <YOUR_API_KEY> \
  --openai.model_name <YOUR_MODEL_NAME> \
  --env.total_steps 2 \
  --env.use_wandb false
```

### Full Example with Model Configuration

Here is a complete, real-world example that specifies the model endpoint and API key. This is useful when you want to target a specific model that is not set as your default.

```bash
python3 environments/community/word_hunt/word_hunt_env.py process \
  --openai.base_url https://inference-api.nousresearch.com/v1 \
  --openai.api_key <YOUR_NOUS_API_KEY> \
  --openai.model_name DeepHermes-3-Llama-3-8B-Preview \
  --env.data_path_to_save_groups word_hunt_rollouts.jsonl \
  --env.use_wandb false \
  --env.total_steps 2
```

## Configuration

The primary configuration for this environment is handled in `environments/community/word_hunt/word_hunt_config.py`. Key options include:

- `prompt_style`: How the board is presented to the model. Can be `grid_visual`, `text_description`, or `both`.
- `include_instructions`: Whether to include the game rules in the prompt.
- `include_scoring_info`: Whether to show the scoring system in the prompt.
- `board_size`: The dimensions of the game board (default is 4).
- `dictionary_path`: The file path to the dictionary used for word validation.
- `letter_frequencies`: The probability distribution used for generating the letter grid.
- `scoring_system`: A dictionary mapping word lengths to point values.
- `max_tokens_per_game`: The maximum number of tokens the model is allowed to generate for its response. This simulates a "time limit" and can be adjusted to make the task easier or harder (default: 100).
