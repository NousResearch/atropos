# Gymnasium Environments

This directory contains RL environments based on OpenAI's Gymnasium.

## Blackjack Environment

A reinforcement learning environment for Blackjack that uses a best-of-n approach to select actions. The model is trained to make function calls with proper formatting to choose between "hit" and "stick" actions. The training objective here is decisive decision making, and to experiment with an environment where even playing correct strategy can result in a loss (perfect Blackjack strategy results in a statistical edge over time, but never guarantess a win). Will the LLM learn the correct strategy in an emergent fashion after enough games, from the cumulative effect of all rewards?

The environment implements a strategy for multi-step decision-making, aiming to refine credit assignment. This approach is designed to be compatible with GRPO-style RL training and includes the following key components:

1.  **Alternative Generation (Best-of-N):** At each turn, the LLM generates `G` (configurable via `group_size` in the YAML configuration) alternative responses. Each response typically includes a "thinking" phase (e.g., enclosed in `<think> </think>` tags) followed by a structured tool call ("hit" or "stick").
2.  **Monte Carlo (MC) Value Estimation:** To evaluate game states, the environment employs MC rollouts, inspired by methods like VinePPO to avoid a separate learned value network.
    *   For the current state `s_t`, `K` (configurable via `mc_samples`) full game playouts are simulated using the current LLM policy. The average total environment reward from these rollouts is used as the estimate `V(s_t)`.
    *   Similarly, for each of the `G` alternatives leading to potential next states `s'_i`, the value `V(s'_i)` is estimated.
3.  **Advantage Calculation:** For each alternative `i`, an advantage `A_i = R_combined_i + V(s'_i) - V(s_t)` is computed. `R_combined_i` includes both the immediate environment reward from taking that alternative and any format-based rewards (e.g., for correct tool use and thinking tags). A discount factor of γ=1 is used.
4.  **Action Selection:** The alternative with the highest calculated advantage is chosen as the canonical action to advance the game trajectory.
5.  **Data for Trainer:** Detailed information for all `G` alternatives (including their messages, tokenized representations, and calculated advantage scores) is collected at each step, providing rich data for policy training.
6.  **Handling Long Sequences:** Necessary to prevent blowing up sequence length limits on RL trainers. Blackjack episodes are typically quite short, but can go for enough turns for it to be a problem, especially with excessively long thinking blocks
    *   **Thinking Blocks:** Long "thinking" blocks are permitted. To manage history length for the LLM's context, these thoughts are truncated in messages representing past turns. Currently this is just taking the last paragraph of text (as this often contains the LLMs final conclusions), in other examples will move to a summarisation with the LLM itself
    *   **Token Limits:** The environment actively manages the overall token length of conversation histories by truncating older message pairs (system prompt is preserved, last agent/environment exchange is preserved) to ensure sequences fit within the trainer's maximum token limits.

### Features

- Uses Gymnasium's Blackjack-v1 environment
- Teaches models to use tool-calling format for actions
- Combines environment rewards with format rewards for better training
- Configurable via YAML files

## Usage

### Running with Default Configuration

To run the Blackjack environment with the default configuration:

```bash
python environments/game_environments/gymnasium/blackjack_local_server.py
```

This will use the default configuration from `configs/envs/blackjack.yaml`.

### Custom Configuration

You can specify a custom configuration file:

```bash
python environments/game_environments/gymnasium/blackjack_local_server.py --config my_custom_config
```

The `--config` parameter can be:

1. A name (without `.yaml` extension) which will be looked up in `configs/envs/`
2. A relative or absolute path to a YAML file

For example:
```bash
# Using a config in configs/
python environments/game_environments/gymnasium/blackjack_local_server.py --config blackjack_default

# Using a config with full path
python environments/game_environments/gymnasium/blackjack_local_server.py --config /path/to/my/config.yaml
```

## Configuration

The environment's behavior is controlled via YAML configuration files (e.g., `environments/game_environments/gymnasium/configs/blackjack_default.yaml`).

Key aspects controlled by the configuration include:
*   LLM model parameters (e.g., `temperature`, `top_p`).
*   Training strategy parameters (`group_size` for N alternatives, `mc_samples` for value estimation, `max_turns`).
*   Reward function setup and weights (`reward_functions`, `format_reward_weight`, `environment_reward_weight`).
*   Tokenization (`tokenizer_name`) and sequence length limits (`max_token_length`, `max_think_chars_history`).
*   Server endpoints for the LLM.

## Note:
The Monte Carlo sampling greatly increases the amount of calls to the policy model (ie, vLLM, sglang, whatever LLM server is being used). So it's highly suggested to allocate extra nodes to it so this doesn't become a bottleneck

**Example Server Configuration in YAML:**

The `server_configs` section in the YAML directly specifies parameters for the LLM API, including the model name, base URL, and API key. The `api_key` should be set directly in the YAML or handled by the server/client using it if a placeholder like "x" is used (eg, add a dotenv file and read the key from the environment).

```yaml
# Server configuration
server_configs:
  - model_name: "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
    base_url: "http://localhost:9004/v1"
    api_key: "x" # Or your actual API key
    num_requests_for_eval: 256
```
