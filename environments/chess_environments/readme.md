# ♟️ Chess LLM Training Environments

**Author:** EaswarGn

---

## 📝 Overview

This repository contains three custom reinforcement learning environments designed to fine-tune Large Language Models (LLMs) to understand and solve chess puzzles.

The main environment trains models to solve chess puzzles with Stockfish-aligned rewards.

Two pre-training environments progressively teach the model to:

1. Play legal moves ⚖️
2. Visualize multiple moves ahead 🔮

This ensures a strong foundation before puzzle-solving fine-tuning.

Think of it as a **curriculum for chess-trained LLMs**:

> Learn the rules ⚖️ → Learn to visualize ahead 🔮 → Solve puzzles like a grandmaster 🏆

---

## 🏗️ Environment Structure

### Core Components

- **`chess_rules_env.py`** – teaches the LLM to generate legal chess moves.
- **`chess_board_visualization_env.py`** – helps the LLM visualize moves ahead (lookahead training).
- **`chess_puzzle_env.py`** – the main environment that fine-tunes the model to solve chess puzzles and compare its solutions with Stockfish using a novel reward system.

---

## 🧩 Environments in Detail

### 1️⃣ Chess Rules Environment (`chess_rules_env`)

**📚 Goal:** Teach the LLM to play legal chess moves.

- Ensures model outputs valid UCI moves.
- Ideal as a first step before tackling puzzles.
- Model takes in board position and then outputs all possible legal moves, gets more reward for the more moves that it gets correct
- Uses my custom hugging face dataset to generate tough positions: https://huggingface.co/datasets/codingmonster1234/chess_puzzles_dataset

---

### 2️⃣ Chess Board Visualization Environment (`chess_board_visualization_env`)

**🔮 Goal:** Train the LLM to visualize multiple moves ahead, improving its internal chess “lookahead” ability.

- Provides scenarios requiring multi-move reasoning.
- Builds intuition for complex board states.
- Model recieves a sequence of moves and then must internally visualize the final board state and output it correctly. The reward is proportional to how close the model is to actual final position.
- Uses the custom ChessInstruct dataset: https://huggingface.co/datasets/Thytu/ChessInstruct

---

### 3️⃣ Chess Puzzle Environment (`chess_puzzle_env`)

**♟️ Goal:** Fine-tune the model to solve real chess puzzles.

- Compares the model’s move sequence to Stockfish’s evaluation.
- Uses a novel reward strategy to align the LLM’s reasoning with engine-level accuracy.
- Supports step-by-step reasoning and reward shaping for deeper insights.
- Uses custom puzzle dataset curated from lichess.org: https://huggingface.co/datasets/codingmonster1234/chess_puzzles_dataset

## 🏆 Stockfish-Based Reward Function

This environment uses a **novel reward function** to evaluate a model's predicted chess moves relative to the ground truth moves. Unlike a simple correct/incorrect metric, the reward measures **closeness to Stockfish evaluation**, encouraging the model to generate high-quality moves that align with expert evaluations.

---

### **How It Works**

1. **Inputs**
   - `initial_fen`: The starting board position in FEN format.
   - `pred_moves`: A sequence of predicted moves in UCI format.
   - `correct_moves`: The reference moves for the puzzle in UCI format.
   - `stockfish_path`: Path to the Stockfish engine.

2. **Step-wise Evaluation**
   - Each predicted move is played on a copy of the board.
   - The board is evaluated using Stockfish at a fixed depth (e.g., depth=12).
   - The predicted evaluation is compared with the evaluation of the correct move sequence.

3. **Aggregation**
   - Step rewards are summed and normalized over the total number of moves.
   - A final non-linear transformation is applied to keep the reward in `[0,1]`.


---

## 📊 Training Mechanics

**Curriculum Learning:** Recommended order
chess_rules_env → chess_board_visualization_env → chess_puzzle_env




**Reward Shaping:**

- **Rules env:** Rewards legal move generation. ✅
- **Visualization env:** Rewards correct multi-move predictions. 🔭
- **Puzzle env:** Rewards closeness to Stockfish evaluations. 🏅

**Evaluation Metrics:**

- Move legality ✅
- Lookahead accuracy 🔭
- Puzzle solution quality vs. Stockfish 🏅

## 🔬 Requirements
Only the chess library, pip install chess and atropos dependencies. Also, must download stockfish from: https://stockfishchess.org/download/


## 🔬 Research Applications

These environments can be used to explore key research questions in chess AI and learning:

1. **Chess Reasoning and Lookahead**
   - Study how LLMs develop multi-step planning and board visualization skills.
   - Analyze the effect of progressive training (rules → visualization → puzzles) on move prediction and strategy.

2. **Puzzle-Solving and Engine Alignment**
   - Investigate how closely LLMs can match Stockfish evaluations when solving chess puzzles.
   - Explore reward shaping strategies to align model reasoning with engine-level accuracy.

3. **Explainable Chess AI**
   - Examine step-by-step reasoning generated by LLMs.
   - Compare model move explanations to human or engine strategies for interpretability in chess decision-making.
