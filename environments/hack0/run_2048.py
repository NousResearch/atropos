import argparse
import random
import asyncio
import json
from typing import Dict, Any, List, Tuple, Optional, Literal

# Direct import instead of the try/except block
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game_2048 import Game2048
from env_2048 import Environment2048

class RandomAgent:
    """A simple agent that makes random moves in the 2048 game."""
    
    def __init__(self):
        self.moves = ['left', 'right', 'up', 'down']
    
    async def act(self, observation: Dict[str, Any], prompt: str) -> str:
        """Generate a random move."""
        move = random.choice(self.moves)
        return f"<move>{move}</move>"

class HeuristicAgent:
    """A simple heuristic agent that tries to keep the highest numbers in a corner."""
    
    def __init__(self):
        self.moves = ['left', 'right', 'up', 'down']
        
    async def act(self, observation: Dict[str, Any], prompt: str) -> str:
        """Generate a move based on a simple heuristic."""
        # Convert board to numpy for easier manipulation
        import numpy as np
        board = np.array(observation['board'])
        
        # Try each move and see which one gives the highest score
        best_score = -1
        best_move = random.choice(self.moves)  # Fallback to random
        
        # Define weights - prefer larger values in the corner and along the edge
        weights = np.array([
            [4, 3, 2, 1],
            [3, 2, 1, 0],
            [2, 1, 0, 0],
            [1, 0, 0, 0]
        ])
        
        for move in self.moves:
            # Simulate the move
            env_copy = Environment2048(winning_value=observation['winning_value'])
            env_copy.game.board = np.array(observation['board'])
            env_copy.game.score = observation['score']
            env_copy.game.moves = observation['moves']
            
            _, score_added, changed = env_copy.game.move(move)
            
            if not changed:
                continue
                
            # Calculate a heuristic score based on:
            # 1. The actual score added
            # 2. How well the board aligns with our weight matrix (highest values in corner)
            board_after = env_copy.game.board
            
            # Score for having high values in the corner
            position_score = np.sum(board_after * weights)
            
            # Score for having adjacent same values
            merge_potential = 0
            for r in range(4):
                for c in range(3):
                    if board_after[r, c] != 0 and board_after[r, c] == board_after[r, c + 1]:
                        merge_potential += board_after[r, c]
            
            for r in range(3):
                for c in range(4):
                    if board_after[r, c] != 0 and board_after[r, c] == board_after[r + 1, c]:
                        merge_potential += board_after[r, c]
            
            # Avoid empty moves
            if not changed:
                continue
                
            total_score = position_score + merge_potential + score_added
            
            if total_score > best_score:
                best_score = total_score
                best_move = move
        
        return f"<move>{best_move}</move>"

async def run_episode(env: Environment2048, agent, max_steps: int = 1000) -> Dict[str, Any]:
    """Run a single episode."""
    observation, prompt = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        # Get action from agent
        action = await agent.act(observation, prompt)
        
        # Take a step in the environment
        observation, prompt, done, reward, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        # Print game state
        print(env.render())
        print(f"Step: {steps}, Action: {action}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
        
        if done:
            print("Episode finished!")
            if info.get('win', False):
                print("Won the game!")
            else:
                print(f"Game over! Max tile: {observation['max_tile']}")
    
    # Return episode statistics
    return {
        'steps': steps,
        'total_reward': total_reward,
        'max_tile': observation['max_tile'],
        'score': observation['score'],
        'win': observation['max_tile'] >= env.winning_value
    }

async def main():
    parser = argparse.ArgumentParser(description='Run the 2048 game environment.')
    parser.add_argument('--agent', type=str, default='heuristic', choices=['random', 'heuristic'],
                        help='Agent type to use.')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run.')
    parser.add_argument('--winning-value', type=int, default=2048, help='Winning tile value.')
    args = parser.parse_args()
    
    # Create the environment
    env = Environment2048(winning_value=args.winning_value)
    
    # Create the agent
    if args.agent == 'random':
        agent = RandomAgent()
    else:
        agent = HeuristicAgent()
    
    # Run episodes
    episode_stats = []
    for i in range(args.episodes):
        print(f"\nEpisode {i+1}/{args.episodes}")
        stats = await run_episode(env, agent)
        episode_stats.append(stats)
    
    # Print summary
    print("\n===== Summary =====")
    wins = sum(1 for stat in episode_stats if stat['win'])
    avg_steps = sum(stat['steps'] for stat in episode_stats) / args.episodes
    avg_reward = sum(stat['total_reward'] for stat in episode_stats) / args.episodes
    avg_score = sum(stat['score'] for stat in episode_stats) / args.episodes
    max_tiles = [stat['max_tile'] for stat in episode_stats]
    
    print(f"Episodes: {args.episodes}")
    print(f"Wins: {wins} ({wins/args.episodes*100:.1f}%)")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average score: {avg_score:.1f}")
    print(f"Max tiles achieved: {max_tiles}")

if __name__ == "__main__":
    asyncio.run(main()) 