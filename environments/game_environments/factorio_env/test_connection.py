#!/usr/bin/env python3
"""
Simple test to verify Factorio connection and single action.
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle"))

import fle
from fle.env import FactorioInstance
from fle.env.gym_env.environment import FactorioGymEnv
from fle.eval.tasks import TaskFactory
from fle.env.gym_env.registry import get_environment_info
from fle.commons.models.game_state import GameState
from fle.env.gym_env.action import Action


def main():
    print("Testing Factorio connection and single action...\n")
    
    # Step 1: Create instance (should see "Connected to localhost client at tcp/27000")
    print("Step 1: Creating FactorioInstance...")
    instance = FactorioInstance(
        address="localhost",
        tcp_port=27000,
        fast=True,
        num_agents=1
    )
    print(f"Instance created: {instance}\n")
    
    # Step 2: Create task
    print("Step 2: Creating task...")
    task_name = "iron_ore_throughput"
    env_info = get_environment_info(task_name)
    task_path = env_info["task_config_path"]
    task = TaskFactory.create_task(task_path)
    task.setup(instance)
    print(f"Task created: {task}\n")
    
    # Step 3: Create gym environment
    print("Step 3: Creating gym environment...")
    env = FactorioGymEnv(instance=instance, task=task)
    print(f"Gym environment created: {env}\n")
    
    # Step 4: Reset environment
    print("Step 4: Resetting environment...")
    obs, info = env.reset(options={"game_state": None})
    print(f"Environment reset!")
    print(f"Observation keys: {obs.keys() if isinstance(obs, dict) else type(obs)}")
    print(f"Info: {info}\n")
    
    # Step 5: Take a single action
    print("Step 5: Taking a single action (inspect_inventory)...")
    code = "print(inspect_inventory())"
    current_game_state = GameState.from_instance(env.instance)
    action = Action(agent_idx=0, game_state=current_game_state, code=code)
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action executed!")
    print(f"Reward: {reward}")
    print(f"Result: {info.get('result', 'No result') if info else 'No info'}")
    print(f"Done: {terminated or truncated}\n")
    
    # Step 6: Close
    print("Step 6: Closing environment...")
    env.close()
    print("Environment closed.")
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()