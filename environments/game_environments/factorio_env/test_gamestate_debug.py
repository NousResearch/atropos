#!/usr/bin/env python3
"""
Debug test for GameState.from_instance() hanging issue.
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fle"))

import fle
from fle.env import FactorioInstance
from fle.env.gym_env.environment import FactorioGymEnv
from fle.eval.tasks import TaskFactory
from fle.env.gym_env.registry import get_environment_info
from fle.commons.models.game_state import GameState
from fle.env.gym_env.action import Action


def main():
    print("Testing GameState.from_instance() and action execution...\n")
    
    # Step 1: Create instance
    print("Step 1: Creating FactorioInstance...")
    instance = FactorioInstance(
        address="localhost",
        tcp_port=27000,
        fast=True,
        num_agents=1
    )
    print(f"Instance created successfully\n")
    
    # Step 2: Create task
    print("Step 2: Creating task...")
    task_name = "iron_ore_throughput"
    env_info = get_environment_info(task_name)
    task_path = env_info["task_config_path"]
    task = TaskFactory.create_task(task_path)
    task.setup(instance)
    print(f"Task created successfully\n")
    
    # Step 3: Create gym environment
    print("Step 3: Creating gym environment...")
    env = FactorioGymEnv(instance=instance, task=task)
    print(f"Gym environment created\n")
    
    # Step 4: Reset environment
    print("Step 4: Resetting environment...")
    obs, info = env.reset(options={"game_state": None})
    print(f"Environment reset successfully")
    print(f"Observation keys: {obs.keys()}\n")
    
    # Step 5: Test GameState.from_instance() multiple times
    print("Step 5: Testing GameState.from_instance() calls...")
    
    for i in range(3):
        print(f"\nTest {i+1}:")
        print(f"  Calling GameState.from_instance()...")
        start_time = time.time()
        
        try:
            game_state = GameState.from_instance(env.instance)
            elapsed = time.time() - start_time
            print(f"  ✅ GameState obtained in {elapsed:.3f}s")
            print(f"  Game state type: {type(game_state)}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ Failed after {elapsed:.3f}s: {e}")
    
    # Step 6: Test specific actions
    print("\n\nStep 6: Testing specific actions...")
    
    test_actions = [
        ("nearest", 'print(nearest(type=Resource.IronOre))'),
    ]
    
    for action_name, code in test_actions:
        print(f"\nTesting action: {action_name}")
        print(f"  Code: {code}")
        
        try:
            print(f"  Getting GameState...")
            start_time = time.time()
            current_game_state = GameState.from_instance(env.instance)
            elapsed = time.time() - start_time
            print(f"  GameState obtained in {elapsed:.3f}s")
            
            print(f"  Creating Action...")
            action = Action(agent_idx=0, game_state=current_game_state, code=code)
            
            print(f"  Executing action...")
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = time.time() - start_time
            
            print(f"  ✅ Action executed in {elapsed:.3f}s")
            print(f"  Reward: {reward}")
            print(f"  Result: {info.get('result', 'No result')[:100] if info else 'No info'}")
            
        except Exception as e:
            print(f"  ❌ Action failed: {e}")
    
    # Step 7: Close
    print("\n\nStep 7: Closing environment...")
    env.close()
    print("Environment closed.")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()