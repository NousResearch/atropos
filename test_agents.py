import asyncio
import os
from typing import List, Tuple, Optional, Dict, Union, Any

from dotenv import load_dotenv
# from transformers import AutoTokenizer # No longer directly used here, BaseEnv handles it
from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    Item,
    ScoredDataItem,
    ScoredDataGroup,
)
from atroposlib.envs.server_handling.server_manager import (
    APIServerConfig,
    ServerBaseline,
)
from atroposlib.type_definitions import Message
from environments.agents.atropos_agent import AtroposAgent, AtroposAgentConfig
from environments.agents.atropos_rm import AtroposRM, AtroposRMConfig


class TestAtroposEnvConfig(BaseEnvConfig):
    """Configuration for the test environment."""

    # Override defaults if necessary for testing
    use_wandb: bool = False
    data_path_to_save_groups: Optional[str] = None
    group_size: int = 1 # For simplicity, one trajectory per "group"
    total_steps: int = 3 # Number of turns to simulate
    # Use a known tokenizer, can be overridden by AtroposAgent/RM if they have specific needs handled internally
    tokenizer_name: str = "gpt2" # A small, fast tokenizer for testing general flow
    ensure_scores_are_not_same: bool = False # Not critical for this test
    include_messages: bool = True # Useful for debugging


class TestAtroposEnv(BaseEnv):
    name = "TestAtroposIntegration"
    env_config_cls = TestAtroposEnvConfig

    def __init__(
        self,
        config: TestAtroposEnvConfig,
        server_configs: Union[ServerBaseline, List[APIServerConfig]],
    ):
        super().__init__(config, server_configs)
        self.agent: Optional[AtroposAgent] = None
        self.rm: Optional[AtroposRM] = None
        self.current_game_history: List[Message] = []
        # self.tokenizer is initialized by super().__init__ from BaseEnv

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[TestAtroposEnvConfig, Union[ServerBaseline, List[APIServerConfig]]]:
        load_dotenv()
        env_config = cls.env_config_cls()
        api_server_config = APIServerConfig(
            model_name="gpt-4.1-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        return env_config, [api_server_config]

    async def setup(self):
        """Initialize agents and game state."""
        print("Setting up TestAtroposEnv...")
        
        # Instantiate agent and RM configs
        agent_config = AtroposAgentConfig(
            thinking_enabled=True, # Example: enable thinking for agent
            memory_system_enabled=True, # Try with memory
            player_id_for_logging="TestAgent_01"
        )
        rm_config = AtroposRMConfig(
            thinking=True, # RM will show its thinking
            rm_id_for_logging="TestRM_01"
        )

        if not self.server or not self.server.servers:
            raise RuntimeError("Server not initialized correctly in BaseEnv.")
        
        # server_client is the actual APIServer instance from the list in ServerManager
        llm_server_client = self.server.servers[0]
        
        # Pass the tokenizer from BaseEnv (self.tokenizer)
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized in BaseEnv.")

        self.agent = AtroposAgent(
            server_client=llm_server_client, 
            tokenizer=self.tokenizer, 
            config=agent_config
        )
        self.rm = AtroposRM(
            server_client=llm_server_client, 
            tokenizer=self.tokenizer, 
            config=rm_config
        )

        self.agent.start_new_game_dialogue() # Resets agent's internal history and memory

        self.current_game_history = [
            Message(role="system", content=self.agent.system_prompt_content), # Use agent's actual system prompt
            Message(role="user", content="You stand in a dimly lit forest clearing. A narrow path leads north. To the east, you hear the sound of rushing water. A faint, rhythmic drumming seems to come from the west. What do you do?")
        ]
        print("TestAtroposEnv setup complete.")

    async def get_next_item(self) -> Item:
        """Provides the current game history as an item for processing."""
        # Ensure the history given to collect_trajectory is what the agent should see *now*.
        # For this test, self.current_game_history is the canonical history.
        # Since Item is Any, we pass a dictionary with the expected 'convo' key.
        return {"convo": list(self.current_game_history)} # Pass a copy as a dict

    async def collect_trajectory(
        self, item: Item # Item is Any, will be a dict like {"convo": [...]
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """
        Simulates one turn: agent action + RM evaluation.
        """
        if not self.agent or not self.rm:
            raise RuntimeError("Agents not initialized. Call setup() first.")

        # The item["convo"] is the history leading UP TO this turn's action.
        # The last message in item["convo"] should be the user/environment observation for the agent.
        history_for_agent_action = item["convo"] 
        
        if not history_for_agent_action or history_for_agent_action[-1]["role"] != "user":
            # This shouldn't happen if get_next_item and history updates are correct
            print("Warning: Last message in item.convo is not from user. Appending a generic observation.")
            new_obs_content = "The situation is unclear. Re-evaluate and act."
            history_for_agent_action.append(Message(role="user", content=new_obs_content))
        
        observation_content_for_agent = history_for_agent_action[-1]["content"]
        
        print(f"\n--- Turn Start ---")
        # print(f"History for Agent (from item.convo): {history_for_agent_action}")
        print(f"Observation for Agent (last user message): {observation_content_for_agent}")

        # 1. Agent generates action
        # generate_action expects the history *before* the current observation is appended as a user message.
        # It internally appends observation_content to game_history_window to form its prompt.
        
        # history_for_agent_action (from item.convo) is [..., previous_messages, current_observation_message]
        # observation_content_for_agent is the content of current_observation_message
        # So, the game_history_window for the agent should be all messages *before* current_observation_message
        history_base_for_agent = history_for_agent_action[:-1]

        action_text, history_after_action = await self.agent.generate_action(
            observation_content=observation_content_for_agent,
            game_history_window=history_base_for_agent, 
        )
        print(f"Agent Action: {action_text}")
        if self.agent.config.memory_system_enabled and self.agent.faiss_index:
             print(f"Agent has {self.agent.faiss_index.ntotal} memories.")

        # 2. RM evaluates the action in context
        # history_after_action already includes [..., system, ..., prev_user_obs, agent_action_just_taken]
        print(f"History for RM (includes action taken): {history_after_action}")
        # num_judgements_g is now num_judgements_g in AtroposRM
        judgements = await self.rm.generate_g_judgements(
            game_history_window=history_after_action, 
            num_judgements_g=1 # generate_g_judgements expects num_judgements_g
        )
        if not judgements:
            raise RuntimeError("Reward Model did not return any judgements.")
        
        # Output of generate_g_judgements is List[Tuple[Optional[str], Optional[float], Optional[str]]]
        # (raw_llm_response_content, q_value, thinking_block_content)
        raw_judgement, q_value, rm_thinking = judgements[0] 

        print(f"RM Raw Judgement: {raw_judgement}")
        if rm_thinking:
            print(f"RM Thinking: {rm_thinking}")
        print(f"RM Q-value: {q_value}")

        if q_value is None: # Handle case where Q-value parsing failed
            print("Warning: RM failed to parse Q-value. Using default 0.0.")
            q_value = 0.0

        # 3. Prepare ScoredDataItem
        encoded_action = self.tokenizer.encode(action_text) # self.tokenizer from BaseEnv
        scored_messages = history_after_action # Full history including the evaluated action

        # Update shared history for the next turn.
        # The environment would typically provide the *next* observation.
        # For this test, we append a generic follow-up user message.
        self.current_game_history = history_after_action + [
            Message(role="user", content=f"Narrator: Following your action ('{action_text}'), the world reacts. What is the new situation and what do you do now?")
        ]

        return (
            ScoredDataItem(
                tokens=encoded_action,
                masks=[1] * len(encoded_action), 
                scores=q_value, 
                messages=scored_messages,
                # advantages, ref_logprobs, group_overrides, overrides are None by default in ScoredDataItem
            ),
            [], # No new items to add to backlog
        )

    async def postprocess_histories(
        self,
        trajectories: Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]],
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """Passthrough for this test."""
        return trajectories

    async def evaluate(self, *args, **kwargs):
        """No-op evaluation for this test."""
        print("Evaluate called (no-op).")
        await asyncio.sleep(0.01) 
        pass

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """No-op wandb logging for this test if use_wandb is False."""
        if self.config.use_wandb:
            print(f"Wandb logging (no-op for test): {wandb_metrics if wandb_metrics else {}}")
        pass


async def main():
    print("Starting Atropos Agent and RM test...")
    
    env_config, server_configs_list = TestAtroposEnv.config_init()
    
    env_config.use_wandb = False
    env_config.data_path_to_save_groups = None
    env_config.group_size = 1 
    env_config.total_steps = 3 
    # env_config.tokenizer_name is already "gpt2" by default in TestAtroposEnvConfig
    
    env = TestAtroposEnv(config=env_config, server_configs=server_configs_list)

    await env.setup()
    
    env.n_groups_to_process = env_config.total_steps 
    env.curr_step = 0

    print(f"\nStarting simulation for {env.n_groups_to_process} turns...")

    for _ in range(env.n_groups_to_process):
        print(f"\n===== Processing Turn {env.curr_step + 1}/{env.n_groups_to_process} =====")
        item = await env.get_next_item() # Gets current_game_history
        
        scored_data_group, backlog = await env.collect_trajectories(item)

        if scored_data_group:
            print(f"--- Turn {env.curr_step + 1} Results ---")
            # scored_data_group contains a list for tokens, scores etc.
            print(f"Scored Data Group: Tokens: {scored_data_group['tokens'][0] if scored_data_group['tokens'] else 'N/A'}")
            print(f"Scores: {scored_data_group['scores'][0] if scored_data_group['scores'] else 'N/A'}")
            if scored_data_group.get('messages') and scored_data_group['messages']:
                 # Messages is List[List[Message]], access first element for the single trajectory's messages
                 # print(f"Resulting Messages List: {scored_data_group['messages']}") # Debug: show the list of lists
                 print(f"Resulting Messages for this turn: {scored_data_group['messages'][0]}")
            else:
                print("No messages in scored_data_group")
        else:
            print(f"Turn {env.curr_step + 1} failed to produce a scored data group.")
            break 

        env.curr_step += 1
        if backlog: 
            print(f"Warning: Backlog is not empty: {backlog}")

    print(f"\n===== Simulation Complete =====")
    print(f"Final game history after {env.curr_step} turns:")
    for i, msg in enumerate(env.current_game_history):
        print(f"  {i}: Role: {msg['role']}, Content: {msg['content']}")


if __name__ == "__main__":
    # Ensure .env is loaded before anything else that might need env vars
    load_dotenv() 
    asyncio.run(main()) 