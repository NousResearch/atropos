import asyncio
import os
from typing import List, Optional, Tuple, Union
from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    ServerBaseline,
)
from atroposlib.type_definitions import Item

class MinimalEnv(BaseEnv):
    name = "minimal_test"

    async def setup(self):
        self.iter = 0

    async def evaluate(self, *args, **kwargs):
        pass

    async def get_next_item(self) -> str:
        self.iter += 1
        return f"Test Question {self.iter}"

    async def collect_trajectories(self, item: str) -> Tuple[dict, List[Item]]:
        # This calls the server!
        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            completion = await managed.chat_completion(
                messages=[
                    {"role": "user", "content": f"Please repeat this exactly: {item}"},
                ],
                max_tokens=100,
            )
            content = completion.choices[0].message.content
        
        return {
            "tokens": [1, 2, 3],
            "masks": [1, 1, 1],
            "scores": 1.0,
            "messages": [{"role": "assistant", "content": content}]
        }, []

async def test_ssot_logic():
    print("Initializing MinimalEnv with SSoT enabled...")
    
    env_config = BaseEnvConfig(
        ssot_exploration=True,
        ssot_epsilon=1.0, # Always trigger
        group_size=1,
        use_wandb=False,
        data_path_to_save_groups="ssot_test_output.jsonl",
    )
    
    server_config = APIServerConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct-AWQ",
        base_url="http://localhost:8001/v1",
        api_key="EMPTY",
    )
    
    # We need to set this to allow dummy tokens since we are using OpenAI API
    os.environ["ATROPOS_ALLOW_DUMMY_MANAGED_SERVER"] = "1"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    # ROOT CAUSE FIX: Pass server_config as a LIST to prevent Atropos from 
    # overriding the URL with its default localhost:9004-9007 logic.
    env = MinimalEnv(env_config, [server_config])
    env.n_groups_to_process = 1
    env.group_size_to_process = 1
    
    print("Running process_manager...")
    await env.process_manager()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(test_ssot_logic())
