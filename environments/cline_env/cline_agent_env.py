import json
import logging
import random
from typing import Dict, List, Optional, Tuple
from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig, ScoredDataItem
from atroposlib.type_definitions import Item, Message
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer
from atroposlib.utils.tool_call_parser import parse_tool_call

class ClineAgentEnvConfig(BaseEnvConfig):
    api_server_config: APIServerConfig
    max_steps: int = 10
    initial_prompt: Optional[str] = None
    tools: List[Dict] = []
    scoring_function: Optional[str] = None

class ClineAgentEnv(BaseEnv):
    """This is a Singleton class. State is shared"""
    def __init__(self, config: ClineAgentEnvConfig):
        super().__init__(config)
        self.api_server_config = config.api_server_config
        self.max_steps = config.max_steps
        self.initial_prompt = config.initial_prompt or "You are a helpful assistant."
        self.tools = config.tools
        self.scoring_function = config.scoring_function
        self.current_step = 0
        self.conversation_history: List[Message] = []
        self.logger = logging.getLogger(__name__)

    @classmethod
    def config_init(cls) -> ClineAgentEnvConfig:
        pass

    def setup(self):
        """Setup the environment before use. Eg, the management of the pool of agents, say number of workers x group size."""
        pass

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataItem], List[Item]]:
        """Received the task
        This is called as part of the parent collect_trajectories method and handles a SINGLE trajectory in a group of trajectories.

        Note: see other environments such as blackjack_env_no_thinking. We need to somehow get the Cline agent to use out policy model
        which is being served up by the APIServer & kept on-policy after each training steps. We can probably configure it to use a custom 
        endpoint for the model it uses for inference, and not try and manage the trajectories themselves (we just collect them and score them). 
        """
        pass

    async def get_next_item(self) -> Item:
        """Get the next task that'll be processed by the agent."""
        pass

    async def evaluate(self, *args, **kwargs):
        """Evaluate the performance of the agent on the given task. one trajectory every so many steps."""
        pass

    async def wandb_log(self, wandb_metrics: Optional[Dict[str, float]] = None):
        """Log metrics to wandb. See other envs for examples."""
        pass


if __name__ == "__main__":
    ClineAgentEnv.cli()




