from atroposlib.envs.base import BaseEnvConfig, BaseEnv
from atropos.environments.optimizer.wrapper import score_optimizer

class OptimizerBenchmarkEnvConfig(BaseEnvConfig):
    architecture: str = "mnist"

class OptimizerBenchmarkEnvironment(BaseEnv):
    env_config_cls = OptimizerBenchmarkEnvConfig

    def __init__(self, config: OptimizerBenchmarkEnvConfig, server_configs=None, slurm=False, testing=False):
        super().__init__(config, server_configs, slurm, testing)
        self.architecture = config.architecture

    async def evaluate(self, optimizer_code: str, *args, **kwargs):
        # This method is required by BaseEnv
        return score_optimizer(optimizer_code, self.architecture) 
