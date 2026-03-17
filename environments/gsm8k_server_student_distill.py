from typing import Tuple

from atroposlib.envs.base import APIServerConfig, ServerBaseline
from atroposlib.envs.student_distillation_env import (
    StudentDistillationConfig,
    StudentDistillationEnv,
)

from environments.gsm8k_server import GSM8kEnv


class GSM8kStudentDistillEnv(GSM8kEnv, StudentDistillationEnv):
    name = "gsm8k_student_distill"
    env_config_cls = StudentDistillationConfig

    @classmethod
    def config_init(cls) -> Tuple[StudentDistillationConfig, ServerBaseline]:
        env_config = StudentDistillationConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="gsm8k_student_distill",
            student_distill_enabled=True,
            student_top_k=4,
        )
        server_config = APIServerConfig(
            model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            base_url="http://localhost:9001/v1",
            api_key="x",
            num_requests_for_eval=256,
        )
        return env_config, server_config


if __name__ == "__main__":
    GSM8kStudentDistillEnv.cli()
