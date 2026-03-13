from typing import Tuple

from atroposlib.envs.base import APIServerConfig, ServerBaseline
from atroposlib.envs.teacher_distillation_env import (
    TeacherDistillationConfig,
    TeacherDistillationEnv,
)
from environments.gsm8k_server import GSM8kEnv


class GSM8kTeacherDistillEnv(GSM8kEnv, TeacherDistillationEnv):
    """
    GSM8K environment variant that enables TeacherDistillationEnv config fields.

    This preserves the original `gsm8k_server.py` while providing a separate entrypoint
    for teacher-distillation data collection.
    """

    name = "gsm8k_teacher_distill"
    env_config_cls = TeacherDistillationConfig

    @classmethod
    def config_init(cls) -> Tuple[TeacherDistillationConfig, ServerBaseline]:
        env_config = TeacherDistillationConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="gsm8k_teacher_distill",
            teacher_enabled=True,
            teacher_top_k=4,
        )
        server_config = APIServerConfig(
            model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            base_url="http://localhost:9001/v1",
            api_key="x",
            num_requests_for_eval=256,
        )
        return env_config, server_config

    @classmethod
    def teacher_config_init(cls) -> APIServerConfig:
        return APIServerConfig(
            base_url="http://localhost:9003/v1",
            model_name="mock-teacher",
            api_key="",
            server_type="vllm",
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            timeout=1200,
        )


if __name__ == "__main__":
    GSM8kTeacherDistillEnv.cli()
