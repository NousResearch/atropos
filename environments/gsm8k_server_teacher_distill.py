import os
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
        # Default model/tokenizer, but allow override via env for flexibility
        default_model = "NousResearch/DeepHermes-3-Llama-3-3B-Preview"
        model_name = os.environ.get("STUDENT_MODEL", default_model)
        tokenizer_name = os.environ.get("STUDENT_TOKENIZER", model_name)

        env_config = TeacherDistillationConfig(
            tokenizer_name=tokenizer_name,
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
            model_name=model_name,
            base_url="http://localhost:9001/v1",
            api_key="x",
            num_requests_for_eval=256,
        )
        return env_config, server_config

    @classmethod
    def teacher_config_init(cls) -> APIServerConfig:
        teacher_model = os.environ.get("TEACHER_MODEL", "mock-teacher")
        # Ensure teacher tokenizer matches teacher model by default
        teacher_tokenizer = os.environ.get("TEACHER_TOKENIZER", teacher_model)
        if teacher_model == "mock-teacher" and "TEACHER_TOKENIZER" not in os.environ:
            # Fallback for mock teacher to use student tokenizer
            teacher_tokenizer = os.environ.get("STUDENT_TOKENIZER", "NousResearch/DeepHermes-3-Llama-3-3B-Preview")

        return APIServerConfig(
            base_url="http://localhost:9003/v1",
            model_name=teacher_model,
            api_key="",
            server_type="vllm",
            tokenizer_name=teacher_tokenizer,
            timeout=1200,
        )


if __name__ == "__main__":
    GSM8kTeacherDistillEnv.cli()
