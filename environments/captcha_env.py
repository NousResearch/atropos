import base64
import io
import random
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import re
from datasets import load_dataset
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# System prompt for the CAPTCHA task
captcha_system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
    "Your task is to identify the text present in the CAPTCHA image."
)

captcha_system_prompt += """You are allocated a maximum of 512 tokens, please strive to use less.

You will then provide your answer like this: <answer>your answer here</answer>
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with <answer>your answer here</answer>"""


class CaptchaRow(TypedDict):
    image: Image.Image
    solution: str
    base64_image: Optional[str]


class CaptchaEnv(BaseEnv):
    name = "captcha"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.accuracy_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen2-VL-2B-Instruct",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=512,
            wandb_name="captcha",
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        try:
            wandb_metrics["train/accuracy"] = sum(self.accuracy_buffer) / len(
                self.accuracy_buffer
            )
        except ZeroDivisionError:
            pass

        self.accuracy_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        dataset = load_dataset("project-sloth/captcha-images")
        shuffled_dataset = dataset["train"].shuffle(seed=42)

        self.train_data = []
        self.test_data = []

        for i, item in enumerate(shuffled_dataset):
            img = item["image"]
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            base64_image_str = base64.b64encode(img_bytes).decode("utf-8")

            processed_item = {
                "image": item["image"],
                "solution": str(item["solution"]),
                "base64_image": base64_image_str,
            }
            if i % 10 == 0 :
                 self.test_data.append(processed_item)
            else:
                self.train_data.append(processed_item)

        self.iter = 0
        print(f"Loaded {len(self.train_data)} training examples and {len(self.test_data)} test examples.")


    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, item: CaptchaRow) -> number:
        text_prompt = "What text is shown in this CAPTCHA image?"
        user_content = [
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{item['base64_image']}"},
            },
        ]

        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": captcha_system_prompt},
                {"role": "user", "content": user_content},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )

        model_output = completion.choices[0].message.content.split("</think>")[-1]
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", model_output, re.IGNORECASE | re.DOTALL)
        extracted_answer = match.group(1).strip() if match else model_output.strip()

        score = 1 if extracted_answer.lower() == item["solution"].lower() else 0
        return score

    async def evaluate(self, *args, **kwargs):
        if not self.test_data:
            print("No test data loaded for evaluation.")
            return

        eval_tasks = []
        num_eval_samples = min(len(self.test_data), 100)
        for i in range(num_eval_samples):
            item = self.test_data[i]
            eval_tasks.append(self.rollout_and_score_eval(item))

        if not eval_tasks:
            print("No evaluation tasks created.")
            return

        print(f"Starting evaluation on {len(eval_tasks)} samples...")
        scores = await tqdm_asyncio.gather(*eval_tasks)
        accuracy = sum(scores) / len(scores) if scores else 0
        self.eval_metrics.append(("eval/accuracy", accuracy))
        print(f"Evaluation accuracy: {accuracy}")


    async def collect_trajectories(
        self, item: CaptchaRow
    ) -> Tuple[Optional[ScoredDataGroup], list[Item]]:
        text_prompt = "What text is shown in this CAPTCHA image?"
        user_content = [
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{item['base64_image']}"},
            },
        ]
        system_message_content = captcha_system_prompt
        user_message_content = user_content

        chat_completions = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": user_message_content},
            ],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )
        to_score = list()
        for i, chat_completion in enumerate(chat_completions.choices):
            messages_for_tokenizer = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": text_prompt},
                {"role": "assistant", "content": chat_completion.message.content},
            ]
            to_score.append(
                {
                    "messages_for_tokenizer": messages_for_tokenizer,
                    "model_response_content": chat_completion.message.content,
                    "gold_solution": item["solution"],
                    "finish_reason": chat_completion.finish_reason,
                    "base64_image": item["base64_image"]
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, []


    async def score(
        self, rollout_group_data
    ) -> Optional[ScoredDataGroup]:
        scores_group = ScoredDataGroup()
        scores_group["tokens"] = list()
        scores_group["masks"] = list()
        scores_group["scores"] = list()


        random.shuffle(rollout_group_data)
        for item_data in rollout_group_data:
            model_output_text = item_data["model_response_content"].split("</think>")[-1]
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", model_output_text, re.IGNORECASE | re.DOTALL)
            extracted_answer = match.group(1).strip() if match else model_output_text.strip()
            reward = 1 if extracted_answer.lower() == item_data["gold_solution"].lower() else 0

            out_dict = tokenize_for_trainer(
                self.tokenizer, item_data["messages_for_tokenizer"], item_data["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            if len([i for i in masks if i != -100]) < 5:
                continue

            scores_group["tokens"].append(tokens)
            scores_group["masks"].append(masks)
            scores_group["scores"].append(1.0 if reward else -1.0)

            if len(scores_group["tokens"]) >= self.config.group_size:
                break

        if not scores_group["tokens"]:
            return None

        for score_val in scores_group["scores"]:
            self.accuracy_buffer.append(max(score_val, 0))

        if all(s == scores_group["scores"][0] for s in scores_group["scores"]):
             pass

        return scores_group


    async def get_next_item(self) -> CaptchaRow:
        item_index = self.iter % len(self.train_data)
        next_item = self.train_data[item_index]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    CaptchaEnv.cli() 