import base64
import json
import os
import random
import sys
import traceback
from typing import List, Optional, Tuple

from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class MultimodalComplexEnv(BaseEnv):
    name = "clevr_complex"
    name_config_cls = BaseEnvConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[GameHistory | None, List[Item]]:
        print("DEBUG: Starting collect_trajectories")
        to_score = list()
        to_backlog = list()

        # Get the current image if it was stored
        if hasattr(self, "current_image"):
            print("DEBUG: Using current_image for multimodal content")

            # Convert PIL image to base64
            import io

            img_byte_arr = io.BytesIO()
            self.current_image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            base64_image = base64.b64encode(img_byte_arr).decode("utf-8")

            # Extract text content from item
            user_content = dict(item[0][0]).get("content", "")

            # Try to parse if it's JSON
            if isinstance(user_content, str) and (
                user_content.startswith("[") or user_content.startswith("{")
            ):
                try:
                    parsed = json.loads(user_content)
                    text_content = ""
                    for element in parsed:
                        if element.get("type") == "text":
                            text_content = element.get("text", "")

                    if not text_content:
                        text_content = "Please solve this problem and provide your answer as \\boxed{answer}."
                except Exception as e:
                    print(f"DEBUG: Error parsing JSON: {e}")
                    text_content = "Please solve this problem and provide your answer as \\boxed{answer}."
            else:
                text_content = user_content

            # Create messages with the new format
            print("DEBUG: Creating multimodal message with new format")
            messages = [
                {
                    "role": "system",
                    "content": "You must submit your answer with \\boxed{answer}, please make sure to do this",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                },
            ]

        else:
            print("DEBUG: No image available, using text-only message")
            messages = [
                {
                    "role": "system",
                    "content": "You must submit your answer with \\boxed{answer}",
                },
                dict(item[0][0]),
            ]

        print("DEBUG: About to call chat_completion")
        chat_completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=1024 * 2,
            timeout=60,  # Add timeout to prevent hanging (60 seconds is more reasonable)
        )
        print("DEBUG: chat_completion call successful")

        for i, chat_completion in enumerate(chat_completions.choices):
            print(f"DEBUG: Processing completion {i+1}/{len(chat_completions.choices)}")
            messages = (
                dict(item[0][0]),
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append((messages, item[1], base64_image))

        print("DEBUG: Finished processing completions")

        print("DEBUG: Returning from collect_trajectories")
        return to_score, to_backlog

    async def postprocess_histories(
        self, trajectories: List[GameHistory]
    ) -> ScoredDataGroup:
        pass

    async def evaluate(self, *args, **kwargs):
        """
        Evaluate the environment, this is called every steps_per_eval steps

        :param args:
        :param kwargs:
        :return: None.
        """
        return

    async def setup(self):
        """Setup the environment and load the multimodal dataset"""
        self.dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex")
        self.train = self.dataset["train"]
        self.iter = 0

    async def get_next_item(self) -> Item:
        """
        Get the next items to be rolled out, including the image
        """
        try:
            print("DEBUG: Starting get_next_item")

            # Get next dataset item
            next_item = self.train[self.iter % len(self.train)]
            self.iter += 1

            print(f"DEBUG: Retrieved dataset item {self.iter-1}")

            # For debugging, we'll use a simple text-only prompt and store the image separately
            # This is because the collect_trajectories method will handle the multimodal formatting

            # Store image as a class attribute so collect_trajectories can access it
            self.current_image = next_item["image"]
            print("DEBUG: Stored image in current_image attribute")

            # Create a simple text prompt - the image will be added in collect_trajectories
            # This avoids the unhashable type error with lists in frozensets
            text_prompt = next_item["problem"]

            # Create a simple text-only prompt
            prompt = tuple(
                [frozenset({"role": "user", "content": text_prompt}.items())]
            )
            answer = next_item["solution"]

            # Convert PIL image to base64
            import io

            img_byte_arr = io.BytesIO()
            self.current_image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            base64_image = base64.b64encode(img_byte_arr).decode("utf-8")

            print("DEBUG: Created simple text-only prompt for get_next_item")
            return (prompt, answer, base64_image)

        except Exception as e:
            print(f"DEBUG: Error in get_next_item: {str(e)}")
            traceback.print_exc()

            # Create a dummy item as fallback
            prompt = tuple(
                [
                    frozenset(
                        {"role": "user", "content": "Please solve: 2 + 2 = ?"}.items()
                    )
                ]
            )
            answer = "4"
            return (prompt, answer, "obobob")

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["images"] = list()
        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Extract the answer from the model's response
            try:
                model_answer = (
                    item[0][-1]["content"].split("\\boxed{")[-1].split("}")[0]
                )
                print(
                    f"DEBUG: Model answer: {model_answer} and RG data: {rollout_group_data[0][1]}"
                )

                # Handle both numeric and yes/no answers
                gold_answer = rollout_group_data[0][1]

                # Case-insensitive comparison for yes/no and direct comparison for numbers
                if gold_answer.lower() in ["yes", "no"]:
                    reward = gold_answer.lower() == model_answer.lower()
                else:
                    # For numeric answers
                    reward = gold_answer == model_answer

            except IndexError:
                reward = False

            # remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)

            try:
                scores["images"].append(item[2])
            except IndexError:
                scores["images"].append(None)
            if len(scores["tokens"]) >= self.config.group_size:
                break

        return scores

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable is not set!")
            print("Please set it using: export OPENAI_API_KEY=your_api_key")
            sys.exit(1)

        print(
            f"DEBUG: Using API key starting with: {os.environ.get('OPENAI_API_KEY')[:5]}..."
        )

        config = BaseEnvConfig(
            wandb_name="clevr_complex",
            tokenizer_name="gpt2",
            group_size=2,
            use_wandb=False,
            max_num_workers=2,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1,
            steps_per_eval=10,
            ensure_scores_are_not_same=False,
        )

        print("DEBUG: Creating OpenAI configuration")
        server_configs = [
            APIServerConfig(
                model_name="gpt-4o",  # Using GPT-4o which has multimodal capabilities
                base_url=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                num_requests_for_eval=1,
            ),
        ]

        return config, server_configs


if __name__ == "__main__":
    MultimodalComplexEnv.cli()
