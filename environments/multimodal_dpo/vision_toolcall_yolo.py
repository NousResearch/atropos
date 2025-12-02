import base64
import io
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from datasets import load_dataset
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from ultralytics import YOLO

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


def detect_objects_to_tool_call(
    img: "Image.Image | str | Path",
    model_name: str = "yolov9c.pt",
    conf_thres: float = 0.25,
) -> str:
    """
    Run YOLOv9 on the given image and return a JSON string named `tool_call`
    that lists detections as {label, confidence, [x1,y1,x2,y2]}.
    """
    model = YOLO(model_name)
    pil_img = Image.open(img) if not isinstance(img, Image.Image) else img

    results = model(pil_img, conf=conf_thres)

    detections = []
    for r in results:
        for b in r.boxes:
            cls_id = int(b.cls[0])
            label = model.names[cls_id]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            detections.append(
                {"label": label, "confidence": round(conf, 4), "box": [x1, y1, x2, y2]}
            )

    # ---- stringify and return ----------------------------------------------
    tool_call = json.dumps(detections, indent=2)
    return tool_call


class YoloDetectionEnv(BaseEnv):
    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=32,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 16,
            inference_weight=1.0,
            wandb_name="yolo_detection",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    async def setup(self):
        # Load a dataset with images (you'll need to replace this with your actual dataset)
        self.dataset = load_dataset("coco", split="train")
        self.train = self.dataset
        self.iter = 0

    async def get_next_item(self) -> Item:
        try:
            entry = self.train[self.iter % len(self.train)]
            self.iter += 1

            # Get the image
            image = entry["image"]

            # Convert image to base64
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Get ground truth detections using YOLO
            gold_tool_call = detect_objects_to_tool_call(image)

            # Create prompt with image
            system_msg = {
                "role": "system",
                "content": system_prompt
                + "\n\n"
                + "You are an AI assistant that can detect objects in images. "
                "You must submit your answer enclosed in <tool_call> tags, containing a JSON array of detections. "
                "Each detection should have 'label', 'confidence', and 'box' fields.",
            }

            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What objects do you see in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }

            messages = [system_msg, user_msg]
            prompt = tuple([frozenset(msg.items()) for msg in messages])

            return (prompt, gold_tool_call, base64_image)
        except Exception as e:
            print(f"Error in get_next_item: {e}")
            # Fallback
            fallback_prompt = tuple(
                [
                    frozenset(
                        {"role": "user", "content": "Please solve: 2 + 2 = ?"}.items()
                    )
                ]
            )
            return (fallback_prompt, "<tool_call>[]</tool_call>", None)

    async def collect_trajectories(self, item: Item) -> Tuple[ScoredDataGroup, List]:
        to_score = []
        to_backlog = []

        # Extract messages from the item
        messages = []
        for role_dict in item[0]:
            messages.append(dict(role_dict))

        # Get completions from the model
        completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=1024 * 15,
            temperature=0.8,
        )

        for completion_choice in completions.choices:
            # Create a copy of the prompt messages
            trajectory_messages = []
            for role_dict in item[0]:
                trajectory_messages.append(dict(role_dict))

            # Add the model's response
            trajectory_messages.append(
                {"role": "assistant", "content": completion_choice.message.content}
            )

            # Add to scoring queue with expected answer
            to_score.append(
                (
                    tuple(trajectory_messages),
                    item[1],  # The expected tool call JSON
                )
            )

        # Call score to get the scored data
        scored_data = await self.score(to_score)
        return scored_data, to_backlog

    def _extract_tool_call_jsons(self, text):
        """Extract JSONs from within <tool_call> tags"""
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        tool_calls = []

        for match in matches:
            try:
                json_str = match
                tool_call = json.loads(json_str)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _compare_tool_calls(self, model_response, expected_response):
        """Compare tool calls by extracting JSONs and comparing content"""
        model_jsons = self._extract_tool_call_jsons(model_response)
        expected_jsons = self._extract_tool_call_jsons(expected_response)

        if not model_jsons or not expected_jsons:
            return 0

        remaining_expected_jsons = expected_jsons.copy()

        for model_json in model_jsons:
            found_match = False
            for i, expected_json in enumerate(remaining_expected_jsons):
                if self._json_objects_match(model_json, expected_json):
                    remaining_expected_jsons.pop(i)
                    found_match = True
                    break
            if not found_match:
                return 0

        return 1 if not remaining_expected_jsons else 0

    def _json_objects_match(self, json1, json2):
        """Check if two JSON objects match"""
        try:
            for key in json2:
                if key not in json1:
                    return False
                if isinstance(json2[key], dict) and isinstance(json1[key], dict):
                    for arg_key in json2[key]:
                        if arg_key not in json1[key]:
                            return False
                        if json2[key][arg_key] != json1[key][arg_key]:
                            return False
                elif json2[key] != json1[key]:
                    return False
            return True
        except Exception:
            return False

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            # Extract the model's response
            model_response = item[0][-1]["content"]

            # Score 1 if tool calls match, 0 otherwise
            reward = 1 if self._compare_tool_calls(model_response, item[1]) else 0

            # Tokenize the conversation for learning
            out_dict = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove examples with insufficient context
            if len([1 for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)

            # Break once we have enough examples
            if len(scores["tokens"]) >= self.config.group_size:
                break

        # Record success rate metrics
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Apply length penalty if all responses are correct
        if all([score == 1.0 for score in scores["scores"]]):
            token_lengths = [len(token) for token in scores["tokens"]]
            if max(token_lengths) == 0:
                return None

            max_allowed_length = self.config.max_token_length
            length_threshold = max_allowed_length * 0.5

            scores["scores"] = []
            for length in token_lengths:
                if length <= length_threshold:
                    scores["scores"].append(1.0)
                else:
                    percentage_of_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    percentage_of_range = min(percentage_of_range, 1.0)
                    scores["scores"].append(1.0 - percentage_of_range)

        if all(scores["scores"][0] == score for score in scores["scores"]):
            return None

        return scores

    async def evaluate(self, *args, **kwargs):
        eval_tasks = []
        for test_item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(test_item))
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def rollout_and_score_eval(self, test_item):
        # Extract the image and expected tool call
        image = test_item["image"]
        expected_tool_call = detect_objects_to_tool_call(image)

        # Create messages for model
        system_msg = {
            "role": "system",
            "content": system_prompt
            + "\n\n"
            + "You are an AI assistant that can detect objects in images. "
            "You must submit your answer enclosed in <tool_call> tags, containing a JSON array of detections. "
            "Each detection should have 'label', 'confidence', and 'box' fields.",
        }

        # Convert image to base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What objects do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }

        messages = [system_msg, user_msg]

        # Get model completion
        completion = await self.server.chat_completion(
            messages=messages,
            n=1,
            max_tokens=1024 * 15,
            temperature=1.0,
            split="eval",
        )

        # Extract the model's response
        model_response = completion.choices[0].message.content

        # Score the response
        score = self._compare_tool_calls(model_response, expected_tool_call)
        return score

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = dict()

        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    YoloDetectionEnv.cli()
