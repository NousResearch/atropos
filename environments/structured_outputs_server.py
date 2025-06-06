import json
import logging
import os
import random
import uuid
from typing import Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from jsonschema import ValidationError
from jsonschema import validate as json_validate
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

import wandb
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class StructuredOutputsEnvConfig(BaseEnvConfig):
    """Extended configuration for StructuredOutputsEnv with additional fields."""

    debug_logging: bool = Field(
        default=False,
        description="Enable debug-level logging for more verbose output.",
    )
    suppress_base_env_logs: bool = Field(
        default=True,
        description="Suppress verbose base environment logs (like status dict updates).",
    )
    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump successful rollouts (above threshold) to JSONL files.",
    )
    dump_failed_rollouts: bool = Field(
        default=False,
        description="Whether to dump failed rollouts (all 0 scores) to JSONL files for debugging.",
    )
    rollout_save_score_threshold: float = Field(
        default=0.7,
        description="Minimum score threshold for saving rollouts to data dumps. Only groups with at least one rollout above this threshold will be saved.",  # noqa: E501
    )
    length_penalty_threshold: float = Field(
        default=0.5,
        description="Fraction of max_token_length at which to start applying length penalties (0.0-1.0).",
    )
    min_context_tokens: int = Field(
        default=10,
        description="Minimum number of non-masked tokens required for a valid training example.",
    )
    successful_rollouts_save_interval: int = Field(
        default=100,
        description="Number of processed items before saving successful rollouts to disk.",
    )
    failed_rollouts_save_interval: int = Field(
        default=50,
        description="Number of processed failed items before saving failed rollouts to disk.",
    )
    progress_log_interval: int = Field(
        default=100,
        description="Number of items between progress log messages.",
    )
    data_dump_progress_log_interval: int = Field(
        default=10,
        description="Number of items between data dump progress log messages.",
    )

    def validate_config(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.rollout_save_score_threshold <= 1.0):
            raise ValueError(
                f"rollout_save_score_threshold must be between 0.0 and 1.0, got {self.rollout_save_score_threshold}"
            )
        if self.rollout_save_score_threshold == 1.0:
            print(
                f"Warning: rollout_save_score_threshold is {self.rollout_save_score_threshold}, which may be too strict and result in no saved rollouts."  # noqa: E501
            )

        if not (0.0 <= self.length_penalty_threshold <= 1.0):
            raise ValueError(
                f"length_penalty_threshold must be between 0.0 and 1.0, got {self.length_penalty_threshold}"
            )

        if self.min_context_tokens < 1:
            raise ValueError(
                f"min_context_tokens must be at least 1, got {self.min_context_tokens}"
            )

        if self.successful_rollouts_save_interval < 1:
            raise ValueError(
                f"successful_rollouts_save_interval must be at least 1, got {self.successful_rollouts_save_interval}"
            )

        if self.failed_rollouts_save_interval < 1:
            raise ValueError(
                f"failed_rollouts_save_interval must be at least 1, got {self.failed_rollouts_save_interval}"
            )

        if self.progress_log_interval < 1:
            raise ValueError(
                f"progress_log_interval must be at least 1, got {self.progress_log_interval}"
            )

        if self.data_dump_progress_log_interval < 1:
            raise ValueError(
                f"data_dump_progress_log_interval must be at least 1, got {self.data_dump_progress_log_interval}"
            )


system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract the *first* JSON object that appears immediately after the model’s
    internal <think> … </think> block.

    • Any non‑whitespace characters between </think> and the opening “{” cause
      the extraction to fail (we want the model to output pure JSON).

    • Uses json.JSONDecoder.raw_decode for robust brace matching instead of a
      greedy regex.
    """
    # Keep only the area after the *last* </think>
    after_think = text.split("</think>")[-1]

    # Allow leading whitespace/newlines but *nothing else*
    stripped_leading = after_think.lstrip()
    leading_before_json = after_think[: len(after_think) - len(stripped_leading)]
    if stripped_leading.startswith("{") is False:
        # Either we have non‑whitespace chatter or no JSON at all
        return None
    if any(c not in " \t\r\n" for c in leading_before_json):
        # Non‑whitespace junk before the JSON
        return None

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(stripped_leading)
        return obj
    except json.JSONDecodeError:
        return None


def _subset_match(candidate: dict, reference: dict) -> bool:
    for k, v in reference.items():
        if k not in candidate or candidate[k] != v:
            return False
    return True


def _ensure_schema_dict(schema_raw) -> Optional[dict]:
    if schema_raw is None:
        return None
    return json.loads(schema_raw)


class StructuredOutputsEnv(BaseEnv):
    name = "json_struct"
    env_config_cls = StructuredOutputsEnvConfig

    def __init__(
        self,
        config: StructuredOutputsEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        # Validate configuration before proceeding
        config.validate_config()
        super().__init__(config, server_configs, slurm, testing)

        # Initialize the logger like ReasoningGym
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Add a basic stream handler if no handlers are configured
            _handler = logging.StreamHandler()
            _formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            _handler.setFormatter(_formatter)
            self.logger.addHandler(_handler)
            # Set logging level based on config
            log_level = logging.DEBUG if self.config.debug_logging else logging.INFO
            self.logger.setLevel(log_level)
        # Ensure the logger itself is enabled
        self.logger.disabled = False

        # Suppress base environment logs if requested
        if self.config.suppress_base_env_logs:
            # Set the base environment logger to WARNING level to suppress INFO logs
            base_logger = logging.getLogger("atroposlib.envs.base")
            base_logger.setLevel(logging.WARNING)

        self.percent_correct_buffer = []
        self.eval_metrics = []
        # Rollout visualisation
        self.rollouts_for_wandb = []
        self.completion_lengths = []

        # Data dumping infrastructure
        self.run_uuid = str(uuid.uuid4())
        self.rollouts_to_save_buffer: List[
            Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]
        ] = []
        self.processed_item_count = 0
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data_dumps"
        )
        self.save_file_batch_num = 0

        # For saving failed rollouts (all 0 scores) for debugging
        self.failed_rollouts_to_save_buffer: List[
            Dict[str, Union[str, List[Dict[str, Union[List[Dict[str, str]], float]]]]]
        ] = []
        self.failed_processed_item_count = 0
        self.failed_save_file_batch_num = 0

    @classmethod
    def config_init(cls) -> Tuple[StructuredOutputsEnvConfig, List[APIServerConfig]]:
        env_cfg = StructuredOutputsEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=16 * 1024,
            inference_weight=1.0,
            wandb_name="json_struct",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            debug_logging=False,
            suppress_base_env_logs=True,
            dump_rollouts=False,
            dump_failed_rollouts=False,
            rollout_save_score_threshold=0.7,
            length_penalty_threshold=0.5,
            min_context_tokens=10,
            successful_rollouts_save_interval=100,
            failed_rollouts_save_interval=50,
            progress_log_interval=100,
            data_dump_progress_log_interval=10,
        )
        srv_cfgs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]
        return env_cfg, srv_cfgs

    async def setup(self):
        self.logger.info("Setting up StructuredOutputs environment...")

        self.logger.info("Loading json-mode-agentic dataset...")
        ds = load_dataset("interstellarninja/json-mode-agentic", split="train").shuffle(
            seed=42
        )
        self.logger.info(f"Loaded dataset with {len(ds)} total examples")

        split = ds.train_test_split(0.02, seed=42)
        self.train, self.test = split["train"], split["test"]
        self.logger.info(
            f"Split dataset: {len(self.train)} training, {len(self.test)} test examples"
        )

        self.iter = 0
        self.percent_correct_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []

        self.logger.info(
            "StructuredOutputs environment setup complete. Ready to start training!"
        )
        self.logger.info(
            f"Configuration: group_size={self.config.group_size}, max_token_length={self.config.max_token_length}, steps_per_eval={self.config.steps_per_eval}"  # noqa: E501
        )
        if self.config.dump_rollouts:
            self.logger.info(
                f"Data dumping enabled with score threshold: {self.config.rollout_save_score_threshold}"
            )
        if self.config.dump_failed_rollouts:
            self.logger.info("Failed rollout dumping enabled for debugging analysis")
        self.logger.info(
            "Using strict JSON format enforcement: models must output valid JSON after </think> tags"
        )

    def _score(self, cand_txt: str, gold_txt: str, schema: Optional[dict]) -> float:
        # Assumes schema is either dict or None
        cand = _extract_json(cand_txt)
        gold = _extract_json(gold_txt)

        if cand is None:
            if self.config.debug_logging:
                self.logger.debug("Failed to extract JSON from candidate response")
            return 0.0

        if gold is None:
            if self.config.debug_logging:
                self.logger.debug("Failed to extract JSON from gold response")
            return 0.0

        if schema:
            try:
                json_validate(cand, schema)
            except ValidationError as e:
                if self.config.debug_logging:
                    self.logger.debug(f"JSON schema validation failed: {e}")
                return 0.0

        match_result = _subset_match(cand, gold)
        if self.config.debug_logging:
            self.logger.debug(f"Subset match result: {match_result}")

        return 1.0 if match_result else 0.0

    async def rollout_and_score_eval(self, item) -> float:
        try:
            conv = item["conversations"]
            sys = next((m for m in conv if m["from"] == "system"), None)
            usr = next((m for m in conv if m["from"] == "human"), None)
            gold = next((m for m in conv if m["from"] == "gpt"), None)
        except (KeyError, TypeError, StopIteration) as e:
            self.logger.warning(f"Error parsing evaluation item: {e}")
            return 0.0

        if not usr or not gold:
            self.logger.warning("Missing user or gold message in evaluation item")
            return 0.0

        try:
            schema = _ensure_schema_dict(item.get("schema"))
            msgs = [
                {
                    "role": "system",
                    "content": system_prompt + "\n\n" + (sys["value"] if sys else ""),
                },
                {"role": "user", "content": usr["value"]},
            ]
            prompt = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        except (KeyError, TypeError) as e:
            self.logger.warning(f"Error constructing evaluation prompt: {e}")
            return 0.0

        try:
            comp = await self.server.completion(
                prompt=prompt,
                n=1,
                max_tokens=self.config.max_token_length - len(prompt),
                temperature=0.2,
                split="eval",
            )
            return self._score(comp.choices[0].text, gold["value"], schema)
        except Exception as e:
            self.logger.warning(f"Error during evaluation completion or scoring: {e}")
            return 0.0

    async def evaluate(self, *_, **__):
        self.logger.info("Starting evaluation...")
        if not self.test:
            self.logger.warning("No test items available for evaluation. Skipping.")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        self.logger.info(f"Starting evaluation on {len(self.test)} items...")

        try:
            scs = await tqdm_asyncio.gather(
                *[self.rollout_and_score_eval(t) for t in self.test]
            )
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        if not scs:
            percent_correct = 0.0
            self.logger.warning("No evaluation scores obtained")
        else:
            # Filter out any None values that might have been returned due to errors
            valid_scores = [s for s in scs if s is not None]
            if not valid_scores:
                percent_correct = 0.0
                self.logger.warning("All evaluation attempts failed")
            else:
                percent_correct = sum(valid_scores) / len(valid_scores)
                if len(valid_scores) < len(scs):
                    self.logger.warning(
                        f"Only {len(valid_scores)}/{len(scs)} evaluation items succeeded"
                    )

        self.eval_metrics.append(("eval/percent_correct", percent_correct))
        self.logger.info(
            f"Evaluation finished. Percent correct: {percent_correct:.4f} ({len(valid_scores) if 'valid_scores' in locals() else len(scs)}/{len(self.test)} items)"  # noqa: E501
        )

    async def get_next_item(self):
        try:
            row = self.train[self.iter % len(self.train)]
        except (IndexError, ZeroDivisionError) as e:
            self.logger.error(
                f"Error accessing training data at iteration {self.iter}: {e}"
            )
            return None

        # Enhanced progress tracking with configurable interval
        if self.iter % self.config.progress_log_interval == 0:
            dataset_position = self.iter % len(self.train)
            completion_percentage = (
                (self.iter / (self.config.total_steps * self.config.group_size)) * 100
                if self.config.total_steps > 0
                else 0
            )
            self.logger.info(
                f"Processing item {self.iter} (dataset position: {dataset_position}/{len(self.train)}, "
                f"estimated completion: {completion_percentage:.1f}%)"
            )

        self.iter += 1

        try:
            conv = row["conversations"]
            sys = next((m for m in conv if m["from"] == "system"), None)
            usr = next((m for m in conv if m["from"] == "human"), None)
            gold = next((m for m in conv if m["from"] == "gpt"), None)
        except (KeyError, TypeError, StopIteration) as e:
            self.logger.error(
                f"Error parsing conversation data at iteration {self.iter}: {e}"
            )
            if self.config.debug_logging:
                self.logger.debug(f"Problematic row data: {row}")
            return None

        if not usr:
            self.logger.warning(
                f"No user message found in conversation at iteration {self.iter}"
            )
            return None

        if not gold:
            self.logger.warning(
                f"No gold response found in conversation at iteration {self.iter}"
            )
            return None

        try:
            prompt = []
            if sys:
                prompt.append(
                    frozenset(
                        {
                            "role": "system",
                            "content": system_prompt + "\n\n" + sys["value"],
                        }.items()
                    )
                )
            prompt.append(frozenset({"role": "user", "content": usr["value"]}.items()))
            answer = gold["value"] if gold else ""
            schema = _ensure_schema_dict(row.get("schema"))
            return (tuple(prompt), answer, schema)
        except (KeyError, TypeError) as e:
            self.logger.error(
                f"Error constructing prompt at iteration {self.iter}: {e}"
            )
            return None

    async def collect_trajectories(self, itm) -> Tuple[ScoredDataGroup, List[Item]]:
        try:
            msgs = [dict(p) for p in itm[0]]
            schema = itm[2]
            prompt = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        except (TypeError, IndexError, KeyError) as e:
            self.logger.error(f"Error preparing trajectory collection: {e}")
            return None, []

        try:
            comps = await self.server.completion(
                prompt=prompt,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length - len(prompt),
                temperature=0.8,
            )
        except Exception as e:
            self.logger.error(f"Error during completion generation: {e}")
            return None, []
        try:
            bunch = []
            for ch in comps.choices:
                bunch.append(
                    (
                        msgs + [{"role": "assistant", "content": ch.text}],
                        itm[1],
                        schema,
                        ch.finish_reason,
                    )
                )
        except (AttributeError, TypeError) as e:
            self.logger.error(f"Error processing completion choices: {e}")
            return None, []

        try:
            scored = await self.score(bunch)
        except Exception as e:
            self.logger.error(f"Error during scoring: {e}")
            return None, []

        if scored is not None:
            try:
                await self.add_rollouts_for_wandb(scored, itm)
            except Exception as e:
                self.logger.warning(f"Error adding rollouts for wandb: {e}")

            # If rollouts were generated and scored, and data dumping is enabled, prepare them for saving
            if self.config.dump_rollouts:
                try:
                    # Only save groups that have at least one rollout with score > threshold
                    group_scores = scored.get("scores", [])
                    threshold = self.config.rollout_save_score_threshold
                    if any(score > threshold for score in group_scores):
                        self.logger.debug(
                            f"Saving group with scores: {[f'{s:.3f}' for s in group_scores]} (has high-quality rollout, threshold: {threshold})"  # noqa: E501
                        )
                        rollouts_with_scores_to_save = []

                        num_scored_rollouts = len(group_scores)
                        for i in range(num_scored_rollouts):
                            conversation_messages = bunch[i][0]
                            score_for_rollout = group_scores[i]
                            rollouts_with_scores_to_save.append(
                                {
                                    "conversation": conversation_messages,
                                    "score": score_for_rollout,
                                    "expected_json": bunch[i][1],
                                    "schema": bunch[i][2],
                                }
                            )

                        if rollouts_with_scores_to_save:
                            item_data_to_save = {
                                "item_id": f"structured_outputs_{self.processed_item_count}",
                                "rollouts": rollouts_with_scores_to_save,
                            }
                            self.rollouts_to_save_buffer.append(item_data_to_save)
                            self.processed_item_count += 1

                        # Calculate progress toward next save
                        current_batch_progress = (
                            self.processed_item_count
                            % self.config.successful_rollouts_save_interval
                        )
                        if current_batch_progress == 0:
                            current_batch_progress = (
                                self.config.successful_rollouts_save_interval
                            )

                        # Log progress at configurable intervals or when we hit the save threshold
                        if (
                            current_batch_progress
                            % self.config.data_dump_progress_log_interval
                            == 0
                            or current_batch_progress
                            == self.config.successful_rollouts_save_interval
                        ):
                            self.logger.info(
                                f"Data dump progress: {current_batch_progress}/{self.config.successful_rollouts_save_interval} items buffered "  # noqa: E501
                                f"(Total processed: {self.processed_item_count}, Buffer size: {len(self.rollouts_to_save_buffer)})"  # noqa: E501
                            )

                        # Check if it's time to save a batch of rollouts
                        if (
                            self.config.dump_rollouts
                            and self.processed_item_count
                            % self.config.successful_rollouts_save_interval
                            == 0
                            and self.processed_item_count > 0
                        ):
                            log_msg = (
                                f"Reached {self.processed_item_count} processed items. "
                                f"Triggering save for {len(self.rollouts_to_save_buffer)} items "
                                f"(each with multiple scored rollouts)."
                            )
                            self.logger.info(log_msg)
                            await self._save_rollouts_to_jsonl()
                    else:
                        max_score = max(group_scores) if group_scores else 0.0
                        self.logger.debug(
                            f"Skipping group save - no high-quality rollouts (max score: {max_score:.3f}, threshold: {threshold})"  # noqa: E501
                        )
                except Exception as e:
                    self.logger.warning(f"Error during data dumping: {e}")

        return scored, []

    async def score(self, data) -> Optional[ScoredDataGroup]:
        sd = ScoredDataGroup(tokens=[], masks=[], scores=[])
        random.shuffle(data)

        for msgs, gold, schema, fin in data:
            try:
                reward = self._score(msgs[-1]["content"], gold, schema)
            except (IndexError, KeyError, TypeError) as e:
                self.logger.warning(f"Error scoring response: {e}")
                if self.config.debug_logging:
                    self.logger.debug(f"Problematic message data: {msgs}")
                continue

            try:
                out = tokenize_for_trainer(
                    tokenizer=self.tokenizer,
                    chat=msgs,
                    finish_reason=fin,
                    include_messages=True,
                )
            except Exception as e:
                self.logger.warning(f"Error tokenizing conversation: {e}")
                continue

            # Use configurable minimum context tokens
            valid_tokens = len([m for m in out["masks"] if m != -100])
            if valid_tokens < self.config.min_context_tokens:
                if self.config.debug_logging:
                    self.logger.debug(
                        f"Skipping example with insufficient context: {valid_tokens} < {self.config.min_context_tokens}"
                    )
                continue

            sd["tokens"].append(out["tokens"])
            sd["masks"].append(out["masks"])
            sd["scores"].append(reward)
            if len(sd["tokens"]) >= self.config.group_size:
                break
        if not sd["tokens"]:
            self.logger.warning(
                "No valid items were scored in this group - all items had insufficient context or failed scoring"
            )
            return None

        # Calculate and log average score for the current group
        current_scores = sd.get("scores", [])
        if current_scores:
            average_score = sum(current_scores) / len(current_scores)
            log_message_main = f"Group average score: {average_score:.4f}"
            if all(s == 1.0 for s in current_scores):
                self.logger.info(f"{log_message_main} (All correct in this group!)")
            elif all(s == 0.0 for s in current_scores):
                self.logger.info(f"{log_message_main} (All failed - no valid JSON!)")
            else:
                self.logger.info(log_message_main)

        # Apply length penalty if all responses are initially correct
        if sd["scores"] and all(score == 1.0 for score in sd["scores"]):
            token_lengths = [len(toks) for toks in sd["tokens"]]
            max_allowed_length = self.config.max_token_length
            length_threshold = max_allowed_length * self.config.length_penalty_threshold
            penalties_applied = 0
            new_scores = []
            for length in token_lengths:
                if length <= length_threshold:
                    new_scores.append(1.0)
                else:
                    pct = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    pct = min(pct, 1.0)
                    penalized_score = 1.0 - pct
                    new_scores.append(penalized_score)
                    penalties_applied += 1

            if penalties_applied > 0:
                avg_length = sum(token_lengths) / len(token_lengths)
                self.logger.debug(
                    f"Applied length penalty to {penalties_applied}/{len(token_lengths)} responses "
                    f"(avg length: {avg_length:.0f}, threshold: {length_threshold:.0f}, "
                    f"penalty threshold: {self.config.length_penalty_threshold:.1f})"
                )

            sd["scores"] = new_scores

        # Record success rate metrics (scores are already in 0.0-1.0 range)
        for score_val in sd["scores"]:
            self.percent_correct_buffer.append(1.0 if score_val >= 0.5 else 0.0)

        if len(sd["tokens"]) < self.config.group_size:
            self.logger.debug(
                f"Group too small ({len(sd['tokens'])} < {self.config.group_size}), skipping"
            )
            return None

        if all(s == sd["scores"][0] for s in sd["scores"]):
            self.logger.debug(
                f"All scores in group are identical ({sd['scores'][0]:.4f}) - no learning signal, skipping group"
            )

            # Before returning None, check if this is a completely failed group (all 0.0 scores) for debugging
            if self.config.dump_failed_rollouts and all(
                score == 0.0 for score in sd["scores"]
            ):
                self.logger.debug(
                    "Saving failed group (all 0 scores) for debugging analysis"
                )
                await self._save_failed_group_for_debugging(data, sd)

            return None

        return sd

    async def _save_failed_group_for_debugging(
        self, rollout_group_data, scores_container
    ):
        """Helper method to save failed groups (all 0 scores) for debugging analysis."""
        failed_rollouts_with_scores_to_save = []

        # Build the failed rollouts data structure
        for i, (msgs, gold, schema, fin) in enumerate(rollout_group_data):
            if i < len(scores_container["scores"]):
                score_for_rollout = scores_container["scores"][i]
                failed_rollouts_with_scores_to_save.append(
                    {
                        "conversation": msgs,
                        "score": score_for_rollout,
                        "expected_json": gold,
                        "schema": schema,
                    }
                )

        if failed_rollouts_with_scores_to_save:
            failed_item_data_to_save = {
                "item_id": f"structured_outputs_{self.failed_processed_item_count}",
                "rollouts": failed_rollouts_with_scores_to_save,
            }
            self.failed_rollouts_to_save_buffer.append(failed_item_data_to_save)
            self.failed_processed_item_count += 1

            # Calculate progress toward next failed save
            failed_batch_progress = (
                self.failed_processed_item_count
                % self.config.failed_rollouts_save_interval
            )
            if failed_batch_progress == 0:
                failed_batch_progress = self.config.failed_rollouts_save_interval

            # Log progress at configurable intervals or when we hit the save threshold
            if (
                failed_batch_progress % self.config.data_dump_progress_log_interval == 0
                or failed_batch_progress == self.config.failed_rollouts_save_interval
            ):
                self.logger.info(
                    f"Failed rollouts progress: {failed_batch_progress}/{self.config.failed_rollouts_save_interval} items buffered "  # noqa: E501
                    f"(Total failed processed: {self.failed_processed_item_count}, Failed buffer size: {len(self.failed_rollouts_to_save_buffer)})"  # noqa: E501
                )

            # Check if it's time to save a batch of failed rollouts
            if (
                self.config.dump_failed_rollouts
                and self.failed_processed_item_count
                % self.config.failed_rollouts_save_interval
                == 0
                and self.failed_processed_item_count > 0
            ):
                failed_log_msg = (
                    f"Reached {self.failed_processed_item_count} failed items. "
                    f"Triggering save for {len(self.failed_rollouts_to_save_buffer)} failed items "
                    f"(each with multiple failed rollouts)."
                )
                self.logger.info(failed_log_msg)
                await self._save_failed_rollouts_to_jsonl()

    async def _save_rollouts_to_jsonl(self):
        """Saves the buffered rollouts to a JSONL file in the datadumps directory."""
        if not self.rollouts_to_save_buffer:
            self.logger.info("No rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"structured_outputs_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for i, rollout_dict in enumerate(self.rollouts_to_save_buffer):
                    try:
                        json.dump(rollout_dict, f)
                        f.write("\n")
                    except (TypeError, ValueError) as e:
                        self.logger.warning(f"Error serializing rollout {i}: {e}")
                        continue

            saved_count = len(self.rollouts_to_save_buffer)
            self.logger.info(
                f"Successfully saved {saved_count} rollouts to {file_path}"
            )
            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving rollouts to {file_path}: {e}"
            )

    async def _save_failed_rollouts_to_jsonl(self):
        """Saves the buffered failed rollouts (all 0 scores) to a JSONL file for debugging."""
        if not self.failed_rollouts_to_save_buffer:
            self.logger.info("No failed rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
                self.logger.info(f"Created directory: {self.datadumps_dir}")
        except OSError as e:
            self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"structured_outputs_FAILED_rollouts_{self.run_uuid}_{self.failed_save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for i, rollout_dict in enumerate(self.failed_rollouts_to_save_buffer):
                    try:
                        json.dump(rollout_dict, f)
                        f.write("\n")
                    except (TypeError, ValueError) as e:
                        self.logger.warning(
                            f"Error serializing failed rollout {i}: {e}"
                        )
                        continue

            saved_count = len(self.failed_rollouts_to_save_buffer)
            self.logger.info(
                f"Successfully saved {saved_count} FAILED rollouts to {file_path}"
            )
            self.failed_rollouts_to_save_buffer.clear()
            self.failed_save_file_batch_num += 1
        except IOError as e:
            self.logger.error(f"Error writing failed rollouts to {file_path}: {e}")
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while saving failed rollouts to {file_path}: {e}"
            )

    async def create_rollout_table(self, wandb_metrics: Dict):
        if self.rollouts_for_wandb:
            table = wandb.Table(columns=["generation", "score", "expected_json"])
            for group in self.rollouts_for_wandb:
                for gen, score, expected in group:
                    table.add_data(gen, score, expected)
            wandb_metrics["train/rollouts"] = table

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def add_rollouts_for_wandb(
        self,
        scored_data: ScoredDataGroup,
        item: Item,
    ):
        num_keep = getattr(self.config, "num_rollouts_per_group_for_logging", -1)
        if num_keep == -1:
            num_keep = self.config.group_size
        self.rollouts_for_wandb.append(
            [
                (
                    self.tokenizer.decode(scored_data["tokens"][i]),
                    scored_data["scores"][i],
                    item[1],  # expected JSON string
                )
                for i in range(num_keep)
            ]
        )
        if len(self.rollouts_for_wandb) > getattr(
            self.config, "num_rollouts_to_keep", 4
        ):
            self.rollouts_for_wandb.pop(0)

    async def wandb_log(self, metrics: Optional[Dict] = None):
        metrics = metrics or {}
        metrics = await self.create_rollout_table(metrics)
        if self.percent_correct_buffer:
            metrics["train/percent_correct"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
            self.percent_correct_buffer.clear()
        for k, v in self.eval_metrics:
            metrics[k] = v
        self.eval_metrics.clear()
        await super().wandb_log(metrics)

    def save_checkpoint(self, step, data=None):
        """Save checkpoint including current iteration number, completion lengths, and data dumping state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        data["completion_lengths"] = self.completion_lengths
        data["processed_item_count"] = self.processed_item_count
        data["save_file_batch_num"] = self.save_file_batch_num
        data["failed_processed_item_count"] = self.failed_processed_item_count
        data["failed_save_file_batch_num"] = self.failed_save_file_batch_num
        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load checkpoint including iteration number, completion lengths, and data dumping state."""
        # Call the base class method first to load the data
        super().load_checkpoint()

        # The base class loads data into attributes, so we can access them directly
        # if they were saved in save_checkpoint
        if hasattr(self, "iter"):
            # Data was loaded successfully, no need to do anything else
            pass

    async def close(self):
        """Clean up and save any remaining rollouts before exiting."""
        self.logger.info(
            "Closing StructuredOutputsEnv. Attempting to save any remaining rollouts..."
        )

        try:
            if self.config.dump_rollouts and self.rollouts_to_save_buffer:
                self.logger.info(
                    f"Found {len(self.rollouts_to_save_buffer)} rollouts in buffer. Saving now."
                )
                await self._save_rollouts_to_jsonl()
            else:
                self.logger.info("No rollouts in buffer to save upon closing.")
        except Exception as e:
            self.logger.error(f"Error saving remaining rollouts during close: {e}")

        try:
            # Also save any remaining failed rollouts
            if self.config.dump_failed_rollouts and self.failed_rollouts_to_save_buffer:
                self.logger.info(
                    f"Found {len(self.failed_rollouts_to_save_buffer)} failed rollouts in buffer. Saving now."
                )
                await self._save_failed_rollouts_to_jsonl()
            else:
                self.logger.info("No failed rollouts in buffer to save upon closing.")
        except Exception as e:
            self.logger.error(
                f"Error saving remaining failed rollouts during close: {e}"
            )

        try:
            # Call the superclass's close method if it exists
            if hasattr(super(), "close"):
                await super().close()
        except Exception as e:
            self.logger.error(f"Error calling superclass close method: {e}")

        self.logger.info("StructuredOutputsEnv closed.")


if __name__ == "__main__":
    StructuredOutputsEnv.cli()
