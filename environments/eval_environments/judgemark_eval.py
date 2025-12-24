"""
JudgeMark v2 Evaluation Environment

This environment evaluates how well a language model can judge creative writing.
It measures the model's ability to:
- Assign consistent, discriminative scores
- Correlate with human preferences (LMSYS Arena rankings)
- Separate good writing from bad writing

Based on: https://github.com/EQ-bench/Judgemark-v2
Paper/Leaderboard: https://eqbench.com/judgemark-v2.html

The benchmark presents pre-generated creative writing samples to the judge model,
asks for 0-10 scores on 17 literary criteria, then computes:
- Raw and calibrated score distributions
- Kendall's tau correlation with reference rankings
- Score stability across repeated runs
- Inter-model separability metrics
- Final composite Judgemark score

Usage:
    python judgemark_eval.py evaluate \
        --openai.base_url https://api.openai.com/v1 \
        --openai.api_key $OPENAI_API_KEY \
        --openai.model_name gpt-4o \
        --env.data_dir_to_save_evals ../evals/judgemark/gpt-4o
"""

import asyncio
import json
import math
import os
import re
import statistics
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
import scipy.stats
from eval_helpers import (
    create_system_content,
    get_default_thinking_prompt,
    save_eval_results,
)
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig

# Path to JudgeMark data files (relative to this file)
JUDGEMARK_DATA_DIR = Path(__file__).parent.parent.parent / "Judgemark-v2" / "data"


# =============================================================================
# Constants (from Judgemark-v2/config/constants.py)
# =============================================================================

# Reference model scores for correlation (LMSYS Arena ELO-like scores)
REFERENCE_MODEL_SCORES = {
    "kimi-k2": 1387,
    "claude-opus-4": 1417,
    "claude-sonnet-4": 1380,
    "chatgpt-4o-latest": 1425,
    "gpt-4.1": 1399,
    "qwen3-235b-a22b": 1366,
    "gemma-3-27b-it": 1355,
    "mistral-small-3.2-24b": 1334,
    "reka-flash-3": 1250,
    "grok-3-beta": 1401,
    "gpt-4.1-mini": 1349,
    "gemma-3-12b-it": 1333,
    "gemma-3-4b-it": 1282,
    "gpt-4.1-nano": 1309,
}

# Negative criteria markers - these get inverted (higher = worse writing)
NEGATIVE_MARKERS = [
    "meandering",
    "weak dialogue",
    "tell-don't-show",
    "unsurprising or uncreative",
    "amateurish",
    "purple prose",
    "overwrought",
    "incongruent ending positivity",
    "unearned transformations",
]


# =============================================================================
# Scoring Functions (from Judgemark-v2/core/scoring.py)
# =============================================================================


def parse_scores(judge_response: str) -> Dict[str, float]:
    """
    Parse score lines from judge output with flexible formatting.

    Accepts formats like:
      **Quality:** 7.5
      Quality: 7.5
      **Quality:** [7.5]
    """
    pattern = (
        r"^\s*"
        r"(?:\*\*)?([^\n:\[\]]{2,100}):(?:\*\*)?"
        r"\s*"
        r"(?:\[)?(?:\*\*)?"
        r"(-?\d+(?:\.\d+)?)"
        r"(?:\*\*)?(?:\])?"
        r"\s*$"
    )

    matches = re.findall(pattern, judge_response, re.MULTILINE)
    scores = {metric.strip(): float(score) for metric, score in matches}
    return scores


def compute_raw_score(
    scores: Dict[str, float], scoring_min: float = 0, scoring_max: float = 10
) -> Optional[float]:
    """
    Compute aggregated raw score from parsed criterion scores.

    - Filters to valid range [min, max]
    - Inverts negative criteria (e.g., "purple prose" where high = bad)
    - Averages and scales to 1-10 range
    """
    valid_scores = {k: v for k, v in scores.items() if scoring_min <= v <= scoring_max}

    if len(valid_scores) < 5:
        return None

    total = 0.0
    count = 0

    for criteria, val in valid_scores.items():
        crit_lower = criteria.lower().strip()
        if crit_lower in NEGATIVE_MARKERS:
            # Invert negative criteria
            new_val = (scoring_min + scoring_max) - val
        else:
            new_val = val
        total += new_val
        count += 1

    avg = total / count

    # Scale to 1-10 range
    if scoring_max == scoring_min:
        scaled = 1
    else:
        scaled = 1 + (avg - scoring_min) * (9 / (scoring_max - scoring_min))

    return round(scaled, 2)


def confidence_interval_95(data: List[float]) -> float:
    """Compute 95% confidence interval for the mean."""
    n = len(data)
    if n < 2:
        return 0.0
    stdev = statistics.stdev(data)
    return 1.96 * (stdev / math.sqrt(n))


def compute_detailed_distribution(scores: List[float]) -> Dict:
    """Compute detailed distribution statistics."""
    if not scores:
        return {}
    return {
        "count": len(scores),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "mean": round(statistics.mean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "stdev": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
        "p10": round(float(np.percentile(scores, 10)), 3),
        "p25": round(float(np.percentile(scores, 25)), 3),
        "p75": round(float(np.percentile(scores, 75)), 3),
        "p90": round(float(np.percentile(scores, 90)), 3),
    }


def build_landmark_calibration_config(
    scores: List[float], desired_points: List[float] = None
) -> Dict:
    """
    Build piecewise-linear calibration from raw distribution landmarks.
    Maps [min, Q1, median, Q3, max] to desired_points [0, 3, 5, 7, 10].
    """
    if not scores or len(scores) < 2:
        return {"method": "piecewise_landmark", "in_landmarks": [], "out_landmarks": []}

    if desired_points is None:
        desired_points = [0, 3, 5, 7, 10]

    in_min = min(scores)
    in_q1 = float(np.percentile(scores, 25))
    in_med = float(statistics.median(scores))
    in_q3 = float(np.percentile(scores, 75))
    in_max = max(scores)

    return {
        "method": "piecewise_landmark",
        "in_landmarks": [in_min, in_q1, in_med, in_q3, in_max],
        "out_landmarks": desired_points,
    }


def apply_landmark_calibration(x: float, config: Dict) -> float:
    """Apply piecewise-linear calibration transform."""
    inL = config.get("in_landmarks", [])
    outL = config.get("out_landmarks", [])

    if len(inL) != 5 or len(outL) != 5:
        return x

    in_min, in_q1, in_med, in_q3, in_max = inL
    out_min, out_q1, out_med, out_q3, out_max = outL

    def linear_map(val, old_lo, old_hi, new_lo, new_hi):
        if abs(old_hi - old_lo) < 1e-12:
            return new_lo
        frac = (val - old_lo) / (old_hi - old_lo)
        return new_lo + frac * (new_hi - new_lo)

    if x <= in_q1:
        return linear_map(x, in_min, in_q1, out_min, out_q1)
    elif x <= in_med:
        return linear_map(x, in_q1, in_med, out_q1, out_med)
    elif x <= in_q3:
        return linear_map(x, in_med, in_q3, out_med, out_q3)
    else:
        return linear_map(x, in_q3, in_max, out_q3, out_max)


def normalize(
    val: float, min_val: float, max_val: float, ascending: bool = True
) -> float:
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5

    normalized = (val - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))

    if not ascending:
        normalized = 1.0 - normalized

    return normalized


# =============================================================================
# JudgeMark Configuration
# =============================================================================


class JudgeMarkEvalConfig(BaseEnvConfig):
    """JudgeMark v2 evaluation configuration."""

    # Data files (defaults to bundled Judgemark-v2 data)
    samples_file: str = Field(
        default="", description="Path to samples JSON (uses bundled if empty)"
    )
    prompts_file: str = Field(
        default="", description="Path to prompts JSON (uses bundled if empty)"
    )

    # Scoring settings
    scoring_min: float = Field(default=0, description="Minimum score value")
    scoring_max: float = Field(default=10, description="Maximum score value")

    # Generation settings
    eval_max_tokens: int = Field(
        default=0, description="Max tokens for judge response (0 = model default)"
    )
    eval_temperature: float = Field(
        default=0.0, description="Temperature for judge model"
    )

    # Thinking mode (optional - can help reasoning about scores)
    thinking_mode: bool = Field(
        default=False, description="Enable thinking mode for judge"
    )
    custom_thinking_prompt: Optional[str] = Field(default=None)
    custom_system_prompt: Optional[str] = Field(default=None)

    # Retry settings
    max_retries: int = Field(default=3, description="Max retries on API failure")
    retry_delay: float = Field(default=1.0, description="Delay between retries")

    # Debug
    full_debug: bool = Field(default=False, description="Save full judge responses")

    # Subset filtering (optional)
    max_samples: Optional[int] = Field(
        default=None, description="Limit number of samples to evaluate (None = all)"
    )


class JudgeMarkEvalEnv(BaseEnv):
    """JudgeMark v2 evaluation environment."""

    name = "judgemark_eval"

    def __init__(
        self,
        config: JudgeMarkEvalConfig,
        server_configs: List[APIServerConfig],
        slurm_config=None,
    ):
        super().__init__(config, server_configs, slurm_config)
        self.config: JudgeMarkEvalConfig = config

        # Initialize OpenAI client
        server_config = server_configs[0]
        self.client = openai.AsyncOpenAI(
            api_key=server_config.api_key,
            base_url=server_config.base_url,
        )
        self.model_name = server_config.model_name

        # Storage for results
        self.samples_data = {}
        self.judge_prompts = {}
        self.rubric_criteria = ""
        self.score_anchoring = ""

    @classmethod
    def config_init(cls) -> Tuple[JudgeMarkEvalConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        return (
            JudgeMarkEvalConfig(
                eval_max_tokens=0,
                use_wandb=True,
                wandb_name="judgemark_eval",
            ),
            [
                APIServerConfig(
                    model_name="gpt-4o",
                    base_url="https://api.openai.com/v1",
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                )
            ],
        )

    async def setup(self):
        """Load JudgeMark data files."""
        print("\nLoading JudgeMark v2 data...")

        # Determine data directory
        data_dir = JUDGEMARK_DATA_DIR
        if not data_dir.exists():
            raise FileNotFoundError(
                f"JudgeMark data not found at {data_dir}. "
                "Please clone Judgemark-v2 into the atropos root directory."
            )

        # Load samples
        samples_path = (
            Path(self.config.samples_file)
            if self.config.samples_file
            else data_dir / "judgemark_v3_samples_3_iter.json"
        )
        with open(samples_path) as f:
            self.samples_data = json.load(f)

        # Load prompts
        prompts_path = (
            Path(self.config.prompts_file)
            if self.config.prompts_file
            else data_dir / "judge_prompts_v3_noref_nocot_noanchor_x96.json"
        )
        with open(prompts_path) as f:
            self.judge_prompts = json.load(f)

        # Load rubric files
        with open(data_dir / "rubric_criteria.txt") as f:
            self.rubric_criteria = f.read()

        with open(data_dir / "rubric_score_anchoring.txt") as f:
            self.score_anchoring = f.read()

        # Inject rubric into prompts
        for key, prompt in self.judge_prompts.items():
            if isinstance(prompt, str):
                prompt = prompt.replace("<RUBRIC_CRITERIA>", self.rubric_criteria)
                prompt = prompt.replace("<SCORE_ANCHORING>", self.score_anchoring)
                self.judge_prompts[key] = prompt

        # Count total samples
        total_samples = sum(
            len(items)
            for model_info in self.samples_data.values()
            for items in model_info.get("samples", {}).values()
        )

        print(f"  Writer models: {len(self.samples_data)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Judge prompts: {len(self.judge_prompts)}")

        if self.config.max_samples:
            print(f"  Limiting to: {self.config.max_samples} samples")

    def _create_system_content(self) -> Optional[str]:
        """Create system message for the judge."""
        base_content = create_system_content(
            self.config.thinking_mode,
            self.config.custom_thinking_prompt,
            self.config.custom_system_prompt,
        )
        return base_content

    async def _send_to_judge(self, prompt: str) -> str:
        """Send prompt to judge model and get response."""
        system_content = self._create_system_content()

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.config.max_retries):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.config.eval_temperature,
                }
                if self.config.eval_max_tokens > 0:
                    kwargs["max_tokens"] = self.config.eval_max_tokens

                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise e

        return ""

    async def _evaluate_single_sample(
        self,
        model_name: str,
        iteration_key: str,
        item_id: str,
        item_text: str,
        prompt_template: str,
    ) -> Dict:
        """Evaluate a single writing sample."""
        result = {
            "writer_model": model_name,
            "iteration": iteration_key,
            "item_id": item_id,
            "text_length": len(item_text),
            "parsed_scores": {},
            "aggregated_score_raw": None,
            "error": None,
        }

        try:
            # Build the full prompt
            final_prompt = prompt_template.replace(
                "[TEST MODEL RESPONSE]", "[TEST MODEL RESPONSE]\n" + item_text
            )

            # Get judge response
            judge_response = await self._send_to_judge(final_prompt)

            if self.config.full_debug:
                result["judge_response"] = judge_response

            # Parse scores
            extracted_scores = parse_scores(judge_response)
            result["parsed_scores"] = extracted_scores

            # Compute raw score
            raw_score = compute_raw_score(
                extracted_scores, self.config.scoring_min, self.config.scoring_max
            )
            result["aggregated_score_raw"] = raw_score

            if raw_score is None:
                result["error"] = (
                    f"Only {len(extracted_scores)} valid scores parsed (need 5+)"
                )

        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
            if self.config.full_debug:
                result["traceback"] = traceback.format_exc()

        return result

    def _compute_cross_model_stats(
        self, scores_by_model: Dict[str, List[float]]
    ) -> Dict:
        """Compute cross-model statistics including correlation with reference."""
        arrays = list(scores_by_model.values())

        if len(arrays) < 2:
            return {
                "anova_f": 0,
                "anova_p": 1,
                "kw_stat": 0,
                "kw_p": 1,
                "std_dev_across_models": 0,
                "pearson_r": 0,
                "kendall_tau": 0,
            }

        # ANOVA and Kruskal-Wallis
        f_stat, f_p = scipy.stats.f_oneway(*arrays)
        kw_stat, kw_p = scipy.stats.kruskal(*arrays)

        # Std across model means
        model_means = [statistics.mean(scores) for scores in arrays]
        std_across = statistics.pstdev(model_means)

        # Correlation with reference rankings
        ref_pairs = []
        for model, scores in scores_by_model.items():
            if model in REFERENCE_MODEL_SCORES:
                ref_pairs.append(
                    (statistics.mean(scores), REFERENCE_MODEL_SCORES[model])
                )

        if len(ref_pairs) >= 2:
            means, refs = zip(*ref_pairs)
            pearson_r, _ = scipy.stats.pearsonr(means, refs)
            kendall_tau, _ = scipy.stats.kendalltau(means, refs)
        else:
            pearson_r, kendall_tau = 0.0, 0.0

        return {
            "anova_f": f_stat,
            "anova_p": f_p,
            "kw_stat": kw_stat,
            "kw_p": kw_p,
            "std_dev_across_models": std_across,
            "pearson_r": pearson_r,
            "kendall_tau": kendall_tau,
            "num_models_with_reference": len(ref_pairs),
        }

    async def evaluate(self):
        """Run the full JudgeMark evaluation."""
        print(f"\n{'='*60}")
        print("Starting JudgeMark v2 Evaluation")
        print(f"{'='*60}")
        print(f"  Judge model: {self.model_name}")
        print(f"  Thinking mode: {self.config.thinking_mode}")
        print(f"{'='*60}\n")

        # Build list of items to process
        items_to_process = []
        for model_name, model_info in self.samples_data.items():
            samples_dict = model_info.get("samples", {})
            for iteration_key, iteration_items in samples_dict.items():
                for item_id, item_text in iteration_items.items():
                    if item_id not in self.judge_prompts:
                        continue

                    items_to_process.append(
                        {
                            "model_name": model_name,
                            "iteration_key": iteration_key,
                            "item_id": item_id,
                            "item_text": item_text,
                            "prompt_template": self.judge_prompts[item_id],
                        }
                    )

        # Apply sample limit if specified
        if self.config.max_samples:
            items_to_process = items_to_process[: self.config.max_samples]

        print(f"Processing {len(items_to_process)} samples...\n")

        # Process all samples
        tasks = [
            self._evaluate_single_sample(
                item["model_name"],
                item["iteration_key"],
                item["item_id"],
                item["item_text"],
                item["prompt_template"],
            )
            for item in items_to_process
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Judging samples")

        # Aggregate results by writer model
        raw_scores_by_model = defaultdict(list)
        calibrated_scores_by_model = defaultdict(list)

        valid_results = [r for r in results if r["aggregated_score_raw"] is not None]
        failed_results = [r for r in results if r["aggregated_score_raw"] is None]

        for r in valid_results:
            raw_scores_by_model[r["writer_model"]].append(r["aggregated_score_raw"])

        # Compute calibration
        all_raw_scores = [r["aggregated_score_raw"] for r in valid_results]
        calibration_config = build_landmark_calibration_config(all_raw_scores)

        # Apply calibration
        for r in valid_results:
            calibrated = apply_landmark_calibration(
                r["aggregated_score_raw"], calibration_config
            )
            r["aggregated_score_calibrated"] = calibrated
            calibrated_scores_by_model[r["writer_model"]].append(calibrated)

        # Compute statistics
        raw_distribution = compute_detailed_distribution(all_raw_scores)
        calibrated_distribution = compute_detailed_distribution(
            [r["aggregated_score_calibrated"] for r in valid_results]
        )

        raw_cross_stats = self._compute_cross_model_stats(dict(raw_scores_by_model))
        calibrated_cross_stats = self._compute_cross_model_stats(
            dict(calibrated_scores_by_model)
        )

        # Per-model stats
        model_stats = {}
        for model, scores in raw_scores_by_model.items():
            if len(scores) > 0:
                model_stats[model] = {
                    "count": len(scores),
                    "mean_raw": statistics.mean(scores),
                    "mean_calibrated": statistics.mean(
                        calibrated_scores_by_model[model]
                    ),
                    "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "ci95": confidence_interval_95(scores),
                }

        # Compute final Judgemark score
        # Normalize components
        kendall_norm = normalize(calibrated_cross_stats["kendall_tau"], 0.15, 0.75)
        kw_norm = normalize(calibrated_cross_stats["kw_stat"], 100.0, 1300.0)
        std_norm = normalize(calibrated_cross_stats["std_dev_across_models"], 0.0, 2.6)

        # Separability score (simplified)
        separability = (kw_norm + std_norm) / 2.0

        # Final score: correlation + separability (simplified version)
        final_judgemark_score = (kendall_norm + separability) / 2.0

        # Build metrics summary
        metrics = {
            "final_judgemark_score": round(final_judgemark_score, 4),
            "kendall_tau_calibrated": round(calibrated_cross_stats["kendall_tau"], 4),
            "kendall_tau_raw": round(raw_cross_stats["kendall_tau"], 4),
            "kruskal_wallis_stat": round(calibrated_cross_stats["kw_stat"], 2),
            "std_dev_across_models": round(
                calibrated_cross_stats["std_dev_across_models"], 4
            ),
            "num_models_with_reference": calibrated_cross_stats[
                "num_models_with_reference"
            ],
            "total_samples": len(results),
            "valid_samples": len(valid_results),
            "failed_samples": len(failed_results),
            "raw_score_distribution": raw_distribution,
            "calibrated_score_distribution": calibrated_distribution,
            "calibration_config": calibration_config,
            "model_stats": model_stats,
            "normalized_components": {
                "kendall_tau_norm": round(kendall_norm, 4),
                "kw_stat_norm": round(kw_norm, 4),
                "std_dev_norm": round(std_norm, 4),
                "separability": round(separability, 4),
            },
        }

        # Print summary
        print(f"\n{'='*60}")
        print("JudgeMark v2 Evaluation Results")
        print(f"{'='*60}")
        print(f"  Final Judgemark Score: {final_judgemark_score:.4f}")
        print(
            f"  Kendall's τ (calibrated): {calibrated_cross_stats['kendall_tau']:.4f}"
        )
        print(f"  Kruskal-Wallis stat: {calibrated_cross_stats['kw_stat']:.2f}")
        print(
            f"  Std dev across models: {calibrated_cross_stats['std_dev_across_models']:.4f}"
        )
        print(f"\n  Valid samples: {len(valid_results)}/{len(results)}")
        print(
            f"  Models with reference: {calibrated_cross_stats['num_models_with_reference']}"
        )

        print("\n  Per-model averages (calibrated):")
        sorted_models = sorted(
            model_stats.items(), key=lambda x: x[1]["mean_calibrated"], reverse=True
        )
        for model, stats in sorted_models[:10]:
            print(
                f"    {model:.<40} {stats['mean_calibrated']:.3f} ±{stats['ci95']:.3f}"
            )
        if len(sorted_models) > 10:
            print(f"    ... and {len(sorted_models) - 10} more models")

        print(f"{'='*60}\n")

        # Save results
        if self.config.data_dir_to_save_evals:
            save_eval_results(self.config.data_dir_to_save_evals, metrics, results)

        return metrics, results

    # Required BaseEnv methods (not used for CLI evaluation)

    async def get_next_item(self):
        pass

    async def collect_trajectories(self, item):
        pass

    async def score(self, rollout_group_data):
        pass

    async def wandb_log(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    JudgeMarkEvalEnv.cli()
