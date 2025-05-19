import random
import re
import os
import asyncio # For parallel sub-agent calls
from typing import Dict, List, Optional, Tuple, TypedDict, Union
import json

from tqdm.asyncio import tqdm_asyncio

# Assuming these are part of your atroposlib, as per the original example
from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
    APIServer,
)
from atroposlib.envs.server_handling.openai_server import OpenAIServer
from atroposlib.type_definitions import number # Item might not be used directly
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


# --- System Prompts ---

# MA Stage 1: Main agent reasons and decomposes into subtasks
MULTI_AGENT_STAGE1_PROMPT = """You are a sophisticated AI assistant that breaks down complex user requests into manageable subtasks for a team of specialized sub-agents.
Given the user's prompt, your first step is to:
1.  Provide your initial reasoning and understanding of the prompt in <|reasoning|>...</|reasoning|> tags. This reasoning should explain your plan to address the user's request.
2.  Identify and list exactly 3 to 5 distinct, actionable subtasks that, when completed by sub-agents, will provide the necessary information to generate a comprehensive final answer. Output these subtasks, each enclosed in <|subtask|>...</|end_subtask|> tags. Each subtask should be a clear instruction for a sub-agent. Ensure subtasks are self-contained and directly contribute to answering the main prompt.

Example of your expected output format:
<|reasoning|>
The user wants to understand X and how to do Y. I will first gather information about X, then about Y, and finally combine this to provide actionable advice.
</|reasoning|>
<|subtask|>Research and clearly define X, including its main characteristics.<|end_subtask|>
<|subtask|>Identify and list common methods or steps for doing Y.<|end_subtask|>
<|subtask|>Provide practical tips and common pitfalls associated with starting Y for a beginner.<|end_subtask|>
=================================

=================================
"""

# MA Stage 2: Sub-agent executes a single subtask
SUB_AGENT_PROMPT_TEMPLATE = """You are a specialized AI sub-agent. Your task is to meticulously execute the given subtask and provide a concise, factual response.
Focus solely on the subtask provided. Present your response directly.

Subtask:
{subtask_text}

Your Response:
"""

# MA Stage 3: Main agent uses sub-agent results to produce a final answer (no explicit Synthesis tag)
MULTI_AGENT_STAGE3_SYSTEM_INSTRUCTIONS = """You are a sophisticated AI assistant. You have previously analyzed a user's request and delegated several subtasks to a team of sub-agents. Now, you have received their responses.
Your current goal is to use this information to generate a final, comprehensive answer for the user.

Review all the information provided: the original prompt, your initial reasoning, the subtasks you set, and the results from the sub-agents.
Then, directly generate the "Final Output:" for the user. This should be a polished, coherent, and complete answer that directly addresses the original user prompt, based on all available information.
Ensure your output starts *exactly* with "Final Output:".
"""

MULTI_AGENT_STAGE3_USER_TEMPLATE = """Original User Prompt:
{user_prompt}

Your Initial Reasoning (from Stage 1):
{initial_reasoning}

Subtasks You Delegated and Their Corresponding Results:
{subtasks_and_results_formatted_string}

Based on all the above, please now provide the "Final Output:".
"""

# Single Agent: Now includes a reasoning step
SINGLE_AGENT_REASONING_PROMPT = """You are a highly capable and thoughtful AI assistant. Please address the user's request as thoroughly and insightfully as possible.
First, provide your step-by-step reasoning to understand the request and plan your answer. Enclose this reasoning in <|reasoning|>...</|reasoning|> tags. Be sure to consider any nuances, potential challenges, and common misconceptions related to the topic.
After your reasoning, provide the "Final Output:" that thoroughly, directly addresses the user's request. Your answer should be clear, complete, and, where appropriate, include practical examples or expert tips.

Your response must be structured as follows:
<|reasoning|>
[Your detailed reasoning here.]
</|reasoning|>
Final Output:
[Your final answer here.]
"""

# Judge prompt
JUDGE_SYSTEM_PROMPT_TEMPLATE = """You are an impartial and discerning AI evaluator. You will be given a user prompt and two responses, Response A and Response B, to that prompt.
Your task is to evaluate the quality of each response *independently* based on the following criteria:
1.  **Helpfulness and Relevance**: How well does the response address the user's prompt? Is it on-topic?
2.  **Correctness and Accuracy**: Is the information provided accurate?
3.  **Completeness**: Does the response cover the prompt adequately, or does it miss key aspects?
4.  **Clarity and Conciseness**: Is the response easy to understand? Is it unnecessarily verbose or perfectly concise?

Provide a score for each response on a scale of 0 to 100, where 0 represents a response that is completely irrelevant, incorrect, or nonsensical, and 100 represents a response that is exceptionally helpful, accurate, complete, and clear.

After providing the scores, briefly explain your reasoning for each score, referencing the criteria above.

User Prompt:
{user_prompt}

Response A:
{response_A_final_output}

Response B:
{response_B_final_output}

Please format your evaluation *strictly* as follows:
Score A: [score for A, e.g., 75]
Reasoning A: [Your detailed reasoning for A's score, addressing the criteria.]
Score B: [score for B, e.g., 80]
Reasoning B: [Your detailed reasoning for B's score, addressing the criteria.]
"""

# --- Type Definitions ---
class ChatPrompt(TypedDict):
    prompt: str
    id: Optional[str]

class JudgeEvaluationResult(TypedDict):
    score_a: Optional[float]
    score_b: Optional[float]
    reason_a: Optional[str]
    reason_b: Optional[str]
    raw_judge_response: str

class DelegationCritiqueEnvConfig(BaseEnvConfig):
    judge_server_config: Optional[APIServerConfig] = None
    sub_agent_server_config: Optional[APIServerConfig] = None
    penalty_stage1_failure: float = -10
    penalty_stage3_format_failure: float = -10
    penalty_judge_failure: float = -10

# --- Helper Functions ---
def extract_text_between_tags(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    start_index = text.find(start_tag)
    if start_index == -1: return None
    start_index += len(start_tag)
    end_index = text.find(end_tag, start_index)
    if end_index == -1: return None
    return text[start_index:end_index].strip()

def parse_stage1_output(text: str) -> Tuple[Optional[str], Optional[List[str]]]:
    reasoning = extract_text_between_tags(text, "<|reasoning|>", "</|reasoning|>")
    subtask_matches = re.findall(r"<\|subtask\|>(.*?)<\|end_subtask\|>", text, re.DOTALL)
    subtasks = [match.strip() for match in subtask_matches] if subtask_matches else None
    return reasoning, subtasks

def check_stage1_format(reasoning: Optional[str], subtasks: Optional[List[str]]) -> bool:
    if not (reasoning and isinstance(reasoning, str) and reasoning.strip()): return False
    if not (subtasks and isinstance(subtasks, list) and 3 <= len(subtasks) <= 5): return False
    return all(isinstance(st, str) and st.strip() for st in subtasks)

def extract_final_output_from_text(text: str, marker: str = "Final Output:") -> Optional[str]:
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    return None

def check_stage3_format(stage3_output_text: str) -> bool:
    final_output = extract_final_output_from_text(stage3_output_text)
    return bool(final_output and final_output.strip())

def parse_single_agent_output(text: str) -> Tuple[Optional[str], Optional[str]]:
    reasoning = extract_text_between_tags(text, "<|reasoning|>", "</|reasoning|>")
    final_output = None
    search_start_index = 0
    if reasoning is not None:
        end_reasoning_tag_idx = text.rfind("</|reasoning|>")
        if end_reasoning_tag_idx != -1:
            search_start_index = end_reasoning_tag_idx + len("</|reasoning|>")
    marker_idx = text.find("Final Output:", search_start_index)
    if marker_idx != -1:
        final_output_text = text[marker_idx + len("Final Output:"):].strip()
        if final_output_text:
            final_output = final_output_text
    if final_output is None:
        final_output_fallback = extract_final_output_from_text(text)
        if final_output_fallback:
            if reasoning:
                reasoning_start_idx = text.find("<|reasoning|>")
                reasoning_end_idx = text.rfind("</|reasoning|>")
                fo_marker_idx = text.rfind("Final Output:")
                if reasoning_start_idx != -1 and reasoning_end_idx != -1 and \
                   reasoning_start_idx < fo_marker_idx < reasoning_end_idx:
                    final_output = None
                else:
                    final_output = final_output_fallback
            elif final_output_fallback:
                final_output = final_output_fallback
    return reasoning, final_output

def check_single_agent_format(reasoning: Optional[str], final_output: Optional[str]) -> bool:
    return bool(reasoning and reasoning.strip() and final_output and final_output.strip())

def parse_judge_scores(judge_response_text: str) -> JudgeEvaluationResult:
    score_a, score_b = None, None
    reason_a_text, reason_b_text = None, None
    score_a_match = re.search(r"Score A:\s*(\d+\.?\d*)", judge_response_text, re.IGNORECASE)
    if score_a_match:
        try: score_a = float(score_a_match.group(1))
        except ValueError: pass
    score_b_match = re.search(r"Score B:\s*(\d+\.?\d*)", judge_response_text, re.IGNORECASE)
    if score_b_match:
        try: score_b = float(score_b_match.group(1))
        except ValueError: pass
    reason_a_search = re.search(r"Reasoning A:\s*(.*?)(?=\n\s*Score B:|\n\s*Reasoning B:|$)", judge_response_text, re.DOTALL | re.IGNORECASE)
    if reason_a_search: reason_a_text = reason_a_search.group(1).strip()
    reason_b_search = re.search(r"Reasoning B:\s*(.*)", judge_response_text, re.DOTALL | re.IGNORECASE)
    if reason_b_search:
        reason_b_text = reason_b_search.group(1).strip()
    return JudgeEvaluationResult(score_a=score_a, score_b=score_b, reason_a=reason_a_text, reason_b=reason_b_text, raw_judge_response=judge_response_text)


class DelegationCritiqueEnv(BaseEnv):
    name = "delegation_env"

    def __init__(
        self,
        config: DelegationCritiqueEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.judge_server = OpenAIServer(config.judge_server_config)
        if config.sub_agent_server_config:
            self.sub_agent_server = OpenAIServer(config.sub_agent_server_config)
        else:
            self.sub_agent_server = self.server
        
        self.ma_scores_buffer, self.sa_scores_buffer, self.score_diff_buffer = [], [], []
        self.stage1_format_ok_buffer, self.stage3_format_ok_buffer, self.sa_format_ok_buffer = [], [], []
        self.judge_response_failure_buffer, self.completion_lengths_stage1 = [], []
        self.eval_metrics = list()
        self.config = config

    @classmethod
    def config_init(cls) -> Tuple[DelegationCritiqueEnvConfig, List[APIServerConfig]]:
        env_config = DelegationCritiqueEnvConfig(
            tokenizer_name="NousResearch/Llama-3-8B-Instruct-Preview", # From example, for trainer
            group_size=2,
            use_wandb=True,
            rollout_server_url="http://localhost:8000", # Not used if APIServer configured directly
            total_steps=1000,
            batch_size=2,
            steps_per_eval=50,
            max_token_length=2048,
            wandb_name="cc_3stage_v4_list_prompt_env",
            judge_server_config=APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                num_requests_for_eval=16,
                max_retry_attempts=2,
                timeout=120,
            ),
            # sub_agent_server_config=APIServerConfig(model_name="gpt-4.1-nano", ...), # Optional
            penalty_stage1_failure = -10,
            penalty_stage3_format_failure = -10,
            penalty_judge_failure = -10,
        )
        generator_server_configs = [
            APIServerConfig(
                model_name="gpt-4.1-nano",
                base_url=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                num_requests_for_eval=32,
                max_retry_attempts=2,
                timeout=180,
            ),
        ]
        return env_config, generator_server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None: wandb_metrics = {}
        def _safe_avg(buffer): return sum(buffer) / len(buffer) if buffer else 0.0
        
        wandb_metrics["train/avg_ma_score"] = _safe_avg(self.ma_scores_buffer)
        wandb_metrics["train/avg_sa_score"] = _safe_avg(self.sa_scores_buffer)
        wandb_metrics["train/avg_score_diff"] = _safe_avg(self.score_diff_buffer)
        wandb_metrics["train/stage1_format_ok_rate"] = _safe_avg(self.stage1_format_ok_buffer)
        wandb_metrics["train/stage3_format_ok_rate"] = _safe_avg(self.stage3_format_ok_buffer)
        wandb_metrics["train/sa_format_ok_rate"] = _safe_avg(self.sa_format_ok_buffer)
        wandb_metrics["train/judge_parse_failure_rate"] = _safe_avg(self.judge_response_failure_buffer)
        wandb_metrics["train/avg_stage1_completion_length"] = _safe_avg(self.completion_lengths_stage1)

        self.ma_scores_buffer, self.sa_scores_buffer, self.score_diff_buffer = [], [], []
        self.stage1_format_ok_buffer, self.stage3_format_ok_buffer, self.sa_format_ok_buffer = [], [], []
        self.judge_response_failure_buffer, self.completion_lengths_stage1 = [], []
        
        for item in self.eval_metrics: wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Using a simple built-in list of example_user_prompts
        example_user_prompts = [
            "Explain the concept of blockchain technology and its potential applications beyond cryptocurrency.",
            "Outline a comprehensive marketing strategy for a new eco-friendly coffee shop in a competitive urban area.",
            "Describe the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
            "Compare and contrast three major ethical theories: utilitarianism, deontology, and virtue ethics, with examples.",
            "Provide a step-by-step guide for building a simple recommendation system using collaborative filtering.",
            "What are the primary causes of deforestation, and what are some effective global strategies to combat it?",
            "Discuss the impact of social media on adolescent mental health, citing potential positives and negatives.",
            "Plan a detailed 3-day itinerary for a first-time visitor to Rome, Italy, balancing historical sites with cultural experiences."
        ]
        self.train_prompts = [ChatPrompt(prompt=p, id=f"train_{i}") for i, p in enumerate(example_user_prompts)]
        
        # Create a distinct test set if possible, or a small subset for quick eval
        if len(example_user_prompts) > 3:
            test_prompts_sample = random.sample(example_user_prompts, 3)
        else:
            test_prompts_sample = example_user_prompts[:] # Use all if list is too short
        self.test_prompts = [ChatPrompt(prompt=p, id=f"test_{i}") for i, p in enumerate(test_prompts_sample)]
        self.iter = 0

    async def _get_judge_evaluation(self, user_prompt: str, final_output_ma: str, final_output_sa: str, split: str = "train") -> Tuple[Optional[float], Optional[float], str]:
        is_ma_response_A = random.choice([True, False])
        response_A = final_output_ma if is_ma_response_A else final_output_sa
        response_B = final_output_sa if is_ma_response_A else final_output_ma
        judge_prompt_filled = JUDGE_SYSTEM_PROMPT_TEMPLATE.format(user_prompt=user_prompt, response_A_final_output=response_A, response_B_final_output=response_B)
        
        # [JUDGE_EVALUATE]
        judge_completion = await self.judge_server.chat_completion(
            messages=[{"role": "user", "content": judge_prompt_filled}], 
            n=1, max_tokens=1024, temperature=0.0, split=split
        )
        judge_response_text = judge_completion.choices[0].message.content
        parsed_scores = parse_judge_scores(judge_response_text)
        ma_score, sa_score, judge_parse_failed = None, None, True
        if parsed_scores['score_a'] is not None and parsed_scores['score_b'] is not None:
            judge_parse_failed = False
            ma_score, sa_score = (parsed_scores['score_a'], parsed_scores['score_b']) if is_ma_response_A else (parsed_scores['score_b'], parsed_scores['score_a'])
        if split == "train": self.judge_response_failure_buffer.append(1 if judge_parse_failed else 0)
        return ma_score, sa_score, judge_response_text

    async def _run_full_3stage_ma_path(self, prompt_text: str, split: str = "train") -> Dict:
        # [MAIN_AGENT_DECOMPOSE] - Stage 1 Call
        stage1_messages_for_llm = [{"role": "system", "content": MULTI_AGENT_STAGE1_PROMPT}, {"role": "user", "content": prompt_text}]
        stage1_completion = await self.server.chat_completion(messages=stage1_messages_for_llm, n=1, max_tokens=self.config.max_token_length, temperature=0.7, split=split)
        stage1_raw_output = stage1_completion.choices[0].message.content
        stage1_finish_reason = stage1_completion.choices[0].finish_reason
        initial_reasoning, subtasks = parse_stage1_output(stage1_raw_output)
        is_stage1_format_ok = check_stage1_format(initial_reasoning, subtasks)

        base_result = {"stage1_ok": is_stage1_format_ok, "stage1_raw_output": stage1_raw_output, "stage1_finish_reason": stage1_finish_reason, "final_ma_output_from_stage3": "", "stage3_format_ok": False}
        if not is_stage1_format_ok: return base_result

        sub_agent_llm_tasks = []
        for subtask_text in subtasks:
            # [SUB_AGENT_EXECUTE]
            sub_agent_llm_tasks.append(self.sub_agent_server.chat_completion(messages=[{"role": "system", "content": SUB_AGENT_PROMPT_TEMPLATE.format(subtask_text=subtask_text)}], n=1, max_tokens=self.config.max_token_length // 2, temperature=0.5, split=split))
        sub_agent_completions = await asyncio.gather(*sub_agent_llm_tasks)
        sub_agent_results = [comp.choices[0].message.content for comp in sub_agent_completions]

        subtasks_and_results_str = "\n\n".join([f"Subtask {i+1}: {st}\nResult {i+1}: {sr}" for i, (st, sr) in enumerate(zip(subtasks, sub_agent_results))]) # type: ignore
        stage3_user_prompt_filled = MULTI_AGENT_STAGE3_USER_TEMPLATE.format(user_prompt=prompt_text, initial_reasoning=initial_reasoning, subtasks_and_results_formatted_string=subtasks_and_results_str.strip()) # type: ignore
        stage3_messages_for_llm = [{"role": "system", "content": MULTI_AGENT_STAGE3_SYSTEM_INSTRUCTIONS}, {"role": "user", "content": stage3_user_prompt_filled}]
        
        # [MAIN_AGENT_SYNTHESIZE] - Stage 3 Call (generates final output)
        stage3_completion = await self.server.chat_completion(messages=stage3_messages_for_llm, n=1, max_tokens=self.config.max_token_length, temperature=0.7, split=split)
        stage3_assistant_output_raw = stage3_completion.choices[0].message.content
        final_ma_output_from_stage3 = extract_final_output_from_text(stage3_assistant_output_raw)
        is_stage3_format_ok = bool(final_ma_output_from_stage3 and final_ma_output_from_stage3.strip())
        base_result.update({"final_ma_output_from_stage3": final_ma_output_from_stage3 or "", "stage3_format_ok": is_stage3_format_ok})
        return base_result

    async def _run_single_agent_path(self, prompt_text: str, split: str = "train") -> Dict:
        sa_messages_for_llm = [{"role": "system", "content": SINGLE_AGENT_REASONING_PROMPT}, {"role": "user", "content": prompt_text}]
        # [SINGLE_AGENT_REASONING]
        sa_completion = await self.server.chat_completion(messages=sa_messages_for_llm, n=1, max_tokens=self.config.max_token_length, temperature=0.0 if split == "eval" else 0.7, split=split)
        sa_full_raw_output = sa_completion.choices[0].message.content
        sa_reasoning, final_sa_output = parse_single_agent_output(sa_full_raw_output)
        is_sa_format_ok = check_single_agent_format(sa_reasoning, final_sa_output)
        return {"sa_full_raw_output": sa_full_raw_output, "final_sa_output": final_sa_output or "", "sa_format_ok": is_sa_format_ok}

    async def evaluate(self, *args, **kwargs):
        eval_ma_scores, eval_sa_scores, eval_score_diffs = [], [], []
        eval_s1_format_ok, eval_s3_format_ok, eval_sa_format_ok = [], [], []

        if not self.test_prompts: # Guard against empty test set
            print("Warning: No test prompts available for evaluation.")
            return

        for item in self.test_prompts:
            prompt_text = item["prompt"]
            sa_data = await self._run_single_agent_path(prompt_text, split="eval")
            eval_sa_format_ok.append(1 if sa_data["sa_format_ok"] else 0)

            ma_data = await self._run_full_3stage_ma_path(prompt_text, split="eval")
            eval_s1_format_ok.append(1 if ma_data["stage1_ok"] else 0)

            if not ma_data["stage1_ok"]: continue
            eval_s3_format_ok.append(1 if ma_data["stage3_format_ok"] else 0)
            if not ma_data["stage3_format_ok"]: continue
            if not sa_data["sa_format_ok"] or not sa_data["final_sa_output"]: # Ensure SA output is valid for judging
                 # print(f"Warning: SA output for prompt '{prompt_text[:30]}' malformed or empty, skipping judge for this pair in eval.")
                 continue


            ma_score, sa_score, _ = await self._get_judge_evaluation(prompt_text, ma_data["final_ma_output_from_stage3"], sa_data["final_sa_output"], split="eval")
            if ma_score is not None: eval_ma_scores.append(ma_score)
            if sa_score is not None: eval_sa_scores.append(sa_score)
            if ma_score is not None and sa_score is not None: eval_score_diffs.append(ma_score - sa_score)

        def _safe_avg(buffer): return sum(buffer) / len(buffer) if buffer else 0.0
        self.eval_metrics.append(("eval/avg_ma_score", _safe_avg(eval_ma_scores)))
        self.eval_metrics.append(("eval/avg_sa_score", _safe_avg(eval_sa_scores)))
        self.eval_metrics.append(("eval/avg_score_difference", _safe_avg(eval_score_diffs)))
        self.eval_metrics.append(("eval/stage1_format_ok_rate", _safe_avg(eval_s1_format_ok)))
        self.eval_metrics.append(("eval/stage3_format_ok_rate", _safe_avg(eval_s3_format_ok)))
        self.eval_metrics.append(("eval/sa_format_ok_rate", _safe_avg(eval_sa_format_ok)))

    async def collect_trajectories(self, item: ChatPrompt) -> Tuple[Optional[ScoredDataGroup], list]:
        prompt_text = item["prompt"]
        sa_data = await self._run_single_agent_path(prompt_text, split="train")
        
        items_to_score_list = []
        ma_path_tasks = [self._run_full_3stage_ma_path(prompt_text, split="train") for _ in range(self.config.group_size)]
        multi_agent_path_results = await asyncio.gather(*ma_path_tasks)

        for ma_data_from_path in multi_agent_path_results:
            combined_data_for_scoring = {"prompt_text": prompt_text, **sa_data, **ma_data_from_path}
            items_to_score_list.append(combined_data_for_scoring)
            
        scored_data_group_or_none = await self.score(items_to_score_list)
        return scored_data_group_or_none, []

    async def score(self, rollout_group_data_list: List[Dict]) -> Optional[ScoredDataGroup]:
        scored_data_group = ScoredDataGroup(tokens=[], masks=[], scores=[], reasoning=[], subtasks=[])
        
        for data_item in rollout_group_data_list:
            prompt_text = data_item["prompt_text"]
            stage1_raw_output = data_item["stage1_raw_output"]
            stage1_finish_reason = data_item["stage1_finish_reason"]
            sa_full_raw_output = data_item.get("sa_full_raw_output", "")
            sa_final_output_for_judge = data_item["final_sa_output"]
            sa_format_ok = data_item["sa_format_ok"]
            final_ma_output = data_item.get("final_ma_output_from_stage3", "")
            is_stage3_format_ok = data_item.get("stage3_format_ok", False)
            subtasks = None
            initial_reasoning = None
            if stage1_raw_output:
                initial_reasoning, subtasks = parse_stage1_output(stage1_raw_output)
            else:
                initial_reasoning, subtasks = None, None

            self.sa_format_ok_buffer.append(1 if sa_format_ok else 0)
            reward = 0.0
            judge_response_text = None
            ma_judge_score = None
            sa_judge_score = None

            if not data_item["stage1_ok"]:
                reward = self.config.penalty_stage1_failure
                self.stage1_format_ok_buffer.append(0)
                self.stage3_format_ok_buffer.append(0)
            else:
                self.stage1_format_ok_buffer.append(1)
                self.stage3_format_ok_buffer.append(1 if is_stage3_format_ok else 0)

                if not sa_format_ok or not sa_final_output_for_judge:
                    pass

                ma_judge_score, sa_judge_score, judge_response_text = await self._get_judge_evaluation(
                    prompt_text, final_ma_output, sa_final_output_for_judge, split="train"
                )

                # --- NEW REWARD LOGIC ---
                if ma_judge_score is None or sa_judge_score is None:
                    reward = self.config.penalty_judge_failure
                elif not is_stage3_format_ok or not sa_format_ok:
                    reward = 0.0  # Pass if format is wrong
                else:
                    diff = ma_judge_score - sa_judge_score
                    if diff < -20:
                        reward = -3.0
                    elif diff < -10:
                        reward = -2.0
                    elif diff < 0:
                        reward = -1.0
                    elif diff > 20:
                        reward = 3.0
                    elif diff > 10:
                        reward = 2.0
                    elif diff > 0:
                        reward = 1.0
                    else:
                        reward = 0.1  # Both correct, but MA not better

                if ma_judge_score is not None: self.ma_scores_buffer.append(ma_judge_score)
                if sa_judge_score is not None: self.sa_scores_buffer.append(sa_judge_score)
                if ma_judge_score is not None and sa_judge_score is not None: self.score_diff_buffer.append(ma_judge_score - sa_judge_score)

            # Save artifact for this item
            artifact = {
                "prompt": prompt_text,
                "stage1_raw_output": stage1_raw_output,
                "subtasks": subtasks,
                "final_ma_output": final_ma_output,
                "sa_full_raw_output": sa_full_raw_output,
                "final_sa_output": sa_final_output_for_judge,
                "judge_response": judge_response_text,
                "reward": reward
            }
            save_artifact(artifact)

            tokenization_messages = [
                {"role": "system", "content": MULTI_AGENT_STAGE1_PROMPT},
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": stage1_raw_output}
            ]
            tokenized_output = tokenize_for_trainer(self.tokenizer, tokenization_messages, stage1_finish_reason)
            if len([m for m in tokenized_output["masks"] if m != -100]) < 10: continue

            scored_data_group["tokens"].append(tokenized_output["tokens"])
            scored_data_group["masks"].append(tokenized_output["masks"])
            scored_data_group["scores"].append(reward)
            scored_data_group["reasoning"].append(initial_reasoning)
            scored_data_group["subtasks"].append(subtasks)
            self.completion_lengths_stage1.append(len(tokenized_output["tokens"]))

        if not scored_data_group["tokens"]: return None
        return scored_data_group

    async def get_next_item(self) -> ChatPrompt:
        if not self.train_prompts: raise ValueError("Training prompts are not loaded. Call setup() first.")
        next_item = self.train_prompts[self.iter % len(self.train_prompts)]
        self.iter += 1
        return next_item

ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "artifacts.json")

def save_artifact(artifact: dict):
    try:
        if os.path.exists(ARTIFACTS_PATH):
            with open(ARTIFACTS_PATH, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(artifact)
        with open(ARTIFACTS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[Artifact Save Error] {e}")

if __name__ == "__main__":
    DelegationCritiqueEnv.cli()