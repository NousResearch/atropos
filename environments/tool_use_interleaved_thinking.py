"""
Interleaved‑Thinking Single‑Block Environment
============================================


This environment lets a model emit *multiple* <tool_call>/<tool_response> pairs
**inside one still‑open <think> block**, then close </think> and write the
final answer – all within a single assistant turn.


Unlike the first draft, this version is **stand‑alone**: it does **NOT**
inherit from SingleToolCallingEnv.  All required boiler‑plate from that
class is copied here so nothing breaks when you swap env names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import httpx
import wandb
from datasets import Dataset, load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Message
from atroposlib.utils.io import parse_http_response
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)

# Set to True to always print debug information.
DEBUG = True
EXECUTION_FEEDBACK = True
TOOL_USAGE_BONUS = 0.2

# Hard caps for generation length
MAX_REPLY_TOKENS = 2048  # truncate any single assistant reply to ≤1024 tokens
MAX_GEN_PER_TURN = 512  # never request more than 512 new tokens from the model
# Maximum number of thinking/tool-use turns per rollout
MAX_ROLLOUT_TURNS = 3


system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)


TOOL_SYSTEM_PROMPT = (
    "You are a function-calling & reasoning AI model. You are provided with "
    "function signatures inside <reasoning_tools> … XML tags. After calling & "
    "executing the functions, you will get results inside <tool_response> … "
    "Here are the available tools:\n\n"
    "<reasoning_tools>\n"
    "[\n"
    "  {\n"
    '    "type": "function",\n'
    '    "function": {\n'
    '      "name": "calculator",\n'
    '      "description": '
    '        "Evaluate a numeric Python expression and return the result.",\n'
    '      "parameters": {\n'
    '        "type": "object",\n'
    '        "properties": {\n'
    '          "expr": {\n'
    '            "type": "string",\n'
    '            "description": '
    '              "A pure-Python arithmetic expression\'"\n'
    "          }\n"
    "        },\n"
    '        "required": ["expr"]\n'
    "      }\n"
    "    }\n"
    "  },\n"
    "  {\n"
    '    "type": "function",\n'
    '    "function": {\n'
    '      "name": "python_interpreter",\n'
    '      "description": '
    '        "Run a short Python snippet and return stdout plus the last '
    'expression.",\n'
    '      "parameters": {\n'
    '        "type": "object",\n'
    '        "properties": {\n'
    '          "code": {\n'
    '            "type": "string",\n'
    '            "description": "Python source code to execute."\n'
    "          }\n"
    "        },\n"
    '        "required": ["code"]\n'
    "      }\n"
    "    }\n"
    "  }\n"
    "]\n"
    "</reasoning_tools>\n\n"
    "You must use reasoning tools such as python_interpreter as a tool call when available "
    "for hard problems such as math before providing your final answer.\n"
    "Always provide your final numeric answer (or final result) in \\\\boxed{...} so it "
    "can be automatically graded right after closing </think> tag.\n\n"
    "For reasoning tools, return interleaved tool calls within <think> </think> tags.\n"
    "<think>\n"
    "<tool_call>{'name': <function-name>, 'arguments': <args-dict>}</tool_call>\n"
    "<!-- system pauses runtime for execution -->\n"
    "<tool_response>{'result': <result>}</tool_response>\n"
    "<!-- assistant resumes within same think -->\n"
    "</think>\n"
    "<!-- plain text answer with \\\\boxed{...} -->\n"
)

SYSTEM_PROMPT = system_prompt + TOOL_SYSTEM_PROMPT


class InterleavedInlineEnv(BaseEnv):
    """
    One episode = user prompt → single assistant message with inline tool
    calls inside a still‑open <think> block.
    """

    name = "interleaved_inline"
    _re_last_call = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>\s*$", re.S)

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer: List[float] = []
        self.eval_metrics: List[Tuple[str, float]] = []
        self.rollouts_for_wandb = []
        self.iter = 0
        import random

        self.rng = random.Random()
        # Dynamic few‑shot pool: list of (user_msg, assistant_msg) tuples
        self.dynamic_pool: List[Tuple[Dict, Dict]] = []
        self.dynamic_pool_max = 0  # keep at most 4 real examples
        self.max_token_len = 8192

    async def get_server_info(self):
        """Override to prevent server from overwriting our max_token_len"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.config.rollout_server_url}/info") as resp:
                data = await parse_http_response(resp, logger)
                if data["batch_size"] != -1:
                    self.config.batch_size = data["batch_size"]
                # Log what the server tried to set max_token_len to
                if data["max_token_len"] != -1:
                    logger.info(
                        f"Server tried to set max_token_len to {data['max_token_len']}\n"
                        f"keeping our value of {self.max_token_len}"
                    )
        if self.config.batch_size == -1:
            logging.warning("Batch size not set by config or server!")
        if self.config.group_size > self.config.batch_size:
            raise ValueError(
                f"group_size ({self.config.group_size}) "
                f"must be less than batch_size ({self.config.batch_size})"
            )

    @classmethod
    def config_init(cls):
        cfg = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=16 * 8192,
            inference_weight=1.0,
            wandb_name="toolcall_interleaved",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            max_gen_per_turn=MAX_GEN_PER_TURN,
        )
        servers = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            )
        ]
        return cfg, servers

    async def setup(self):
        """
        Load a streamed subset of **nvidia/AceReason-Math**.


        We keep only rows whose *answer* looks purely numeric so the
        calculator / python_interpreter tools can verify them automatically.



        The env‑var SUBSET_ROWS (default 1000) controls how many rows we keep.
        """

        N = int(os.getenv("SUBSET_ROWS", "1000"))

        stream_ds = load_dataset(  # ≈50 k rows total → stream
            # "NVIDIA/OpenMathReasoning",
            # split="cot",
            # "NVIDIA/OpenMathReasoning",
            # split="cot",
            # "open-r1/OpenR1-Math-220k",
            "nvidia/AceReason-Math",
            split="train",
            streaming=True,
        )

        _numeric = re.compile(r"^[0-9+\-*/(). %√\\\\sqrt{}]+$").fullmatch

        subset = []
        for ex in stream_ds:
            if len(subset) >= N:
                break
            # some datasets use "answer", others "expected_answer"
            ans_raw = ex.get("answer", ex.get("expected_answer"))
            if ans_raw is None:
                continue
            ans = str(ans_raw).strip()
            if _numeric(ans):
                subset.append(
                    {
                        "problem": ex["problem"],
                        "expected_answer": ans,
                    }
                )

        full = Dataset.from_list(subset)
        if DEBUG:
            print(f"[DEBUG setup] kept {len(subset)} rows from Dataset")

        split = full.train_test_split(test_size=0.02, seed=42)

        split = full.train_test_split(test_size=0.02, seed=42)
        self.train, self.test = split["train"], split["test"]
        self.train = self.train.shuffle(seed=int.from_bytes(os.urandom(2), "big"))

    async def _completion_until(
        self, prompt: str, max_tokens: int, stop: Optional[Union[str, List[str]]] = None
    ) -> str:
        comp = await self.server.completion(
            prompt=prompt,
            stop=stop,
            max_tokens=max_tokens,
            temperature=0.8,
        )
        return comp.choices[0].text

    def _extract_last_call(self, chunk: str):
        """
        Return the JSON dict for the *last* <tool_call> … </tool_call> block
        in `chunk`, or **None** if no such block exists.
        Also handles incomplete tool calls (missing </tool_call> tag).
        """
        # First try to find complete tool calls
        matches = self._re_last_call.findall(chunk)
        if matches:
            try:
                return json.loads(matches[-1])
            except Exception:
                pass

        # If no complete tool calls, look for incomplete ones (missing </tool_call>)
        last_tool_call_pos = chunk.rfind("<tool_call>")
        if last_tool_call_pos != -1:
            json_start = last_tool_call_pos + len("<tool_call>")
            json_text = chunk[json_start:].strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                # Try partial JSON extraction
                brace_count = 0
                json_end = 0
                for i, char in enumerate(json_text):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                if json_end > 0:
                    try:
                        return json.loads(json_text[:json_end])
                    except json.JSONDecodeError:
                        pass
        return None

    def _is_new_tool_call(self, raw: str) -> bool:
        """
        Return True if there's an unresponded <tool_call> in raw
        (i.e., open call without matching </tool_response>).
        """
        pos = raw.rfind("<tool_call>")
        if pos == -1:
            return False
        return "</tool_response>" not in raw[pos:]

    @staticmethod
    def _canon_num(txt: str) -> str:
        """Return number string without commas / spaces; keep leading sign."""
        return txt.strip().replace(",", "").replace(" ", "")

    # boxed{answer} pattern for final numeric result
    _re_box = re.compile(r"\\boxed\{([^}]*)\}")

    def _boxed_after_think(self, text: str) -> Optional[str]:
        """
        Return the first \\boxed{…} that appears *after* the closing </think>
        tag.  Returns None if </think> is missing or no boxed answer exists.
        """
        think_pos = text.find("</think>")
        if think_pos == -1:
            return None
        m = self._re_box.search(text, pos=think_pos)
        return m.group(1).strip() if m else None

    async def _exec_tool(self, call_json: Dict):
        """
        Execute reasoning‑time tools.


        • python_interpreter → POST code to the local coding server running at  localhost:5002/execute
                        and return {"stdout":..., "result":...}
        • calculator  → eval(expr) in a math‑only sandbox and return the number.
        """
        name = call_json["name"]
        args = call_json["arguments"]

        if name == "python_interpreter":
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    payload = {"code": args["code"], "input": ""}
                    resp = await client.post(
                        "http://localhost:5002/execute", json=payload
                    )
                    data = resp.json()
            except httpx.ConnectError:
                print(
                    "❌ [CRITICAL] Python interpreter server not available at localhost:5002"
                )
                print("Please ensure the code_exec_server Docker container is running")
                raise RuntimeError(
                    "Python interpreter server not available - cannot continue without verification"
                )
            if DEBUG:
                print(f"[DEBUG _exec_tool] {name} result → {data}")
            return {
                "stdout": data.get("output", ""),
                "result": data.get("output", "").strip(),
            }
        elif name == "calculator":
            import math

            expr = args["expr"]
            val = eval(expr, {"__builtins__": {}}, {"math": math})
            if DEBUG:
                print(f"[DEBUG _exec_tool] {name} result → {val}")
            return {"value": val}
        else:
            raise ValueError(f"Unknown tool name {name}")

    async def _execute_turn_inference(
        self,
        turn_idx: int,
        prompts: List[str],
        ridx_map: List[int],
        expected_calls_by_turn: List[List[str]],
    ) -> List[str]:
        """Execute inference for a turn using optimal batching strategy."""
        print(f"\n\033[95m=== Expected Tool Calls for Turn {turn_idx+1} ===\033[0m")
        print(f"\033[95m{expected_calls_by_turn[turn_idx]}\033[0m\n")

        # Always use batched identical prompts for turn 0, heterogeneous for others
        if turn_idx == 0:
            choices = await self._batch_identical_prompts(
                prompts[0], len(ridx_map), turn_idx
            )
        else:
            choices = await self._batch_heterogeneous_prompts(prompts, turn_idx)

        return choices

    async def _batch_identical_prompts(
        self, prompt: str, count: int, turn_idx: int
    ) -> List[str]:
        """Handle identical prompts efficiently using n parameter."""
        print(
            f"    \033[93m→ TURN {turn_idx+1} prompt full:\033[0m "
            f"\033[92m{prompt}\033[0m"
        )

        # Use the constant instead of config attribute
        resp = await self.server.completion(
            prompt=prompt,
            n=count,
            max_tokens=MAX_GEN_PER_TURN,
            temperature=0.8,
            stop="</tool_call>",
        )
        choices = [c.text for c in resp.choices]

        # Debug: print each rollout
        for i, raw in enumerate(choices):
            print(
                f"    \033[93m· turn {turn_idx+1} rollout raw [{i}]:\033[0m \033[94m{raw}\033[0m"
            )
            if not raw.strip():
                print(f"      → (empty or error string returned for rollout {i})")
        print("    → All turn 1 rollouts printed; moving on.\n" + "-" * 48)

        return choices

    async def _batch_heterogeneous_prompts(
        self, prompts: List[str], turn_idx: int
    ) -> List[str]:
        """Handle heterogeneous prompts using parallel requests."""
        if turn_idx == 1:
            print("=== DEBUG: Now parallelizing Turn 2 prompts ===")
        print(f"    → Parallelizing {len(prompts)} prompts at turn {turn_idx+1}")

        # Print each prompt
        for idx_p, p_str in enumerate(prompts):
            print(
                f"    \033[93m→ TURN-{turn_idx+1} prompt[{idx_p}] full:\033[0m \033[92m{p_str}\033[0m"
            )

        async def _call_single(prompt_str: str) -> str:
            try:
                # Use the constant instead of config attribute
                comp = await self.server.completion(
                    prompt=prompt_str,
                    n=1,
                    max_tokens=MAX_GEN_PER_TURN,
                    temperature=0.8,
                    stop="</tool_call>",
                )
                return comp.choices[0].text
            except Exception as e:
                print(f"    → Turn {turn_idx+1} _call_single exception: {e}")
                return ""

        tasks = [_call_single(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        # Debug: print results for all turns
        choices = []
        for i, rtext in enumerate(results):
            raw = rtext or ""
            print(
                f"    \033[93m· rollout {i} (Turn {turn_idx+1}) full reply:\033[0m \033[94m{raw}\033[0m\n"
                + "-" * 48
            )
            if not raw:
                print(f"    → Rollout {i} returned empty or error string")
            choices.append(raw)

        return choices

    def _json_objects_match(self, j1, j2):
        try:
            for k in j2:
                if k not in j1:
                    return False
                if isinstance(j2[k], dict):
                    if not self._json_objects_match(j1[k], j2[k]):
                        return False
                elif j1[k] != j2[k]:
                    return False
            return True
        except Exception:
            return False

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        """
        One prompt → `n = group_size` sampled assistant completions in
        parallel (single OpenAI request with n completions).  Mirrors the
        logic in SingleToolCallingEnv.
        """
        messages_tuple, expected_raw = item
        expected = (
            json.loads(expected_raw) if isinstance(expected_raw, str) else expected_raw
        )

        # Re‑inflate frozensets to normal dicts
        prompt_msgs = [dict(r) for r in messages_tuple]

        if EXECUTION_FEEDBACK:
            # MODE: Real interleaved tool execution
            return await self._collect_trajectories_with_execution(
                prompt_msgs, expected
            )
        else:
            # MODE: Static generation for data collection (current behavior)
            return await self._collect_trajectories_static(prompt_msgs, expected)

    async def _collect_trajectories_static(
        self, prompt_msgs: List[Dict], expected
    ) -> Tuple[ScoredDataGroup, List]:
        """
        Original static generation mode - no tool execution, just data collection.
        """
        # Convert to text prompt
        prompt_txt = self.tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, tokenize=False
        )

        if DEBUG:
            clean_prompt = prompt_txt.replace("<|eot_id|>", "")
            print(
                f"\n\033[93m▶ BATCH PROMPT (tokens {len(prompt_txt)}):\033[0m "
                f"\033[92m{clean_prompt}\033[0m\n{'-'*60}"
            )

        # One API call → many completions
        completions = await self.server.completion(
            prompt=prompt_txt,
            n=self.config.group_size,
            max_tokens=MAX_GEN_PER_TURN,
            temperature=0.8,
        )

        scored: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "advantages": None,
            "ref_logprobs": None,
            "messages": None,
            "group_overrides": {},
            "overrides": None,
            "images": None,
        }

        for idx, choice in enumerate(completions.choices):
            raw = choice.text or ""
            toks = self.tokenizer.encode(raw)
            if len(toks) > MAX_REPLY_TOKENS:
                toks = toks[:MAX_REPLY_TOKENS]
                raw = self.tokenizer.decode(toks)
                raw = self.tokenizer.decode(toks)
                if DEBUG:
                    print(f"[DEBUG] truncated reply {idx} to {len(toks)} tokens")
            assistant_msg = {"role": "assistant", "content": raw}

            # Create the full context for tokenization - cast to Message type
            full_ctx: List[Message] = prompt_msgs + [assistant_msg]

            # Outcome‑based reward: compare boxed answer to expected expr
            expr = (
                expected["arguments"]["code"][6:-1]
                if (
                    isinstance(expected, dict)
                    and "arguments" in expected
                    and "code" in expected["arguments"]
                    and expected["arguments"]["code"].startswith("print(")
                    and expected["arguments"]["code"].endswith(")")
                )
                else None
            )
            expr = (
                expected["arguments"]["code"][6:-1]
                if (
                    isinstance(expected, dict)
                    and "arguments" in expected
                    and "code" in expected["arguments"]
                    and expected["arguments"]["code"].startswith("print(")
                    and expected["arguments"]["code"].endswith(")")
                )
                else None
            )
            boxed = self._boxed_after_think(raw)

            same = boxed == expr or (
                boxed and expr and self._canon_num(boxed) == self._canon_num(expr)
            )
            reward = 1.0 if same else -1.0
            if "</think>" not in raw:
                reward = -1.0  # invalid – did not close think block
            else:
                # no tool_call tags are allowed *outside* the think block
                end_pos = raw.lower().find("</think>")
                if "<tool_call" in raw[end_pos + len("</think>") :].lower():
                    if DEBUG:
                        print(
                            "[DEBUG] tool_call found outside </think>; setting reward = -1"
                        )

                    reward = -1.0

            if DEBUG:
                print(
                    f"\033[95m--- COMPLETION {idx+1}/{self.config.group_size} ---\033[0m\n"
                    f"\033[94m{raw}\033[0m\nreward={reward}\n{'='*60}"
                )

            tok = tokenize_for_trainer(self.tokenizer, full_ctx)
            scored["tokens"].append(tok["tokens"])
            scored["masks"].append(tok["masks"])
            scored["scores"].append(reward)
            self.percent_correct_buffer.append(max(reward, 0))

        # --- harvest a success for dynamic few‑shots --------------------
        for idx, sc in enumerate(scored["scores"]):
            reply_txt = completions.choices[idx].text
            has_call = "<tool_call" in reply_txt.lower()
            if sc >= 1.0 and has_call:
                # Build (user, assistant) pair from this successful rollout
                u = {"role": "user", "content": prompt_msgs[-1]["content"]}
                a = {"role": "assistant", "content": reply_txt}
                self.dynamic_pool.append((u, a))
                if len(self.dynamic_pool) > self.dynamic_pool_max:
                    self.dynamic_pool.pop(0)
                break

        return scored, []

    async def _collect_trajectories_with_execution(
        self, prompt_msgs: List[Dict], expected
    ) -> Tuple[ScoredDataGroup, List]:
        """
        Real interleaved tool execution mode - stops at tool calls, executes them, and continues.
        Uses turn-based parallel execution for maximum efficiency.
        """
        print(
            f"\n🚀 [EXECUTION MODE] Running {self.config.group_size} rollouts with parallel turn-based execution"
        )

        scored: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "advantages": None,
            "ref_logprobs": None,
            "messages": None,
            "group_overrides": {},
            "overrides": None,
            "images": None,
        }

        # Initialize per-rollout state
        num_rollouts = self.config.group_size
        rollout_ctxs = [prompt_msgs.copy() for _ in range(num_rollouts)]
        assistant_msgs = [
            {"role": "assistant", "content": ""} for _ in range(num_rollouts)
        ]
        done = [False] * num_rollouts
        final_results = [None] * num_rollouts
        executed_tools = [[] for _ in range(num_rollouts)]

        # Track the most recent generation chunk for each rollout
        last_turns: List[str] = [""] * num_rollouts

        turn_idx = 0
        max_turns = MAX_ROLLOUT_TURNS

        while not all(done) and turn_idx < max_turns:
            print(
                f"\n[TURN {turn_idx + 1}] Processing {sum(1 for d in done if not d)} active rollouts"
            )

            # Build prompts for active rollouts only
            active_prompts = []
            active_indices = []

            for i in range(num_rollouts):
                if not done[i]:
                    prompt_txt = self.tokenizer.apply_chat_template(
                        rollout_ctxs[i],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    prompt_txt += assistant_msgs[i]["content"]
                    active_prompts.append(prompt_txt)
                    active_indices.append(i)

            if not active_prompts:
                break

            # Execute inference for this turn
            if turn_idx == 0:
                # First turn: all prompts are identical, use batched inference
                print(
                    f"[TURN {turn_idx + 1}] Batching {len(active_prompts)} identical prompts"
                )
                replies = await self._batch_identical_prompts(
                    active_prompts[0], len(active_prompts), turn_idx
                )
            else:
                # Subsequent turns: prompts may be heterogeneous, use parallel inference
                print(
                    f"⚡ [TURN {turn_idx + 1}] Parallelizing {len(active_prompts)} heterogeneous prompts"
                )
                replies = await self._batch_heterogeneous_prompts(
                    active_prompts, turn_idx
                )

            # Process each active rollout's reply
            for prompt_idx, rollout_idx in enumerate(active_indices):
                if done[rollout_idx]:
                    continue

                reply = replies[prompt_idx]
                # Save this turn's delta for summary
                last_turns[rollout_idx] = reply
                assistant_msgs[rollout_idx]["content"] += reply

                raw = assistant_msgs[rollout_idx]["content"]

                if "</think>" in raw:
                    # Think block closed
                    boxed = self._boxed_after_think(raw)
                    if boxed:
                        # Boxed answer found after </think>
                        print(
                            f"🎯 [ROLLOUT {rollout_idx}] Found boxed answer after </think> - marking complete"
                        )
                        done[rollout_idx] = True
                        rollout_ctxs[rollout_idx].append(assistant_msgs[rollout_idx])
                        final_results[rollout_idx] = raw
                        continue
                    else:
                        # Think block closed but no boxed answer
                        print(
                            f"❌ [ROLLOUT {rollout_idx}] </think> closed but no boxed answer - marking failed"
                        )
                        done[rollout_idx] = True
                        final_results[rollout_idx] = raw
                        continue
                else:
                    # Think block not closed
                    if self._is_new_tool_call(raw):
                        # Tool call present, continue to next turn after executing tool
                        print(
                            f"🔧 [ROLLOUT {rollout_idx}] Tool call detected - extracting and executing"
                        )
                        call_json = self._extract_last_call(raw)
                        if call_json is None:
                            print(
                                f"❌ [ROLLOUT {rollout_idx}] Failed to parse tool call JSON - marking inactive"
                            )
                            done[rollout_idx] = True
                            final_results[rollout_idx] = raw
                            continue

                        print(
                            f"🔧 [ROLLOUT {rollout_idx}] Executing {call_json['name']}\n"
                            f"with args: {call_json['arguments']}"
                        )
                        try:
                            result = await self._exec_tool(call_json)
                            executed_tools[rollout_idx].append(call_json)

                            print(f"✅ [ROLLOUT {rollout_idx}] Tool result: {result}")
                            # Clean up any malformed/partial closing tags before appending
                            content = assistant_msgs[rollout_idx]["content"]
                            content = re.sub(
                                r"</tool_call.*?$", "", content, flags=re.MULTILINE
                            )
                            assistant_msgs[rollout_idx]["content"] = content
                            # Append proper closing tag and response
                            assistant_msgs[rollout_idx]["content"] += "</tool_call>\n"
                            assistant_msgs[rollout_idx][
                                "content"
                            ] += (
                                f"<tool_response>{json.dumps(result)}</tool_response>\n"
                            )
                            print(
                                f"📝 [ROLLOUT {rollout_idx}] Added tool response to context"
                            )
                            continue
                        except Exception as e:
                            print(
                                f"❌ [ROLLOUT {rollout_idx}] Tool execution failed: {e}"
                            )
                            done[rollout_idx] = True
                            final_results[rollout_idx] = raw
                            continue
                    else:
                        # No new tool call or boxed answer yet
                        if turn_idx + 1 < max_turns:
                            print(
                                f"🔄 [ROLLOUT {rollout_idx}] Still thinking—continuing to next turn"
                            )
                            continue
                        # max turns reached, fail
                        print(
                            f"⚠️ [ROLLOUT {rollout_idx}] Max turns reached without completion—marking failed"
                        )
                        done[rollout_idx] = True
                        final_results[rollout_idx] = raw
                        continue

            turn_idx += 1

        # Process final results and score
        print(f"\n🏁 [EXECUTION COMPLETE] Processed {turn_idx} turns")

        # -- Summary of all rollouts before scoring --
        expr = None
        if (
            isinstance(expected, dict)
            and "arguments" in expected
            and "code" in expected["arguments"]
        ):
            code_str = expected["arguments"]["code"]
            if code_str.startswith("print(") and code_str.endswith(")"):
                expr = code_str[6:-1]

        print("\n\033[96m🔎 Final rollout results:\033[0m")
        any_success = False
        for i in range(num_rollouts):
            # Get full text and boxed value
            raw_full = (
                final_results[i]
                if final_results[i] is not None
                else assistant_msgs[i]["content"]
            )
            boxed_val = self._boxed_after_think(raw_full)
            # Determine correctness against expected
            is_correct = False
            if expr is not None and boxed_val is not None:
                is_correct = boxed_val == expr or self._canon_num(
                    boxed_val
                ) == self._canon_num(expr)
            # Choose color and label
            if is_correct:
                label = "CORRECT"
                lbl_color = "\033[92m"
                any_success = True
            elif boxed_val is not None:
                label = "WRONG"
                lbl_color = "\033[93m"
            else:
                label = "NO_BOX"
                lbl_color = "\033[91m"
            reset = "\033[0m"
            last = last_turns[i]
            print()
            print(f"\033[93m--- ROLLOUT {i} ---\033[0m")
            print(f"Result: {lbl_color}{label}{reset}")
            # Last turn content
            print("Last turn output:")
            print(f"\033[96m{last}\033[0m")
            # Boxed vs expected
            print(f"Boxed answer: {boxed_val!r}")
            print(f"Expected answer: {expr!r}")

        if not any_success:
            print(
                f"⚠️ All {num_rollouts} rollouts failed to produce a boxed answer. Invalidating group."
            )
            return None, []
        # -- End summary --

        for rollout_idx in range(num_rollouts):
            try:
                raw = (
                    final_results[rollout_idx]
                    if final_results[rollout_idx] is not None
                    else assistant_msgs[rollout_idx]["content"]
                )

                toks = self.tokenizer.encode(raw)
                if len(toks) > MAX_REPLY_TOKENS:
                    toks = toks[:MAX_REPLY_TOKENS]
                    raw = self.tokenizer.decode(toks)

                final_assistant_msg = {"role": "assistant", "content": raw}
                full_ctx: List[Message] = prompt_msgs + [final_assistant_msg]

                expr = (
                    expected["arguments"]["code"][6:-1]
                    if (
                        isinstance(expected, dict)
                        and "arguments" in expected
                        and "code" in expected["arguments"]
                        and expected["arguments"]["code"].startswith("print(")
                        and expected["arguments"]["code"].endswith(")")
                    )
                    else None
                )
                boxed = self._boxed_after_think(raw)

                same = boxed == expr or (
                    boxed and expr and self._canon_num(boxed) == self._canon_num(expr)
                )
                reward = 1.0 if same else -1.0

                if "</think>" not in raw:
                    reward = -1.0
                else:
                    end_pos = raw.lower().find("</think>")
                    # Check for tool calls or responses after </think>
                    if (
                        "<tool_call" in raw[end_pos + len("</think>") :].lower()
                        or "<tool_response" in raw[end_pos + len("</think>") :].lower()
                    ):
                        reward = -1.0
                    # Add bonus for tool usage if the completion was successful
                    elif reward > 0 and len(executed_tools[rollout_idx]) > 0:
                        print(
                            f"🌟 [ROLLOUT {rollout_idx}] Adding tool usage bonus (+{TOOL_USAGE_BONUS})"
                        )
                        reward += TOOL_USAGE_BONUS

                tok = tokenize_for_trainer(self.tokenizer, full_ctx)
                scored["tokens"].append(tok["tokens"])
                scored["masks"].append(tok["masks"])
                scored["scores"].append(reward)
                self.percent_correct_buffer.append(max(reward, 0))

                # Add successful completions to dynamic pool regardless of number of turns
                if (
                    reward >= 1.0
                ):  # This will now include both 1.0 and 1.0 + TOOL_USAGE_BONUS
                    u = {"role": "user", "content": prompt_msgs[-1]["content"]}
                    a = {"role": "assistant", "content": raw}
                    self.dynamic_pool.append((u, a))
                    if len(self.dynamic_pool) > self.dynamic_pool_max:
                        self.dynamic_pool.pop(0)

            except Exception:
                scored["tokens"].append([])
                scored["masks"].append([])
                scored["scores"].append(-1.0)
                self.percent_correct_buffer.append(0.0)

        print(
            "\n🏁 [EXECUTION MODE] Completed all rollouts. Average reward: \n"
            f"{sum(scored['scores'])/len(scored['scores']):.3f}"
        )

        # -- Per-rollout score summary --
        print("\n\033[96m📊 Rollout score summary:\033[0m")
        reset = "\033[0m"
        for i, score in enumerate(scored["scores"]):
            color = "\033[92m" if score > 0 else "\033[91m"
            print(f"  \033[93m[ROLLOUT {i}]\033[0m Score: {color}{score}{reset}")

        # Add warning if all rollouts failed
        if all(score < 0 for score in scored["scores"]):
            print(
                f"⚠️ [WARNING] All {len(scored['scores'])} rollouts failed with negative rewards!"
            )
            print(
                "   This may indicate a problem with the model, prompt, or token budget."
            )
            # Signal failure to the outer loop
            return None, []

        return scored, []

    # --------------------- evaluation loop -------------------------------- #
    async def evaluate(self, *_, **__):
        """
        Simple eval: run one rollout per test item, compute binary correctness
        based on the boxed answer.  Adds a metric 'eval/percent_correct'.
        """
        if not hasattr(self, "test"):
            return

        total, correct = 0, 0
        for sample in self.test:
            # Build prompt exactly like get_next_item but without mutating self.iter
            prompt_text = sample["problem"]
            expr = sample["expected_answer"].strip()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            comp = await self.server.completion(
                prompt=prompt,
                n=1,
                max_tokens=1024,
                temperature=0.0,
                split="eval",
            )
            model_reply = comp.choices[0].text
            boxed = self._boxed_after_think(model_reply)
            if boxed and boxed == expr:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        self.eval_metrics.append(("eval/percent_correct", accuracy))

    # --------------------- dataset iterator ------------------------------- #
    async def get_next_item(self):
        idx = self.rng.randint(0, len(self.train) - 1)
        sample = self.train[idx]

        prompt_text = sample["problem"]
        expr = sample["expected_answer"].strip()
        answer_call = {
            "name": "python_interpreter",
            "arguments": {"code": f"print({expr})"},
        }

        # ---------------- few‑shot demonstration ---------------- #
        fewshot_user = {
            "role": "user",
            "content": "Compute the integral of x^2 from 0 to 1.",
        }
        fewshot_assistant = {
            "role": "assistant",
            "content": (
                "<think>\n"
                "Let's evaluate the definite integral ∫₀¹ x² dx. This is a basic power rule integral.\n"
                "We know:\n"
                "∫ xⁿ dx from a to b = [xⁿ⁺¹ / (n+1)] from a to b.\n"
                "So for x²:\n"
                "= [x³ / 3] from 0 to 1\n"
                "= (1³ / 3) - (0³ / 3) = 1/3 - 0 = 1/3\n"
                "That checks out, but let's confirm with SymPy just to be sure.\n"
                '<tool_call>{"name":"python_interpreter", '
                '"arguments":{"code":'
                '"import sympy as sp\\n'
                "x=sp.symbols('x')\\n"
                'print(sp.integrate(x**2,(x,0,1)))"}}\n'
                "</tool_call>\n"
                '<tool_response>{"result": 1/3}</tool_response>\n'
                "The interpreter returns 1/3, so the value is 0.333̅.\n"
                "</think>\n\n"
                "The integral equals \\boxed{\\tfrac{1}{3}} \\approx 0.333."
            ),
        }

        # --- second tiny example: simple arithmetic with calculator ---- #
        fewshot_user2 = {"role": "user", "content": "What is (2 + 3) * 4 ?"}
        fewshot_assistant2 = {
            "role": "assistant",
            "content": (
                "<think>\n"
                "I need (2+3)*4.  Quick mental math gives 5*4 = 20, "
                "but I'll confirm with the calculator tool.\n"
                '<tool_call>{"name":"calculator", '
                '"arguments":{"expr":"(2+3)*4"}}</tool_call>\n'
                '<tool_response>{"value": 20}</tool_response>\n'
                "The tool also says 20, matching my head‑math.\n"
                "</think>\n\n"
                "Therefore the answer is \\boxed{20}."
            ),
        }

        # --------------- build final prompt messages ------------ #
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        real_user = {
            "role": "user",
            "content": (
                f"{prompt_text} \n"
                "This is a math problem, you must use the python_interpreter or calculator tool call to solve it."
                # "Before you call the tools, try to solve it step-by-step and then use the tool to verify"
            ),
        }

        # Optionally insert one real demo from dynamic_pool
        dyn = list(self.dynamic_pool[-1]) if self.dynamic_pool else []

        messages = (
            [
                system_msg,
                fewshot_user,
                fewshot_assistant,
                fewshot_user2,
                fewshot_assistant2,
            ]
            + dyn
            + [real_user]
        )

        # Freeze for hashing
        frozen = tuple(frozenset(m.items()) for m in messages)

        return (frozen, answer_call)

    # --------------------- wandb logging ---------------------------------- #
    async def create_rollout_table(self, metrics):
        if self.rollouts_for_wandb:
            table = wandb.Table(columns=["text", "score"])
            for grp in self.rollouts_for_wandb:
                for txt, sc in grp:
                    table.add_data(txt, sc)
            metrics["train/rollouts"] = table
        self.rollouts_for_wandb = []
        return metrics

    async def wandb_log(self, metrics: Dict = None):
        metrics = metrics or {}
        if self.percent_correct_buffer:
            metrics["train/percent_correct"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
        self.percent_correct_buffer = []
        for k, v in self.eval_metrics:
            metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(metrics)


if __name__ == "__main__":
    InterleavedInlineEnv.cli()
