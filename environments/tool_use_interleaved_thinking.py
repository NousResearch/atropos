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

import itertools
import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Set to True to always print debug information.
DEBUG = True  # or toggle via env var if you prefer: bool(os.getenv("DEBUG_INTERLEAVED", "1"))

# Hard caps for generation length
MAX_REPLY_TOKENS = 2048  # truncate any single assistant reply to ≤1024 tokens
MAX_GEN_PER_TURN = 1024  # never request more than 512 new tokens from the model

import wandb
from datasets import Dataset, load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# -------------------------------------------------------------------------- #
#  Constants
# -------------------------------------------------------------------------- #
system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the "
    "problem and deliberate with yourself via systematic reasoning processes to help come to a correct "
    "solution prior to answering. You should enclose your thoughts and internal monologue inside <think> "
    "</think> tags, and then provide your solution or response to the problem."
)

TOOL_SYSTEM_PROMPT = """
You are a function calling & reasoning AI model. You are provided with function signatures within <reasoning_tools> </reasoning_tools> XML tags for internal reasoning tools. After calling & executing the functions, you will be provided with function results within <tool_response> </tool_response> XML tags. Here are the available tools:

<reasoning_tools>
[
  {
    "type": "function",
    "function": {
      "name": "calculator",
      "description": "Evaluate a numeric Python expression and return the result.",
      "parameters": {
        "type": "object",
        "properties": {
          "expr": {
            "type": "string",
            "description": "A pure‑Python arithmetic expression, e.g. '3*(4+5)'"
          }
        },
        "required": ["expr"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "python_interpreter",
      "description": "Run a short Python snippet and return stdout plus the last expression.",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "Python source code to execute."
          }
        },
        "required": ["code"]
      }
    }
  }
]
</reasoning_tools>

You must use reasoning tools such as python_interpreter as a tool call when available for hard problems such as math before providing your final answer.
Always wrap your final numeric answer (or final result) in \\boxed{...} so it can be automatically graded.

For reasoning tools, return interleaved tool calls within <think> </think> tags.
<think>
<tool_call>\n{'name': <function-name>, 'arguments': <args-dict>}\n</tool_call>
<!-- system pauses runtime for execution -->
<tool_response>\n{'result': <result>}\n</tool_response>
<!-- assistant resumes within same think -->
</think>
"""

SYSTEM_PROMPT = system_prompt + TOOL_SYSTEM_PROMPT


# -------------------------------------------------------------------------- #
#  Environment
# -------------------------------------------------------------------------- #
class InterleavedInlineEnv(BaseEnv):
    """
    One episode = user prompt → single assistant message with inline tool
    calls inside a still‑open <think> block.
    """

    name = "interleaved_inline"
    _re_last_call = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>\s*$", re.S)

    # --------------------- BaseEnv boiler‑plate --------------------------- #
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
        self.dynamic_pool_max = 4  # keep at most 4 real examples

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
            max_token_length=16 * 1024,
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
        import os
        import re

        N = int(os.getenv("SUBSET_ROWS", "1000"))

        stream_ds = load_dataset(  # ≈50 k rows total → stream
            "NVIDIA/OpenMathReasoning",
            split="cot",
            # "open-r1/OpenR1-Math-220k",
            # "nvidia/AceReason-Math",
            # split="train",
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
        self.train, self.test = split["train"], split["test"]
        self.train = self.train.shuffle(seed=int.from_bytes(os.urandom(2), "big"))

    # --------------------- helper methods --------------------------------- #
    async def _completion_until(
        self, prompt: str, max_tokens: int, stop: Optional[str] = None
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
        """
        matches = self._re_last_call.findall(chunk)
        if not matches:
            return None
        try:
            return json.loads(matches[-1])
        except Exception:
            return None

    # ---- helper: canonicalize numeric strings (remove commas & spaces) ----
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
            import asyncio

            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {"code": args["code"], "input": ""}
                resp = await client.post("http://localhost:5002/execute", json=payload)
                data = resp.json()
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

    # --------------------- rollout logic (interleaved) ------------------- #
    async def _run_one_episode(self, ctx: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate–execute–resume loop:

        assistant emits  <think> … <tool_call> …, generation stops at
        </tool_call>.  We parse & execute, append <tool_response> right after
        the call *inside the same assistant message*, then continue
        generation.  Loop ends when </think> is produced.
        """
        if DEBUG:
            print("\033[93m---------- NEW EPISODE (interleaved) ----------\033[0m")

        executed: List[Dict] = []
        first_turn = True
        assistant_msg = {"role": "assistant", "content": ""}  # buffer

        while True:
            # Build prompt
            prompt_txt = self.tokenizer.apply_chat_template(
                ctx + [assistant_msg],
                add_generation_prompt=first_turn,  # header only before first gen
                tokenize=False,
            )
            first_turn = False

            if DEBUG:
                clean_prompt = prompt_txt.replace("<|eot_id|>", "")
                print(
                    f"\n\033[93m▶ PROMPT (tokens {len(prompt_txt)}):\033[0m "
                    f"\033[92m{clean_prompt}\033[0m\n{'-'*60}"
                )

            # Stop at </tool_call>  OR   </think>
            reply = await self._completion_until(
                prompt_txt,
                max_tokens=MAX_GEN_PER_TURN,
                stop=["</tool_call>", "</think>"],
            )
            if DEBUG:
                print(
                    f"\033[93m◆ MODEL CHUNK:\033[0m "
                    f"\033[94m{reply}\033[0m\n{'='*60}"
                )

            assistant_msg["content"] += reply

            # Did the model stop because of </tool_call> ?
            if assistant_msg["content"].strip().endswith("</tool_call>"):
                call_json = self._extract_last_call(assistant_msg["content"])
                if call_json is None:
                    # malformed, continue generation
                    continue
                # Execute
                result = await self._exec_tool(call_json)
                executed.append(call_json)

                # Append tool_response inline
                assistant_msg[
                    "content"
                ] += f"\n<tool_response>{json.dumps(result)}</tool_response>\n"
                # continue loop (model will keep thinking)
                continue

            # Otherwise, stop if </think> produced
            if "</think>" in assistant_msg["content"]:
                break

        # Push final assistant message into ctx
        ctx.append(assistant_msg)
        return ctx, executed

    # --------------------- scoring & data --------------------------------- #
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

        scored = ScoredDataGroup(tokens=[], masks=[], scores=[])
        for idx, choice in enumerate(completions.choices):
            raw = choice.text or ""
            toks = self.tokenizer.encode(raw)
            if len(toks) > MAX_REPLY_TOKENS:
                toks = toks[:MAX_REPLY_TOKENS]
                raw = self.tokenizer.decode(toks)
                if DEBUG:
                    print(f"[DEBUG] truncated reply {idx} to {len(toks)} tokens")
            assistant_msg = {"role": "assistant", "content": raw}

            full_ctx = prompt_msgs + [assistant_msg]

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
            boxed = self._boxed_after_think(raw)

            same = boxed == expr or (
                boxed and expr and self._canon_num(boxed) == self._canon_num(expr)
            )
            reward = 1.0 if same else -1.0
            if "</think>" not in raw:
                reward = -1.0  # invalid – did not close think block
            else:
                # NEW RULE: no tool_call tags are allowed *outside* the think block
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
            has_call = (
                "<tool_call" in reply_txt.lower()
            )  # ensure an interleaved call exists
            if sc >= 1.0 and has_call:
                # Build (user, assistant) pair from this successful rollout
                u = {"role": "user", "content": prompt_msgs[-1]["content"]}
                a = {"role": "assistant", "content": reply_txt}
                self.dynamic_pool.append((u, a))
                if len(self.dynamic_pool) > self.dynamic_pool_max:
                    self.dynamic_pool.pop(0)  # FIFO
                break  # only harvest one per group

        return scored, []

    # --------------------- evaluation loop -------------------------------- #
    async def evaluate(self, *_, **__):
        """
        Simple eval: run one rollout per test item, compute binary correctness
        based on the boxed answer.  Adds a metric 'eval/percent_correct'.
        """
        if not hasattr(self, "test"):
            return  # setup not yet called in some dry‑run modes

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
        # ----- choose random sample from training set -----
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
                "Let's be sure of the definite integral ∫₀¹ x² dx.  It's easy by hand "
                "but I'll run SymPy to avoid mistakes.\n"
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
                f"{prompt_text} \nThis is a math problem, you must use the python_interpreter or calculator tool call to solve it."
            ),
        }

        # Optionally insert one real demo from dynamic_pool
        dyn = list(self.dynamic_pool[-1]) if self.dynamic_pool else []

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
            + dyn  # 0 or 2 msgs
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


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    InterleavedInlineEnv.cli()
