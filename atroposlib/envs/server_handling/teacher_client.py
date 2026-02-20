import os
from typing import Dict, List, Optional, Tuple

import aiohttp


class TeacherClient:
    """
    Transport/parsing client for teacher top-k logprobs.

    This keeps distillation HTTP and parsing logic out of BaseEnv.
    """

    def __init__(self, config, tokenizer, logger):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger

    async def get_teacher_logprobs(
        self,
        token_sequences: List[List[int]],
        messages_list: Optional[List[List[Dict]]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
        self.logger.info(
            "[TEACHER] get_teacher_logprobs called with %s sequences",
            len(token_sequences),
        )
        self.logger.info("[TEACHER] teacher_base_url=%s", self.config.teacher_base_url)

        if not self.config.teacher_base_url:
            self.logger.warning("[TEACHER] No teacher_base_url configured, returning empty")
            return [], []

        if top_k is None:
            top_k = self.config.teacher_top_k

        api_key = self.config.teacher_api_key or os.environ.get("TEACHER_API_KEY", "")
        model_name = self.config.teacher_model_name or "default"
        self.logger.info("[TEACHER] Using model=%s, top_k=%s", model_name, top_k)

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        token_id_results: List[List[List[int]]] = []
        logprob_results: List[List[List[float]]] = []

        try:
            async with aiohttp.ClientSession() as session:
                for i, tokens in enumerate(token_sequences):
                    self.logger.info(
                        "[TEACHER] Processing sequence %s/%s, %s tokens",
                        i + 1,
                        len(token_sequences),
                        len(tokens),
                    )
                    base_text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                    steering_prefix = ""
                    if self.config.teacher_system_prompt:
                        steering_prefix += (
                            "System instruction:\n"
                            f"{self.config.teacher_system_prompt.strip()}\n\n"
                        )
                    if self.config.teacher_prefix_text:
                        steering_prefix += self.config.teacher_prefix_text
                    full_text = steering_prefix + base_text
                    prefix_token_len = (
                        len(
                            self.tokenizer.encode(
                                steering_prefix, add_special_tokens=False
                            )
                        )
                        if steering_prefix
                        else 0
                    )

                    request_data = {
                        "model": model_name,
                        "prompt": full_text,
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "logprobs": top_k,
                        "echo": True,
                    }
                    try:
                        async with session.post(
                            f"{self.config.teacher_base_url}/completions",
                            json=request_data,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=120),
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                seq_token_ids, seq_logprobs = self._parse_completion_logprobs(
                                    data, top_k
                                )
                                if seq_token_ids and seq_logprobs:
                                    aligned_ids, aligned_lps = self._align_teacher_topk_to_tokens(
                                        seq_token_ids,
                                        seq_logprobs,
                                        target_token_len=len(tokens),
                                        prefix_token_len=prefix_token_len,
                                    )
                                    token_id_results.append(aligned_ids)
                                    logprob_results.append(aligned_lps)
                                    continue
                    except Exception:
                        pass

                    if messages_list and i < len(messages_list):
                        messages = list(messages_list[i])
                        if self.config.teacher_system_prompt:
                            messages = [
                                {
                                    "role": "system",
                                    "content": self.config.teacher_system_prompt,
                                }
                            ] + messages
                    else:
                        messages = []
                        if self.config.teacher_system_prompt:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": self.config.teacher_system_prompt,
                                }
                            )
                        messages.append({"role": "user", "content": full_text})

                    chat_request = {
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "logprobs": True,
                        "top_logprobs": top_k,
                    }
                    try:
                        async with session.post(
                            f"{self.config.teacher_base_url}/chat/completions",
                            json=chat_request,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=120),
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                seq_token_ids, seq_logprobs = self._parse_chat_logprobs(
                                    data, top_k
                                )
                                if seq_token_ids and len(seq_token_ids) >= len(tokens):
                                    aligned_ids, aligned_lps = self._align_teacher_topk_to_tokens(
                                        seq_token_ids,
                                        seq_logprobs,
                                        target_token_len=len(tokens),
                                        prefix_token_len=0,
                                    )
                                else:
                                    aligned_ids = [[] for _ in range(len(tokens))]
                                    aligned_lps = [[] for _ in range(len(tokens))]
                                token_id_results.append(aligned_ids)
                                logprob_results.append(aligned_lps)
                            else:
                                self.logger.warning(
                                    "Teacher API returned %s", response.status
                                )
                                token_id_results.append([[] for _ in range(len(tokens))])
                                logprob_results.append([[] for _ in range(len(tokens))])
                    except Exception as e:
                        self.logger.warning("Teacher chat request failed: %s", e)
                        token_id_results.append([[] for _ in range(len(tokens))])
                        logprob_results.append([[] for _ in range(len(tokens))])

            return token_id_results, logprob_results
        except Exception as e:
            self.logger.error("Error fetching teacher logprobs: %s", e)
            return [], []

    def _align_teacher_topk_to_tokens(
        self,
        seq_token_ids: List[List[int]],
        seq_logprobs: List[List[float]],
        target_token_len: int,
        prefix_token_len: int = 0,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        n = min(len(seq_token_ids), len(seq_logprobs))
        aligned_ids = list(seq_token_ids[:n])
        aligned_lps = list(seq_logprobs[:n])

        if prefix_token_len > 0:
            aligned_ids = aligned_ids[prefix_token_len:]
            aligned_lps = aligned_lps[prefix_token_len:]

        aligned_ids = aligned_ids[:target_token_len]
        aligned_lps = aligned_lps[:target_token_len]

        if len(aligned_ids) < target_token_len:
            pad_count = target_token_len - len(aligned_ids)
            aligned_ids.extend([[] for _ in range(pad_count)])
            aligned_lps.extend([[] for _ in range(pad_count)])

        return aligned_ids, aligned_lps

    def _parse_completion_logprobs(
        self, data: Dict, top_k: int
    ) -> Tuple[List[List[int]], List[List[float]]]:
        try:
            choice = data.get("choices", [{}])[0]
            logprobs_data = choice.get("logprobs", {})
            top_logprobs = logprobs_data.get("top_logprobs", [])
            if not top_logprobs:
                return [], []

            seq_token_ids: List[List[int]] = []
            seq_logprobs: List[List[float]] = []
            for pos_logprobs in top_logprobs:
                if pos_logprobs is None:
                    seq_token_ids.append([])
                    seq_logprobs.append([])
                elif isinstance(pos_logprobs, dict):
                    sorted_items = sorted(
                        pos_logprobs.items(), key=lambda x: x[1], reverse=True
                    )[:top_k]
                    pos_ids: List[int] = []
                    pos_lps: List[float] = []
                    for token_str, logprob in sorted_items:
                        token_ids = self.tokenizer.encode(
                            token_str, add_special_tokens=False
                        )
                        if token_ids:
                            pos_ids.append(int(token_ids[0]))
                            pos_lps.append(float(logprob))
                    seq_token_ids.append(pos_ids)
                    seq_logprobs.append(pos_lps)
                else:
                    seq_token_ids.append([])
                    seq_logprobs.append([])
            return seq_token_ids, seq_logprobs
        except Exception as e:
            self.logger.warning("Error parsing completion logprobs: %s", e)
            return [], []

    def _parse_chat_logprobs(
        self, data: Dict, top_k: int
    ) -> Tuple[List[List[int]], List[List[float]]]:
        try:
            choice = data.get("choices", [{}])[0]
            logprobs_data = choice.get("logprobs", {})
            if not logprobs_data:
                return [], []

            content = logprobs_data.get("content", [])
            seq_token_ids: List[List[int]] = []
            seq_logprobs: List[List[float]] = []
            for token_data in content:
                top_logprobs = token_data.get("top_logprobs", [])
                pos_ids: List[int] = []
                pos_lps: List[float] = []
                for item in top_logprobs[:top_k]:
                    token_str = item.get("token", "")
                    logprob = item.get("logprob", 0.0)
                    token_ids = self.tokenizer.encode(
                        token_str, add_special_tokens=False
                    )
                    if token_ids:
                        pos_ids.append(int(token_ids[0]))
                        pos_lps.append(float(logprob))
                seq_token_ids.append(pos_ids)
                seq_logprobs.append(pos_lps)
            return seq_token_ids, seq_logprobs
        except Exception as e:
            self.logger.warning("Error parsing chat logprobs: %s", e)
            return [], []
