import re
import random
from typing import List, Dict, Any

class SSoTExplorationWrapper:
    """
    SSoTExplorationWrapper for RL environment policy.
    Implements Reasoning-Guided Exploration using the String Seed of Thought (SSoT) protocol.
    """

    def __init__(self, epsilon: float = 0.15, random_string_len: int = 16):
        self.epsilon = epsilon
        self.random_string_len = random_string_len
        self.ssot_instruction = (
            f"\n\n[SYSTEM: STRATEGIC EXPLORATION ACTIVE]\n"
            f"To ensure diversity-aware decision making, you MUST first generate a {self.random_string_len}-character "
            f"alphanumeric <random_string>. Then, perform a Polynomial Rolling Hash in a <thinking> block to map "
            f"this entropy to a specific sub-strategy. Finally, provide your action in the <action>...</action> tags."
        )

    def should_inject_ssot(self, turn_idx: int) -> bool:
        """Inject at root state (Turn 0) or probabilistically based on epsilon."""
        if turn_idx == 0:
            return True
        return random.random() < self.epsilon

    def wrap_prompt(self, prompt: str, turn_idx: int) -> str:
        """
        Appends SSoT instruction to the prompt if exploration is triggered for this turn.
        Injected at the end of the prompt to preserve KV cache prefix caching.
        """
        if self.should_inject_ssot(turn_idx):
            return prompt + self.ssot_instruction
        return prompt

    def flush_context(self, history: List[Dict[str, Any]], raw_output: str, parsed_action: str) -> List[Dict[str, Any]]:
        """
        Prevents SSoT entropy and CoT math from being appended to the multi-turn history.
        Appends only the deterministic parsed_action to the context.
        """
        history.append({
            "role": "assistant",
            "content": parsed_action
        })
        return history

class ActionParserInterceptor:
    """
    Interceptor to clean LLM outputs.
    Regex-strips <random_string> and <thinking> blocks from the policy's raw output.
    Returns ONLY the parsed <action>...</action> payload.
    """
    
    @staticmethod
    def strip_ssot_tags(text: str) -> str:
        """
        Regex-strips <random_string> and <thinking> blocks from text.
        Case-insensitive and handles multi-line blocks.
        """
        if not text:
            return ""
        # Remove SSoT-specific blocks
        clean_text = re.sub(r'<random_string>.*?</random_string>', '', text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<random_seed>.*?</random_seed>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r'<thinking>.*?</thinking>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
        return clean_text.strip()

    @classmethod
    def intercept_response(cls, raw_output: str) -> str:
        # Remove SSoT-specific blocks to prevent noise from entering the reward model/environment
        clean_text = cls.strip_ssot_tags(raw_output)
        
        # Extract strictly the action payload
        action_match = re.search(r'<action>(.*?)</action>', clean_text, re.DOTALL | re.IGNORECASE)
        if action_match:
            # Re-wrap in action tags for environment consistency
            return f"<action>{action_match.group(1).strip()}</action>"
        
        # Fallback if the model hallucinated the tag structure, though FSM usually prevents this
        return clean_text
