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
    
    @classmethod
    def intercept_response(cls, raw_output: str) -> str:
        """
        Surgically extracts the action.
        1. Find the <action> block (handling [SYSTEM: ] prefixes and missing closing tags).
        2. If found, return only that.
        3. If not found, return the text with reasoning tags stripped.
        """
        if not raw_output:
            return ""

        # 1. Try to find an action block (case-insensitive, handles [SYSTEM: <action>])
        # We look for <action> and take everything until </action> OR the end of the string
        # This is the most reliable way to get the tool call while ignoring reasoning noise.
        action_pattern = r"(?:\[SYSTEM:\s*)?<action>(.*?)(?:</action>|$)"
        action_match = re.search(action_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        
        if action_match:
            # We found the action! Return it (re-wrapped for environment consistency)
            action_content = action_match.group(1).strip()
            # If the model left a trailing ] from the [SYSTEM: prefix, clean it
            if action_content.endswith("]"):
                action_content = action_content[:-1].strip()
            return f"<action>{action_content}</action>"

        # 2. Fallback: No action block found, so just strip the noise
        clean_text = cls.strip_ssot_tags(raw_output)
        return clean_text

    @staticmethod
    def strip_ssot_tags(text: str) -> str:
        """
        Basic stripping of known tags.
        """
        if not text:
            return ""
        # Remove SSoT-specific blocks
        clean_text = re.sub(r"(?:\[SYSTEM:\s*)?<random_string>.*?(?:</random_string>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r"(?:\[SYSTEM:\s*)?<thinking>.*?(?:</thinking>|$)", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
        # Clean up any lingering brackets
        clean_text = clean_text.replace("]", "").strip()
        return clean_text
