"""Reward function for checking if completions have specific XML-style tags."""

import logging
import re
from typing import Any, List, Optional

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class FormatReward(RewardFunction):
    """Reward function that checks if completions have XML-style tags."""

    def __init__(
        self,
        preferred_tags: Optional[List[str]] = None,
        require_all_tags: bool = False,
        case_sensitive: bool = False,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the format reward function.

        Args:
            preferred_tags: List of tag names to search for (defaults to ['think', 'answer'])
            require_all_tags: If True, require all tags to be present for a reward
            case_sensitive: If True, perform case-sensitive tag matching
            weight: Weight for this reward
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.preferred_tags = preferred_tags or ["think", "answer"]
        self.require_all_tags = require_all_tags
        self.case_sensitive = case_sensitive

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Check if completions have the expected XML-style tags.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context

        Returns:
            List of rewards for each completion (1.0 for good format, -1.0 otherwise)
        """
        completion_contents = [
            self.get_content(completion) for completion in completions
        ]

        rewards = []
        flags = 0 if self.case_sensitive else re.IGNORECASE
        flags |= re.DOTALL  # Allow . to match newlines

        for idx, content in enumerate(completion_contents):
            logger.debug(f"[{self.name}] Checking completion {idx}: Content='{content[:100]}...' Tags={self.preferred_tags}, RequireAll={self.require_all_tags}")
            current_reward = -1.0 # Default to penalty
            
            if self.require_all_tags:
                all_tags_present = True
                missing_tag = None
                for tag in self.preferred_tags:
                    pattern = f"<{tag}>.*?</{tag}>"
                    if not re.search(pattern, content, flags):
                        all_tags_present = False
                        missing_tag = tag
                        logger.debug(f"  -> Tag '{tag}' not found.")
                        break
                if all_tags_present:
                    current_reward = 1.0
                    logger.debug(f"  -> All required tags found.")
            else:
                has_tags = False
                found_tag = None
                for tag in self.preferred_tags:
                    pattern = f"<{tag}>.*?</{tag}>"
                    if re.search(pattern, content, flags):
                        has_tags = True
                        found_tag = tag
                        logger.debug(f"  -> Tag '{tag}' found.")
                        break
                if has_tags:
                    current_reward = 1.0
                else:
                    logger.debug(f"  -> No preferred tags found.")
            
            rewards.append(current_reward)
            logger.debug(f"  -> Determined reward for completion {idx}: {current_reward}")

        # Weight is applied by the RewardFunction base class __call__ method.
        # This compute method should return raw scores.
        return rewards


# Legacy function for backward compatibility
def format_reward(
    completions: List[Any], preferred_tags: Optional[List[str]] = None, **kwargs
) -> List[float]:
    """
    Legacy function wrapper for FormatReward.

    Args:
        completions: List of completions to evaluate
        preferred_tags: List of tag names to search for (defaults to ['think', 'answer'])
        **kwargs: Additional keyword arguments

    Returns:
        List of rewards for each completion (1.0 for good format, 0.0 otherwise)
    """
    reward_fn = FormatReward(preferred_tags=preferred_tags)
    return reward_fn.compute(completions, **kwargs)
