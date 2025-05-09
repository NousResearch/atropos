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
            logger.debug(
                f"[{self.name}] Checking completion {idx}: Content='{content[:100]}...' Tags={self.preferred_tags}, RequireAll={self.require_all_tags}"
            )

            all_required_tags_valid = True
            any_preferred_tag_valid = False

            for tag_idx, tag_name in enumerate(self.preferred_tags):
                # Find all non-overlapping instances of <tag_name>content</tag_name>
                # (.*?) captures the content between the tags.
                pattern_for_finding_instances = f"<{tag_name}>(.*?)</{tag_name}>"
                instances_of_current_tag = re.findall(
                    pattern_for_finding_instances, content, flags
                )

                if not instances_of_current_tag:
                    logger.debug(
                        f"  -> Tag '{tag_name}': No instances found (e.g., <{tag_name}>...</{tag_name}>). Missing."
                    )
                    if self.require_all_tags:
                        all_required_tags_valid = False
                        break
                    else:
                        continue

                # Check if at least one instance of this tag_name has valid (non-whitespace) content
                current_tag_type_has_valid_instance = False
                for instance_content in instances_of_current_tag:
                    if re.search(
                        r"\S", instance_content
                    ):  # Check if the *specific* inner_content has a non-whitespace
                        current_tag_type_has_valid_instance = True
                        logger.debug(
                            f"  -> Tag '{tag_name}': Found an instance with valid non-whitespace content: '{instance_content[:50]}...'"
                        )
                        break

                if current_tag_type_has_valid_instance:
                    if not self.require_all_tags:
                        any_preferred_tag_valid = True
                        logger.debug(
                            f"  -> Tag '{tag_name}': Valid instance found, and require_all_tags is False. Marking completion as valid format."
                        )
                        break
                else:
                    logger.debug(
                        f"  -> Tag '{tag_name}': Instances found, but ALL were empty or whitespace-only."
                    )
                    if self.require_all_tags:
                        all_required_tags_valid = False
                        break

            current_reward = -1.0
            if self.require_all_tags:
                if all_required_tags_valid:
                    current_reward = 1.0
                    logger.debug(
                        f"  -> Final check (require_all_tags=True): All required tags were valid. Reward: {current_reward}"
                    )
                else:
                    logger.debug(
                        f"  -> Final check (require_all_tags=True): Not all required tags were valid. Reward: {current_reward}"
                    )
            else:
                if any_preferred_tag_valid:
                    current_reward = 1.0
                    logger.debug(
                        f"  -> Final check (require_all_tags=False): At least one preferred tag was valid. Reward: {current_reward}"
                    )
                else:
                    logger.debug(
                        f"  -> Final check (require_all_tags=False): No preferred tags were valid. Reward: {current_reward}"
                    )

            rewards.append(current_reward)

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
