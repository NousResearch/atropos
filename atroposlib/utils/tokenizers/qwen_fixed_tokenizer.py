"""
Custom Qwen tokenizer wrapper with fixed Jinja2 template.

This wrapper overrides the chat_template to avoid Jinja2 sandbox restrictions
that prevent list.append() operations in the original Qwen tokenizer.

TLDR; tool calls with Qwen are a PITA
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer


class QwenFixedTokenizer:
    """Wrapper around Qwen tokenizer with fixed chat template."""

    @classmethod
    def _load_chat_template(cls) -> str:
        """Load the chat template from the .jinja file."""
        template_path = Path(__file__).parent / "qwen_chat_template.jinja"
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()

    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize the tokenizer wrapper.

        Args:
            model_name_or_path: Model name or path to load tokenizer from
            **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained
        """
        # Load the base tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

        # Override the chat template with our fixed version
        self.tokenizer.chat_template = self._load_chat_template()

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying tokenizer."""
        return getattr(self.tokenizer, name)

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, bool]] = None,
        return_dict: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        """Apply the fixed chat template.

        This method delegates to the underlying tokenizer's apply_chat_template
        but ensures our fixed template is used.
        """
        return self.tokenizer.apply_chat_template(
            conversation,
            tools=tools,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
