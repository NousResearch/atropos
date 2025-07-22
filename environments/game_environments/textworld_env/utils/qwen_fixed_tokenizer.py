"""
Custom Qwen tokenizer wrapper with fixed Jinja2 template.

This wrapper overrides the chat_template to avoid Jinja2 sandbox restrictions
that prevent list.append() operations in the original Qwen tokenizer.
"""

from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Union


class QwenFixedTokenizer:
    """Wrapper around Qwen tokenizer with fixed chat template."""
    
    # Fixed Jinja2 template that avoids sandbox restrictions
    FIXED_CHAT_TEMPLATE = """{%- if tools %}
{{- '<|im_start|>system\\n' }}
{%- if messages[0].role == 'system' %}
{{- messages[0].content + '\\n\\n' }}
{%- endif %}
{{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
{%- for tool in tools %}
{{- "\\n" }}
{{- tool | tojson }}
{%- endfor %}
{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
{%- if messages[0].role == 'system' %}
{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
{%- endif %}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'user' or (message.role == 'assistant' and message.tool_calls is not defined) or (message.role == 'tool' and loop.index > 1) %}
{%- if message.role == 'user' %}
{%- if tools and not loop.first %}
{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
{%- elif not tools or (tools and loop.first) %}
{{- '<|im_start|>user\\n' + message.content + '<|im_end|>\\n' }}
{%- endif %}
{%- elif message.role == 'assistant' %}
{{- '<|im_start|>assistant\\n' + message.content + '<|im_end|>\\n' }}
{%- elif message.role == 'tool' %}
{{- '<|im_start|>user\\n<tool_response>\\n' + message.content + '\\n</tool_response><|im_end|>\\n' }}
{%- endif %}
{%- elif message.role == 'assistant' and message.tool_calls is defined %}
{{- '<|im_start|>assistant\\n' }}
{%- if message.content %}
{{- message.content + '\\n\\n' }}
{%- endif %}
{%- for tool_call in message.tool_calls %}
{%- if tool_call.function is defined %}
{{- '<tool_call>\\n{\\"name\\": \\"' + tool_call.function.name + '\\", \\"arguments\\": ' + tool_call.function.arguments + '}\\n</tool_call>' }}
{%- endif %}
{%- endfor %}
{{- '<|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

    def __init__(self, model_name_or_path: str, **kwargs):
        """Initialize the tokenizer wrapper.
        
        Args:
            model_name_or_path: Model name or path to load tokenizer from
            **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained
        """
        # Load the base tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        
        # Override the chat template with our fixed version
        self.tokenizer.chat_template = self.FIXED_CHAT_TEMPLATE
        
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
        **kwargs
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
            **kwargs
        )