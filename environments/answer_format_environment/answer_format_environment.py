"""
Answer Format Environment

This environment trains models to generate responses in specific formats.
It focuses on format adherence rather than answer correctness, using randomized
format requirements and corresponding parsers.

Key Features:
- Randomized answer format selection from 150+ supported formats
- Strict thinking tag validation (exactly one <think></think> section)
- Format-specific parsers for validation
- Support for multiple input datasets that get shuffled together
- Dataset type-aware format selection (generic, math_only, code_only)
- Dynamic compositor system for complex structured responses
- Comprehensive data dumping and logging following environment conventions
- Format compliance scoring (1.0 for correct format, 0.0 for incorrect)
- Format success rate tracking and monitoring
- Weighted format selection for balanced training
- Optional equivalent ratio enforcement (stops generating formats after N successful groups)

Supported Answer Formats:
- Basic structured data: JSON, YAML, TOML (with confidence scores)
- XML/HTML tags: <answer>, <output>, <result>, nested variants
- LaTeX: \boxed{}, math mode expressions, align blocks, matrices
- Markdown: code blocks, bold, italic, headers, quotes
- Bracket notation: [], [[]], {}, (), <>
- Natural language: "The answer is:", "Final answer:", 15+ variants
- Programming formats: print(), console.log(), comments, docstrings
- Custom delimiters: |answer|, #answer#, _answer_, ~answer~
- Complex multi-tag formats: coding workflows, math derivations, research formats
- Dynamic compositor formats: randomly combined XML/JSON/YAML/TOML structures
- Special formats: TextArena [A], arrows =>, colons Answer:

Dataset Type Support:
- generic: All basic formats (JSON, XML, natural language, etc.)
- math_only: Generic formats + LaTeX math expressions, math workflows
- code_only: Generic formats + programming-specific formats, code workflows

Response Format Requirements:
- Must use exactly ONE <think> opening tag and ONE </think> closing tag
- All reasoning must be inside the thinking tags
- Answer must be in the specified format after </think>
- No additional <think> tags allowed after the first </think> closing tag

Example Usage:
    env_config = AnswerFormatEnvConfig(
        dataset_configs=[
            {
                "name": "your_dataset",
                "split": "train",
                "sample_size": 1000,
                "prompt_field": "question",
                "answer_field": "answer",
                "dataset_type": "generic"  # or "math_only", "code_only"
            }
        ],
        supported_formats=[AnswerFormat.JSON, AnswerFormat.XML],  # Optional filter
        eval_set_percentage=0.1,
        debug_logging=True,
        ensure_equivalent_ratios=True,  # Enable equivalent ratio enforcement
        format_group_threshold=50,  # Stop each format after 50 successful groups
    )
"""

import json
import logging
import os
import random
import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class AnswerFormat(Enum):
    """Enumeration of supported answer formats."""

    # Basic structured data formats (answer only)
    JSON = "json"  # {"answer": "content"}
    JSON_ARRAY = "json_array"  # ["answer"]
    JSON_SIMPLE = "json_simple"  # "answer"
    YAML = "yaml"  # answer: content
    YAML_LIST = "yaml_list"  # - content
    TOML = "toml"  # answer = "content"
    TOML_SECTION = "toml_section"  # [response] \n answer = "content"

    # Structured data with confidence scores
    JSON_CONFIDENCE = "json_confidence"  # {"answer": "content", "confidence": 0.9}
    YAML_CONFIDENCE = "yaml_confidence"  # answer: content \n confidence: 0.9
    TOML_CONFIDENCE = "toml_confidence"  # answer = "content" \n confidence = 0.9

    # XML/HTML tag variations (XML now uses answer tags)
    XML = "xml"  # <answer>content</answer>
    XML_FINAL_ANSWER = "xml_final_answer"  # <answer>Final Answer: content</answer>
    OUTPUT_TAGS = "output_tags"  # <output></output>
    RESULT_TAGS = "result_tags"  # <result></result>
    RESPONSE_TAGS = "response_tags"  # <response></response>
    FINAL_ANSWER_TAGS = "final_answer_tags"  # <final_answer></final_answer>
    SOLUTION_TAGS = "solution_tags"  # <solution></solution>
    CONCLUSION_TAGS = "conclusion_tags"  # <conclusion></conclusion>
    REPLY_TAGS = "reply_tags"  # <reply></reply>
    NESTED_RESPONSE_ANSWER = "nested_response_answer"  # <response>explanation\n<answer>content</answer></response>
    NESTED_SOLUTION_ANSWER = "nested_solution_answer"  # <solution>explanation\n<answer>content</answer></solution>
    NESTED_OUTPUT_RESULT = (
        "nested_output_result"  # <output>explanation\n<result>content</result></output>
    )
    NESTED_ANALYSIS_CONCLUSION = "nested_analysis_conclusion"  # <analysis>explanation\n<conclusion>content</conclusion></analysis> # noqa: E501
    NESTED_REASONING_ANSWER = "nested_reasoning_answer"  # <reasoning>explanation\n<answer>content</answer></reasoning>

    # LaTeX formats (text-friendly)
    LATEX_BOXED = "latex_boxed"  # \boxed{} - can contain text
    LATEX_TEXTBF = "latex_textbf"  # \textbf{} - bold text
    LATEX_TEXTIT = "latex_textit"  # \textit{} - italic text
    LATEX_UNDERLINE = "latex_underline"  # \underline{} - underlined text

    # LaTeX formats (math-only - for math datasets)
    LATEX_BOXED_MATH = "latex_boxed_math"  # $\boxed{}$ - math mode
    LATEX_ALIGN = "latex_align"  # \begin{align} \end{align} - math mode
    LATEX_EQUATION = "latex_equation"  # \begin{equation} \end{equation} - math mode
    LATEX_DISPLAYMATH = "latex_displaymath"  # \[ \] - math mode
    LATEX_INLINE_MATH = "latex_inline_math"  # $ $ - math mode
    LATEX_TEXT_MATH = "latex_text_math"  # $\text{answer}$ - text in math mode
    LATEX_MATHRM = "latex_mathrm"  # $\mathrm{answer}$ - roman text in math
    LATEX_THEREFORE = "latex_therefore"  # $\therefore answer$ - therefore symbol
    LATEX_IMPLIES = "latex_implies"  # $\implies answer$ - implies symbol
    LATEX_EQUIV = "latex_equiv"  # $answer \equiv value$ - equivalence
    LATEX_MATRIX = "latex_matrix"  # \begin{matrix} answer \end{matrix}
    LATEX_PMATRIX = "latex_pmatrix"  # \begin{pmatrix} answer \end{pmatrix}

    # Markdown formats
    MARKDOWN_CODE = "markdown_code"  # ```
    MARKDOWN_BOLD = "markdown_bold"  # **text**
    MARKDOWN_ITALIC = "markdown_italic"  # *text*
    MARKDOWN_HEADER = "markdown_header"  # ## Answer
    MARKDOWN_QUOTE = "markdown_quote"  # > answer

    # Bracket and delimiter formats
    SQUARE_BRACKETS = "square_brackets"  # [answer]
    DOUBLE_SQUARE_BRACKETS = "double_square_brackets"  # [[answer]]
    CURLY_BRACES = "curly_braces"  # {answer}
    PARENTHESES = "parentheses"  # (answer)
    ANGLE_BRACKETS = "angle_brackets"  # <answer>

    # Natural language patterns
    NATURAL_LANGUAGE_ANSWER = "natural_language_answer"  # "The answer is:"
    NATURAL_LANGUAGE_FINAL = "natural_language_final"  # "Final answer:"
    NATURAL_LANGUAGE_CONCLUSION = "natural_language_conclusion"  # "In conclusion:"
    NATURAL_LANGUAGE_THEREFORE = "natural_language_therefore"  # "Therefore:"
    NATURAL_LANGUAGE_RESULT = "natural_language_result"  # "The result is:"

    # Additional natural language patterns
    NATURAL_LANGUAGE_BEST = "natural_language_best"  # "The best answer is:"
    NATURAL_LANGUAGE_MY_FINAL = "natural_language_my_final"  # "My final answer is:"
    NATURAL_LANGUAGE_CORRECT = "natural_language_correct"  # "The correct answer is:"
    NATURAL_LANGUAGE_SOLUTION = "natural_language_solution"  # "The solution is:"
    NATURAL_LANGUAGE_RESPONSE = "natural_language_response"  # "My response is:"
    NATURAL_LANGUAGE_ULTIMATELY = "natural_language_ultimately"  # "Ultimately:"
    NATURAL_LANGUAGE_THUS = "natural_language_thus"  # "Thus:"
    NATURAL_LANGUAGE_HENCE = "natural_language_hence"  # "Hence:"
    NATURAL_LANGUAGE_CONSEQUENTLY = "natural_language_consequently"  # "Consequently:"
    NATURAL_LANGUAGE_TO_SUMMARIZE = "natural_language_to_summarize"  # "To summarize:"
    NATURAL_LANGUAGE_IN_SUMMARY = "natural_language_in_summary"  # "In summary:"
    NATURAL_LANGUAGE_OVERALL = "natural_language_overall"  # "Overall:"
    NATURAL_LANGUAGE_FINAL_VERDICT = (
        "natural_language_final_verdict"  # "Final verdict:"
    )
    NATURAL_LANGUAGE_BOTTOM_LINE = "natural_language_bottom_line"  # "Bottom line:"
    NATURAL_LANGUAGE_KEY_POINT = "natural_language_key_point"  # "The key point is:"

    # Special formats
    TEXTARENA_FORMAT = "textarena_format"  # [A], [B], [C], [D]
    COLON_FORMAT = "colon_format"  # Answer: content
    ARROW_FORMAT = "arrow_format"  # => answer or -> answer

    # HTML formats
    HTML_CODE = "html_code"  # <code></code>
    HTML_PRE = "html_pre"  # <pre></pre>
    HTML_SPAN = "html_span"  # <span></span>
    HTML_DIV = "html_div"  # <div></div>
    HTML_P = "html_p"  # <p></p>

    # Multiple structured tags
    MULTIPLE_TAGS = (
        "multiple_tags"  # <theory></theory><answer></answer><explanation></explanation>
    )

    # Complex multi-tag formats for specific domains
    COMPLEX_CODING_FORMAT = "complex_coding_format"
    COMPLEX_CODING_SIMPLE = "complex_coding_simple"
    COMPLEX_CODING_MINIMAL = "complex_coding_minimal"
    COMPLEX_MATH_FORMAT = "complex_math_format"
    COMPLEX_MATH_SIMPLE = "complex_math_simple"
    COMPLEX_GENERAL_FORMAT = "complex_general_format"
    COMPLEX_GENERAL_SIMPLE = "complex_general_simple"
    COMPLEX_RESEARCH_FORMAT = "complex_research_format"

    # Advanced scratchpad and classification formats
    COMPLEX_SCRATCHPAD_FULL = "complex_scratchpad_full"
    COMPLEX_SCRATCHPAD_SIMPLE = "complex_scratchpad_simple"
    COMPLEX_CLASSIFICATION_FORMAT = "complex_classification_format"
    COMPLEX_ANALYSIS_WITH_ANSWER = "complex_analysis_with_answer"
    COMPLEX_EVALUATION_FORMAT = "complex_evaluation_format"

    # Dynamic compositor formats - randomly combine components in different output formats
    # COMMENTED OUT: These formats are proving too difficult to parse reliably
    # DYNAMIC_SCRATCHPAD_XML = "dynamic_scratchpad_xml"  # Random XML scratchpad components
    # DYNAMIC_SCRATCHPAD_JSON = "dynamic_scratchpad_json"  # Random JSON scratchpad components
    # DYNAMIC_SCRATCHPAD_YAML = "dynamic_scratchpad_yaml"  # Random YAML scratchpad components
    # DYNAMIC_SCRATCHPAD_TOML = "dynamic_scratchpad_toml"  # Random TOML scratchpad components
    # DYNAMIC_ANALYSIS_XML = "dynamic_analysis_xml"  # Random XML analysis components
    # DYNAMIC_ANALYSIS_JSON = "dynamic_analysis_json"  # Random JSON analysis components
    # DYNAMIC_WORKFLOW_XML = "dynamic_workflow_xml"  # Random XML workflow components
    # DYNAMIC_WORKFLOW_JSON = "dynamic_workflow_json"  # Random JSON workflow components

    # Custom delimiters
    PIPE_DELIMITED = "pipe_delimited"  # |answer|
    HASH_DELIMITED = "hash_delimited"  # #answer#
    UNDERSCORE_DELIMITED = "underscore_delimited"  # _answer_
    TILDE_DELIMITED = "tilde_delimited"  # ~answer~

    # Programming-style formats
    FUNCTION_CALL = "function_call"  # answer()
    VARIABLE_ASSIGNMENT = "variable_assignment"  # answer = "content"
    RETURN_STATEMENT = "return_statement"  # return "answer"

    # Additional easy-to-parse formats
    EQUALS_FORMAT = "equals_format"  # = answer
    DASH_FORMAT = "dash_format"  # - answer
    PLUS_FORMAT = "plus_format"  # + answer
    STAR_FORMAT = "star_format"  # * answer
    PERCENT_FORMAT = "percent_format"  # % answer
    AMPERSAND_FORMAT = "ampersand_format"  # & answer
    AT_FORMAT = "at_format"  # @ answer
    EXCLAMATION_FORMAT = "exclamation_format"  # ! answer
    QUESTION_FORMAT = "question_format"  # ? answer (for when answer is a question)
    SEMICOLON_FORMAT = "semicolon_format"  # ; answer
    DOUBLE_COLON_FORMAT = "double_colon_format"  # :: answer
    TRIPLE_DASH_FORMAT = "triple_dash_format"  # --- answer
    DOUBLE_ARROW_FORMAT = "double_arrow_format"  # >> answer
    TRIPLE_ARROW_FORMAT = "triple_arrow_format"  # >>> answer
    BACKTICK_FORMAT = "backtick_format"  # `answer`
    DOUBLE_BACKTICK_FORMAT = "double_backtick_format"  # ``answer``
    QUOTE_FORMAT = "quote_format"  # "answer"
    SINGLE_QUOTE_FORMAT = "single_quote_format"  # 'answer'
    TRIPLE_QUOTE_FORMAT = "triple_quote_format"  # """answer"""
    ANSWER_IS_FORMAT = "answer_is_format"  # ANSWER IS: content
    SOLUTION_IS_FORMAT = "solution_is_format"  # SOLUTION IS: content
    RESULT_IS_FORMAT = "result_is_format"  # RESULT IS: content
    OUTPUT_IS_FORMAT = "output_is_format"  # OUTPUT IS: content

    # Code-specific formats (for code datasets)
    PYTHON_PRINT = "python_print"  # print("answer")
    JAVASCRIPT_CONSOLE = "javascript_console"  # console.log("answer")
    PYTHON_COMMENT = "python_comment"  # # answer
    JAVASCRIPT_COMMENT = "javascript_comment"  # // answer
    C_COMMENT = "c_comment"  # /* answer */
    SHELL_ECHO = "shell_echo"  # echo "answer"
    SHELL_OUTPUT = "shell_output"  # $ answer
    PYTHON_DOCSTRING = "python_docstring"  # """answer"""
    INI_FORMAT = "ini_format"  # [section]\nanswer=value
    ENV_FORMAT = "env_format"  # ANSWER=value


class AnswerFormatEnvConfig(BaseEnvConfig):
    """Custom config class for AnswerFormatEnv with additional parameters."""

    dataset_configs: List[Dict[str, Any]] = Field(
        default=[
            {
                "name": "teknium/OpenHermes-2.5",
                "split": "train",
                "sample_size": 1000,
                "prompt_field": "conversations",
                "answer_field": "conversations",
                "metadata_fields": ["source"],
                "dataset_type": "generic",  # Options: "generic", "math_only", "code_only"
            },
            {
                "name": "gsm8k",
                "split": "train",
                "sample_size": 2000,
                "prompt_field": "question",
                "answer_field": "answer",
                "metadata_fields": [],
                "dataset_type": "math_only",  # GSM8K is a math dataset
            },
        ],
        description="List of dataset configurations to load and combine. Each can specify dataset_type: 'generic', 'math_only', or 'code_only'",  # noqa: E501
    )

    debug_logging: bool = Field(
        default=True, description="Enable detailed debug logging"
    )

    suppress_base_env_logs: bool = Field(
        default=True,
        description="Suppress verbose base environment logs (like status dict updates)",
    )

    dump_rollouts: bool = Field(
        default=False,
        description="Whether to dump rollouts to JSONL files for analysis",
    )

    dump_failed_rollouts: bool = Field(
        default=False,
        description="Whether to dump failed rollouts (all 0 scores) to JSONL files for debugging",
    )

    rollout_save_score_threshold: float = Field(
        default=0.0,
        description="Minimum score threshold for saving rollouts (0.0 saves all)",
        ge=0.0,
        le=1.0,
    )

    eval_set_percentage: float = Field(
        default=0.1,
        description="Percentage of the dataset to use for evaluation (e.g., 0.1 for 10%)",
        ge=0.0,
        le=1.0,
    )

    supported_formats: Optional[List[AnswerFormat]] = Field(
        default=None,
        description="Optional list of AnswerFormat enums to use. If None, all formats are used.",
    )

    ensure_equivalent_ratios: bool = Field(
        default=False,
        description="Ensure equivalent ratios of successful groups across all formats by pausing formats that reach the threshold",  # noqa: E501
    )

    format_group_threshold: int = Field(
        default=50,
        description="Number of successful groups per format before pausing that format (only used when ensure_equivalent_ratios=True)",  # noqa: E501
        ge=1,
        le=1000,
    )

    seed: int = Field(
        default_factory=lambda: random.randint(20, 9999),
        description="Random seed used for dataset shuffling and other random operations",
        ge=1,
        le=99999,
    )


class AnswerFormatEnv(BaseEnv):
    """Environment for training models on answer format adherence."""

    env_config_cls = AnswerFormatEnvConfig

    def __init__(
        self,
        config: AnswerFormatEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)

        # Set up debug logging
        self.debug_logging = getattr(config, "debug_logging", True)
        if self.debug_logging:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.setLevel(logging.DEBUG)
            # Prevent propagation to avoid duplicate messages
            self.logger.propagate = False
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
            self.logger.info("Debug logging enabled for AnswerFormatEnv")
        else:
            self.logger = logging.getLogger(f"{self.__class__.__name__}")
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False

        # Suppress base environment logs if requested
        if getattr(config, "suppress_base_env_logs", True):
            # Set the base environment logger to WARNING level to suppress INFO logs
            base_logger = logging.getLogger("atroposlib.envs.base")
            base_logger.setLevel(logging.WARNING)

        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.rollouts_for_wandb = []
        self.dataset_items: List[Dict[str, Any]] = []

        # Track format success rates
        self.format_success_counts = {}
        self.format_total_counts = {}

        # Track successful groups per format for equivalent ratio enforcement
        self.format_successful_groups = {}  # format_name -> count of successful groups
        self.format_group_threshold = config.format_group_threshold
        self.ensure_equivalent_ratios = config.ensure_equivalent_ratios

        # For saving failed rollouts (all 0 scores) for debugging
        self.failed_rollouts_to_save_buffer: List[Dict[str, Any]] = []
        self.failed_save_file_batch_num = 0

        # Group-level statistics tracking
        self.group_statistics = {
            "total_groups": 0,
            "successful_groups": 0,
            "failed_groups": 0,
            "average_scores": [],
            "format_distribution": {},
        }

        # Define format categories by dataset type
        self.math_only_formats = {
            AnswerFormat.LATEX_BOXED_MATH,
            AnswerFormat.LATEX_ALIGN,
            AnswerFormat.LATEX_EQUATION,
            AnswerFormat.LATEX_DISPLAYMATH,
            AnswerFormat.LATEX_INLINE_MATH,
            AnswerFormat.LATEX_TEXT_MATH,
            AnswerFormat.LATEX_MATHRM,
            AnswerFormat.LATEX_THEREFORE,
            AnswerFormat.LATEX_IMPLIES,
            AnswerFormat.LATEX_EQUIV,
            AnswerFormat.LATEX_MATRIX,
            AnswerFormat.LATEX_PMATRIX,
            AnswerFormat.COMPLEX_MATH_FORMAT,
            AnswerFormat.COMPLEX_MATH_SIMPLE,
        }

        self.code_only_formats = {
            AnswerFormat.SHELL_OUTPUT,
            AnswerFormat.COMPLEX_CODING_FORMAT,
            AnswerFormat.COMPLEX_CODING_SIMPLE,
            AnswerFormat.COMPLEX_CODING_MINIMAL,
        }

        # Generic formats work with any content type
        all_formats = set(AnswerFormat)
        self.generic_formats = (
            all_formats - self.math_only_formats - self.code_only_formats
        )

        # Validate configuration
        self._validate_config(config)

        # Store seed for dataset shuffling and other random operations
        self.seed = config.seed

        # Store base supported formats (will be filtered per item based on dataset type)
        if config.supported_formats and len(config.supported_formats) > 0:
            self.base_supported_formats = config.supported_formats
            if self.debug_logging:
                self.logger.info(
                    f"Using configured base formats: {[f.value for f in self.base_supported_formats]}"
                )
        else:
            self.base_supported_formats = list(AnswerFormat)
            if self.debug_logging:
                self.logger.info(
                    f"Using all formats as base: {len(self.base_supported_formats)} formats"
                )

        # Data dumping setup
        self.run_uuid = str(uuid.uuid4())
        self.rollouts_to_save_buffer: List[Dict[str, Any]] = []
        self.processed_item_count = 0

        # Dynamic format component storage
        # Dynamic formats are commented out - no storage needed

        # Create datadumps directory
        self.datadumps_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "datadumps"
        )
        self.save_file_batch_num = 0

        if self.debug_logging:
            self.logger.info(
                f"Data dumping {'enabled' if config.dump_rollouts else 'disabled'}"
            )
            if config.dump_rollouts:
                self.logger.info(f"Rollouts will be saved to: {self.datadumps_dir}")

    def _validate_config(self, config: AnswerFormatEnvConfig):
        """Validate the configuration for common issues."""
        # Check dataset configurations
        if not config.dataset_configs:
            raise ValueError("At least one dataset configuration is required")

        for i, dataset_config in enumerate(config.dataset_configs):
            if "name" not in dataset_config:
                raise ValueError(f"Dataset config {i} missing 'name' field")

            dataset_type = dataset_config.get("dataset_type", "generic")
            if dataset_type not in ["generic", "math_only", "code_only"]:
                self.logger.warning(
                    f"Unknown dataset_type '{dataset_type}' in config {i}, using 'generic'"
                )
                dataset_config["dataset_type"] = "generic"

        # Check eval percentage
        if not (0.0 <= config.eval_set_percentage <= 1.0):
            raise ValueError(
                f"eval_set_percentage must be between 0.0 and 1.0, got {config.eval_set_percentage}"
            )

        # Check supported formats
        if config.supported_formats:
            invalid_formats = [
                f for f in config.supported_formats if not isinstance(f, AnswerFormat)
            ]
            if invalid_formats:
                raise ValueError(
                    f"Invalid formats in supported_formats: {invalid_formats}"
                )

        # Check equivalent ratio parameters
        if config.format_group_threshold < 1:
            raise ValueError(
                f"format_group_threshold must be at least 1, got {config.format_group_threshold}"
            )

        if self.debug_logging:
            self.logger.info("Configuration validation passed")
            if config.ensure_equivalent_ratios:
                self.logger.info(
                    f"Equivalent ratio enforcement enabled with threshold of {config.format_group_threshold} successful groups per format"  # noqa: E501
                )

    def _should_exclude_format_for_balance(self, answer_format: AnswerFormat) -> bool:
        """Check if a format should be excluded due to equivalent ratio enforcement."""
        if not self.ensure_equivalent_ratios:
            return False

        format_name = answer_format.value
        successful_groups = self.format_successful_groups.get(format_name, 0)

        # If this format has reached the threshold, check if others are still below
        if successful_groups >= self.format_group_threshold:
            # Check if there are any formats still below the threshold
            min_successful_groups = (
                min(self.format_successful_groups.values())
                if self.format_successful_groups
                else 0
            )
            if min_successful_groups < self.format_group_threshold:
                if self.debug_logging:
                    self.logger.debug(
                        f"Excluding format {format_name} (has {successful_groups}, min is {min_successful_groups})"
                    )
                return True

        return False

    def _get_balanced_formats(
        self, available_formats: List[AnswerFormat]
    ) -> List[AnswerFormat]:
        """Filter formats based on equivalent ratio requirements."""
        if not self.ensure_equivalent_ratios:
            return available_formats

        # Filter out formats that have reached the threshold while others haven't
        balanced_formats = [
            f
            for f in available_formats
            if not self._should_exclude_format_for_balance(f)
        ]

        # If all formats are excluded (shouldn't happen in practice), allow all
        if not balanced_formats:
            if self.debug_logging:
                self.logger.warning(
                    "All formats excluded by equivalent ratio enforcement, allowing all formats"
                )
            return available_formats

        if self.debug_logging and len(balanced_formats) != len(available_formats):
            excluded_count = len(available_formats) - len(balanced_formats)
            self.logger.debug(
                f"Equivalent ratio enforcement: excluded {excluded_count} formats that reached threshold"
            )

        return balanced_formats

    def get_equivalent_ratio_status(self) -> Dict[str, Any]:
        """Get current status of equivalent ratio enforcement for debugging/monitoring."""
        if not self.ensure_equivalent_ratios:
            return {"enabled": False}

        total_formats = len(self.base_supported_formats)
        formats_at_threshold = sum(
            1
            for count in self.format_successful_groups.values()
            if count >= self.format_group_threshold
        )

        status = {
            "enabled": True,
            "threshold": self.format_group_threshold,
            "total_formats": total_formats,
            "formats_with_data": len(self.format_successful_groups),
            "formats_at_threshold": formats_at_threshold,
            "completion_percentage": (
                (formats_at_threshold / total_formats) * 100 if total_formats > 0 else 0
            ),
        }

        if self.format_successful_groups:
            successful_counts = list(self.format_successful_groups.values())
            status.update(
                {
                    "min_successful_groups": min(successful_counts),
                    "max_successful_groups": max(successful_counts),
                    "avg_successful_groups": sum(successful_counts)
                    / len(successful_counts),
                }
            )

            # Top 5 formats by successful groups
            sorted_formats = sorted(
                self.format_successful_groups.items(), key=lambda x: x[1], reverse=True
            )
            status["top_formats"] = sorted_formats[:5]

            # Bottom 5 formats by successful groups (that have any data)
            status["bottom_formats"] = (
                sorted_formats[-5:] if len(sorted_formats) >= 5 else sorted_formats
            )

        return status

    def _generate_system_prompt(self, answer_format: AnswerFormat) -> str:
        """Generate a system prompt for the specified answer format."""
        base_prompt = (
            "You are an AI assistant that provides helpful responses. "
            "You may use extremely long chains of thought to deeply consider the "
            "problem and deliberate with yourself via systematic reasoning processes "
            "to help come to a correct solution prior to answering. "
            "You should enclose your thoughts and internal monologue inside <think> </think> tags. "
            "CRITICAL FORMAT REQUIREMENT: After your thinking, you must provide your answer in the EXACT format specified below. "  # noqa: E501
            "Use the specified format EXACTLY ONCE and ONLY ONCE. Do not use the format multiple times or in any other way."  # noqa: E501
        )

        format_instructions = {
            # Basic structured data formats (answer only)
            AnswerFormat.JSON: (
                "CRITICAL: After your thinking, provide your answer as a JSON object with an 'answer' field EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                'Example: {"answer": "your response here"}'
            ),
            AnswerFormat.JSON_ARRAY: (
                "CRITICAL: After your thinking, provide your answer as a JSON array with one element EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                'Example: ["your response here"]'
            ),
            AnswerFormat.JSON_SIMPLE: (
                "CRITICAL: After your thinking, provide your answer as a simple JSON string EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                'Example: "your response here"'
            ),
            AnswerFormat.YAML: (
                "CRITICAL: After your thinking, provide your answer in YAML format with an 'answer' field EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                "Example:\nanswer: your response here"
            ),
            AnswerFormat.YAML_LIST: (
                "CRITICAL: After your thinking, provide your answer as a YAML list with one item EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                "Example:\n- your response here"
            ),
            AnswerFormat.TOML: (
                "CRITICAL: After your thinking, provide your answer in TOML format with an 'answer' field EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                'Example:\nanswer = "your response here"'
            ),
            AnswerFormat.TOML_SECTION: (
                "CRITICAL: After your thinking, provide your answer in TOML format with a section EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                'Example:\n[response]\nanswer = "your response here"'
            ),
            # Structured data with confidence scores
            AnswerFormat.JSON_CONFIDENCE: (
                "CRITICAL: After your thinking, provide your answer as JSON with answer and confidence fields EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                'Example: {"answer": "your response here", "confidence": 0.9}'
            ),
            AnswerFormat.YAML_CONFIDENCE: (
                "CRITICAL: After your thinking, provide your answer in YAML with answer and confidence fields EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example:\nanswer: your response here\nconfidence: 0.9"
            ),
            AnswerFormat.TOML_CONFIDENCE: (
                "CRITICAL: After your thinking, provide your answer in TOML with answer and confidence fields EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example:\nanswer = "your response here"\nconfidence = 0.9'
            ),
            # XML/HTML tag variations (XML now uses answer tags)
            AnswerFormat.XML: (
                "CRITICAL: After your thinking, provide your answer enclosed in <answer></answer> tags EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: <answer>your response here</answer>"
            ),
            AnswerFormat.XML_FINAL_ANSWER: (
                "CRITICAL: After your thinking, provide your answer enclosed in <answer></answer> tags with 'Final Answer:' prefix EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "
                "Example: <answer>Final Answer: your response here</answer>"
            ),
            AnswerFormat.OUTPUT_TAGS: (
                "CRITICAL: After your thinking, provide your answer enclosed in <output></output> tags. "  # noqa: E501
                "Example: <output>your response here</output>"
            ),
            AnswerFormat.RESULT_TAGS: (
                "CRITICAL: After your thinking, provide your answer enclosed in <result></result> tags. "  # noqa: E501
                "Example: <result>your response here</result>"
            ),
            AnswerFormat.RESPONSE_TAGS: (
                "CRITICAL: After your thinking, provide your answer enclosed in <response></response> tags. "  # noqa: E501
                "Example: <response>your response here</response>"
            ),
            AnswerFormat.FINAL_ANSWER_TAGS: (
                "CRITICAL: After your thinking, provide your answer enclosed in <final_answer></final_answer> tags. "  # noqa: E501
                "Example: <final_answer>your response here</final_answer>"
            ),
            AnswerFormat.SOLUTION_TAGS: (
                "CRITICAL: After your thinking, provide your answer enclosed in <solution></solution> tags. "  # noqa: E501
                "Example: <solution>your response here</solution>"
            ),
            AnswerFormat.CONCLUSION_TAGS: (
                "CRITICAL: After your thinking, provide your answer enclosed in <conclusion></conclusion> tags. "  # noqa: E501
                "Example: <conclusion>your response here</conclusion>"
            ),
            AnswerFormat.REPLY_TAGS: (
                "CRITICAL: After your thinking, provide your answer enclosed in <reply></reply> tags. "  # noqa: E501
                "Example: <reply>your response here</reply>"
            ),  # noqa: E501
            AnswerFormat.NESTED_RESPONSE_ANSWER: (
                "CRITICAL: After your thinking, provide explanation and answer in nested response tags EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: <response>Brief explanation of your reasoning\n<answer>your final answer here</answer></response>"  # noqa: E501
            ),
            AnswerFormat.NESTED_SOLUTION_ANSWER: (
                "CRITICAL: After your thinking, provide explanation and answer in nested solution tags EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: <solution>Brief explanation of the approach\n<answer>your final answer here</answer></solution>"  # noqa: E501
            ),
            AnswerFormat.NESTED_OUTPUT_RESULT: (
                "CRITICAL: After your thinking, provide explanation and result in nested output tags EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: <output>Brief explanation of the process\n<result>your final result here</result></output>"  # noqa: E501
            ),
            AnswerFormat.NESTED_ANALYSIS_CONCLUSION: (
                "CRITICAL: After your thinking, provide analysis and conclusion in nested tags EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: <analysis>Brief analysis of the problem\n<conclusion>your final conclusion here</conclusion></analysis>"  # noqa: E501
            ),
            AnswerFormat.NESTED_REASONING_ANSWER: (
                "CRITICAL: After your thinking, provide reasoning and answer in nested tags EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: <reasoning>Brief reasoning summary\n<answer>your final answer here</answer></reasoning>"  # noqa: E501
            ),
            # LaTeX formats (text-friendly)
            AnswerFormat.LATEX_BOXED: (
                "CRITICAL: After your thinking, provide your answer using LaTeX boxed notation EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: \\boxed{your answer here}"
            ),
            AnswerFormat.LATEX_TEXTBF: (
                "CRITICAL: After your thinking, provide your answer using LaTeX bold text notation EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: \\textbf{your answer here}"
            ),
            AnswerFormat.LATEX_TEXTIT: (
                "CRITICAL: After your thinking, provide your answer using LaTeX italic text notation EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: \\textit{your answer here}"
            ),
            AnswerFormat.LATEX_UNDERLINE: (
                "CRITICAL: After your thinking, provide your answer using LaTeX underline notation EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: \\underline{your answer here}"
            ),
            # LaTeX formats (math-only - for mathematical content)
            AnswerFormat.LATEX_BOXED_MATH: (
                "CRITICAL: After your thinking, provide your answer using LaTeX math boxed notation EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for mathematical expressions. "  # noqa: E501
                "Example: $\\boxed{x = 42}$"
            ),
            AnswerFormat.LATEX_ALIGN: (
                "CRITICAL: After your thinking, provide your answer within LaTeX align blocks EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for mathematical expressions. "  # noqa: E501
                "Example:\n\\begin{align}\nx &= 42 \\\\\ny &= x + 1\n\\end{align}"
            ),
            AnswerFormat.LATEX_EQUATION: (
                "CRITICAL: After your thinking, provide your answer within LaTeX equation blocks EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for mathematical expressions. "  # noqa: E501
                "Example:\n\\begin{equation}\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n\\end{equation}"  # noqa: E501
            ),
            AnswerFormat.LATEX_DISPLAYMATH: (
                "CRITICAL: After your thinking, provide your answer within LaTeX display math delimiters EXACTLY ONCE. "  # noqa: E501
                "Example: \\[x = \\frac{a + b}{c}\\]"
            ),
            AnswerFormat.LATEX_INLINE_MATH: (
                "CRITICAL: After your thinking, provide your answer within LaTeX inline math delimiters EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for mathematical expressions. "  # noqa: E501
                "Example: $x = 42$"
            ),
            AnswerFormat.LATEX_TEXT_MATH: (
                "CRITICAL: After your thinking, provide your answer using LaTeX text within math mode EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for text answers in mathematical context. "  # noqa: E501
                "Example: $\\text{your answer here}$"
            ),
            AnswerFormat.LATEX_MATHRM: (
                "CRITICAL: After your thinking, provide your answer using LaTeX mathrm within math mode EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for roman text in mathematical context. "  # noqa: E501
                "Example: $\\mathrm{your answer here}$"
            ),
            AnswerFormat.LATEX_THEREFORE: (
                "CRITICAL: After your thinking, provide your answer using LaTeX therefore symbol EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for mathematical conclusions. "  # noqa: E501
                "Example: $\\therefore \\text{your answer here}$"
            ),
            AnswerFormat.LATEX_IMPLIES: (
                "CRITICAL: After your thinking, provide your answer using LaTeX implies symbol EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for mathematical implications. "  # noqa: E501
                "Example: $\\implies \\text{your answer here}$"
            ),
            AnswerFormat.LATEX_EQUIV: (
                "CRITICAL: After your thinking, provide your answer using LaTeX equivalence symbol EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for mathematical equivalences. "  # noqa: E501
                "Example: $\\text{answer} \\equiv \\text{your solution here}$"
            ),
            AnswerFormat.LATEX_MATRIX: (
                "CRITICAL: After your thinking, provide your answer within LaTeX matrix environment EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for matrix/array answers. "  # noqa: E501
                "Example: $\\begin{matrix} \\text{your answer here} \\end{matrix}$"
            ),
            AnswerFormat.LATEX_PMATRIX: (
                "CRITICAL: After your thinking, provide your answer within LaTeX pmatrix environment EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for parenthesized matrix answers. "  # noqa: E501
                "Example: $\\begin{pmatrix} \\text{your answer here} \\end{pmatrix}$"
            ),
            # Markdown formats
            AnswerFormat.MARKDOWN_CODE: (
                "CRITICAL: After your thinking, provide your answer in a markdown code block without language specification. "  # noqa: E501
                "Example:\n```\nyour answer here\n```"
            ),
            AnswerFormat.MARKDOWN_BOLD: (
                "CRITICAL: After your thinking, provide your answer in bold markdown formatting. "
                "Example: **your answer here**"
            ),
            AnswerFormat.MARKDOWN_ITALIC: (
                "CRITICAL: After your thinking, provide your answer in italic markdown formatting. "
                "Example: *your answer here*"
            ),
            AnswerFormat.MARKDOWN_HEADER: (
                "CRITICAL: After your thinking, provide your answer as a markdown header. "
                "Example: ## your answer here"
            ),
            AnswerFormat.MARKDOWN_QUOTE: (
                "CRITICAL: After your thinking, provide your answer as a markdown quote. "
                "Example: > your answer here"
            ),
            # Bracket and delimiter formats
            AnswerFormat.SQUARE_BRACKETS: (
                "CRITICAL: After your thinking, provide your answer enclosed in square brackets. "
                "Example: [your answer here]"
            ),
            AnswerFormat.DOUBLE_SQUARE_BRACKETS: (
                "CRITICAL: After your thinking, provide your answer enclosed in double square brackets. "
                "Example: [[your answer here]]"
            ),
            AnswerFormat.CURLY_BRACES: (
                "CRITICAL: After your thinking, provide your answer enclosed in curly braces. "
                "Example: {your answer here}"
            ),
            AnswerFormat.PARENTHESES: (
                "CRITICAL: After your thinking, provide your answer enclosed in parentheses. "
                "Example: (your answer here)"
            ),
            AnswerFormat.ANGLE_BRACKETS: (
                "CRITICAL: After your thinking, provide your answer enclosed in angle brackets. "
                "Example: <your answer here>"
            ),
            # Natural language patterns
            AnswerFormat.NATURAL_LANGUAGE_ANSWER: (
                "CRITICAL: After your thinking, provide your answer using 'The answer is:' pattern. "
                "Example: The answer is: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_FINAL: (
                "CRITICAL: After your thinking, provide your answer using 'Final answer:' pattern. "
                "Example: Final answer: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_CONCLUSION: (
                "CRITICAL: After your thinking, provide your answer using 'In conclusion:' pattern. "
                "Example: In conclusion: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_THEREFORE: (
                "CRITICAL: After your thinking, provide your answer using 'Therefore:' pattern. "
                "Example: Therefore: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_RESULT: (
                "CRITICAL: After your thinking, provide your answer using 'The result is:' pattern. "
                "Example: The result is: your response here"
            ),
            # Additional natural language patterns
            AnswerFormat.NATURAL_LANGUAGE_BEST: (
                "CRITICAL: After your thinking, provide your answer using 'The best answer is:' pattern. "
                "Example: The best answer is: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_MY_FINAL: (
                "CRITICAL: After your thinking, provide your answer using 'My final answer is:' pattern. "
                "Example: My final answer is: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_CORRECT: (
                "CRITICAL: After your thinking, provide your answer using 'The correct answer is:' pattern. "
                "Example: The correct answer is: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_SOLUTION: (
                "CRITICAL: After your thinking, provide your answer using 'The solution is:' pattern. "
                "Example: The solution is: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_RESPONSE: (
                "CRITICAL: After your thinking, provide your answer using 'My response is:' pattern. "
                "Example: My response is: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_ULTIMATELY: (
                "CRITICAL: After your thinking, provide your answer using 'Ultimately:' pattern. "
                "Example: Ultimately: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_THUS: (
                "CRITICAL: After your thinking, provide your answer using 'Thus:' pattern. "
                "Example: Thus: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_HENCE: (
                "CRITICAL: After your thinking, provide your answer using 'Hence:' pattern. "
                "Example: Hence: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_CONSEQUENTLY: (
                "CRITICAL: After your thinking, provide your answer using 'Consequently:' pattern. "
                "Example: Consequently: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_TO_SUMMARIZE: (
                "CRITICAL: After your thinking, provide your answer using 'To summarize:' pattern. "
                "Example: To summarize: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_IN_SUMMARY: (
                "CRITICAL: After your thinking, provide your answer using 'In summary:' pattern. "
                "Example: In summary: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_OVERALL: (
                "CRITICAL: After your thinking, provide your answer using 'Overall:' pattern. "
                "Example: Overall: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_FINAL_VERDICT: (
                "CRITICAL: After your thinking, provide your answer using 'Final verdict:' pattern. "
                "Example: Final verdict: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_BOTTOM_LINE: (
                "CRITICAL: After your thinking, provide your answer using 'Bottom line:' pattern. "
                "Example: Bottom line: your response here"
            ),
            AnswerFormat.NATURAL_LANGUAGE_KEY_POINT: (
                "CRITICAL: After your thinking, provide your answer using 'The key point is:' pattern. "  # noqa: E501
                "Example: The key point is: your response here"  # noqa: E501
            ),
            # Special formats
            AnswerFormat.TEXTARENA_FORMAT: (
                "CRITICAL: After your thinking, provide your answer in TextArena format using square brackets with a letter. "  # noqa: E501
                "Example: [A] or [B] or [C] or [D]"
            ),
            AnswerFormat.COLON_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with 'Answer:' prefix. "
                "Example: Answer: your response here"
            ),
            AnswerFormat.ARROW_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with arrow notation. "
                "Example: => your answer here"
            ),
            # HTML formats
            AnswerFormat.HTML_CODE: (
                "CRITICAL: After your thinking, provide your answer within HTML code tags. "
                "Example: <code>your answer here</code>"
            ),
            AnswerFormat.HTML_PRE: (
                "CRITICAL: After your thinking, provide your answer within HTML pre tags. "
                "Example: <pre>your answer here</pre>"
            ),
            AnswerFormat.HTML_SPAN: (
                "CRITICAL: After your thinking, provide your answer within HTML span tags. "
                "Example: <span>your answer here</span>"
            ),
            AnswerFormat.HTML_DIV: (
                "CRITICAL: After your thinking, provide your answer within HTML div tags. "
                "Example: <div>your answer here</div>"
            ),
            AnswerFormat.HTML_P: (
                "CRITICAL: After your thinking, provide your answer within HTML p tags. "
                "Example: <p>your answer here</p>"
            ),
            # Multiple structured tags
            AnswerFormat.MULTIPLE_TAGS: (
                "CRITICAL: After your thinking, provide your response using multiple structured tags. "
                "Example:\n<theory>your reasoning</theory>\n<answer>your final answer</answer>\n<explanation>additional context</explanation>"  # noqa: E501
            ),
            # Complex multi-tag formats for specific domains
            AnswerFormat.COMPLEX_CODING_FORMAT: (
                "CRITICAL: After your thinking, provide your response in the following structured format for coding problems. "  # noqa: E501
                "Note: The <REASONING> and other sections should summarize your previous thinking, not repeat the detailed thought process. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<RESTATEMENT>Restate the problem clearly</RESTATEMENT>\n"  # noqa: E501
                "<REASONING>\n<THOUGHT_1>First key insight</THOUGHT_1>\n<THOUGHT_2>Second key insight</THOUGHT_2>\n</REASONING>\n"  # noqa: E501
                "<PLAN>\n<STEP_1>First step</STEP_1>\n<STEP_2>Second step</STEP_2>\n</PLAN>\n"  # noqa: E501
                "<PYDANTIC_SCHEMAS>\n<SCHEMA_1>First schema</SCHEMA_1>\n<SCHEMA_2>Second schema</SCHEMA_2>\n</PYDANTIC_SCHEMAS>\n"  # noqa: E501
                "<DIAGRAM>UML workflow diagram in natural language</DIAGRAM>\n"  # noqa: E501
                "<REFLECTION>Internal critique and validation</REFLECTION>\n"  # noqa: E501
                "<SOLUTION>Your code solution</SOLUTION>\n"  # noqa: E501
                "<EXPLANATION>Explanation of the code</EXPLANATION>\n"  # noqa: E501
                "<UNIT_TEST>Unit test code</UNIT_TEST>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_CODING_SIMPLE: (
                "CRITICAL: After your thinking, provide your response in the following structured format for coding problems. "  # noqa: E501
                "Note: The sections should summarize your previous thinking, not repeat detailed thought processes. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<RESTATEMENT>Restate the problem</RESTATEMENT>\n"  # noqa: E501
                "<REASONING>\n<THOUGHT_1>Key insight 1</THOUGHT_1>\n<THOUGHT_2>Key insight 2</THOUGHT_2>\n</REASONING>\n"  # noqa: E501
                "<PLAN>\n<STEP_1>Step 1</STEP_1>\n<STEP_2>Step 2</STEP_2>\n</PLAN>\n"  # noqa: E501
                "<SOLUTION>Your code solution</SOLUTION>\n"  # noqa: E501
                "<EXPLANATION>Code explanation</EXPLANATION>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_CODING_MINIMAL: (
                "CRITICAL: After your thinking, provide your response in the following structured format for coding problems. "  # noqa: E501
                "Note: The sections should summarize your previous thinking. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<ANALYSIS>Problem analysis summary</ANALYSIS>\n"  # noqa: E501
                "<APPROACH>Solution approach</APPROACH>\n"  # noqa: E501
                "<SOLUTION>Your code solution</SOLUTION>\n"  # noqa: E501
                "<TEST>Test cases or validation</TEST>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_MATH_FORMAT: (
                "CRITICAL: After your thinking, provide your response in the following structured format for math problems. "  # noqa: E501
                "Note: The sections should summarize your previous thinking, not repeat detailed calculations. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<RESTATEMENT>Restate the mathematical problem</RESTATEMENT>\n"  # noqa: E501
                "<REASONING>\n<INSIGHT_1>First mathematical insight</INSIGHT_1>\n<INSIGHT_2>Second insight</INSIGHT_2>\n</REASONING>\n"  # noqa: E501
                "<APPROACH>Mathematical approach and strategy</APPROACH>\n"  # noqa: E501
                "<DERIVATION>Step-by-step mathematical derivation</DERIVATION>\n"  # noqa: E501
                "<SOLUTION>Final mathematical solution</SOLUTION>\n"  # noqa: E501
                "<VERIFICATION>Verification of the solution</VERIFICATION>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_MATH_SIMPLE: (
                "CRITICAL: After your thinking, provide your response in the following structured format for math problems. "  # noqa: E501
                "Note: The sections should summarize your previous thinking. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<PROBLEM_ANALYSIS>Analysis of the mathematical problem</PROBLEM_ANALYSIS>\n"  # noqa: E501
                "<SOLUTION_STEPS>Step-by-step solution process</SOLUTION_STEPS>\n"  # noqa: E501
                "<FINAL_ANSWER>The final mathematical answer</FINAL_ANSWER>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_GENERAL_FORMAT: (
                "CRITICAL: After your thinking, provide your response in the following structured format for general problems. "  # noqa: E501
                "Note: The sections should summarize your previous thinking, not repeat detailed analysis. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<RESTATEMENT>Restate the problem or question</RESTATEMENT>\n"  # noqa: E501
                "<ANALYSIS>\n<ASPECT_1>First key aspect</ASPECT_1>\n<ASPECT_2>Second key aspect</ASPECT_2>\n</ANALYSIS>\n"  # noqa: E501
                "<REASONING>Logical reasoning process</REASONING>\n"  # noqa: E501
                "<CONCLUSION>Your conclusion or answer</CONCLUSION>\n"  # noqa: E501
                "<REFLECTION>Reflection on the solution quality</REFLECTION>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_GENERAL_SIMPLE: (
                "CRITICAL: After your thinking, provide your response in the following structured format. "  # noqa: E501
                "Note: The sections should summarize your previous thinking. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<ANALYSIS>Problem or question analysis</ANALYSIS>\n"  # noqa: E501
                "<REASONING>Key reasoning points</REASONING>\n"  # noqa: E501
                "<CONCLUSION>Your final conclusion or answer</CONCLUSION>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_RESEARCH_FORMAT: (
                "CRITICAL: After your thinking, provide your response in the following research-style structured format. "  # noqa: E501
                "Note: The sections should summarize your previous thinking and analysis. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<HYPOTHESIS>Research hypothesis or question</HYPOTHESIS>\n"  # noqa: E501
                "<METHODOLOGY>Approach and methodology</METHODOLOGY>\n"  # noqa: E501
                "<ANALYSIS>Data analysis and findings</ANALYSIS>\n"  # noqa: E501
                "<FINDINGS>Key findings and results</FINDINGS>\n"  # noqa: E501
                "<CONCLUSION>Research conclusion</CONCLUSION>"  # noqa: E501
            ),
            # Advanced scratchpad and classification formats
            AnswerFormat.COMPLEX_SCRATCHPAD_FULL: (
                "CRITICAL: After your thinking, provide your response in the following comprehensive scratchpad format. "  # noqa: E501
                "Note: The sections should summarize your previous thinking, not repeat detailed analysis. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<SCRATCHPAD>\n"  # noqa: E501
                "<RESTATEMENT>Restate the problem clearly</RESTATEMENT>\n"  # noqa: E501
                "<REASONING>\n<THOUGHT_1>First insight</THOUGHT_1>\n<THOUGHT_2>Second insight</THOUGHT_2>\n</REASONING>\n"  # noqa: E501
                "<PLAN>\n<STEP_1>First step</STEP_1>\n<CRITIQUE_1>Critique of step 1</CRITIQUE_1>\n<STEP_2>Second step</STEP_2>\n<CRITIQUE_2>Critique of step 2</CRITIQUE_2>\n</PLAN>\n"  # noqa: E501
                "<CITATIONS>\n<CITATION_1>First citation with bibtext in markdown codeblock</CITATION_1>\n</CITATIONS>\n"  # noqa: E501
                "<PYDANTIC_SCHEMAS>\n<SCHEMA_1>First schema</SCHEMA_1>\n</PYDANTIC_SCHEMAS>\n"  # noqa: E501
                "<DIAGRAM>UML workflow diagram in natural language</DIAGRAM>\n"  # noqa: E501
                "<REFLECTION>Harsh internal critique of the plan</REFLECTION>\n"  # noqa: E501
                "<REVISED_PLAN>\n<STEP_1>Revised step 1</STEP_1>\n<STEP_2>Revised step 2</STEP_2>\n</REVISED_PLAN>\n"  # noqa: E501
                "</SCRATCHPAD>\n"  # noqa: E501
                "<SOLUTION>Detailed solution</SOLUTION>\n"  # noqa: E501
                "<EXPLANATION>Solution explanation</EXPLANATION>\n"  # noqa: E501
                "<UNIT_TEST>Unit test code (if applicable)</UNIT_TEST>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_SCRATCHPAD_SIMPLE: (
                "CRITICAL: After your thinking, provide your response in the following simple scratchpad format. "  # noqa: E501
                "Note: The sections should summarize your previous thinking. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<SCRATCHPAD>\n"  # noqa: E501
                "<RESTATEMENT>Restate the problem</RESTATEMENT>\n"  # noqa: E501
                "<REASONING>\n<THOUGHT_1>Key insight 1</THOUGHT_1>\n<THOUGHT_2>Key insight 2</THOUGHT_2>\n</REASONING>\n"  # noqa: E501
                "<PLAN>\n<STEP_1>Step 1</STEP_1>\n<STEP_2>Step 2</STEP_2>\n</PLAN>\n"  # noqa: E501
                "<DIAGRAM>Mermaid workflow diagram in natural language</DIAGRAM>\n"  # noqa: E501
                "</SCRATCHPAD>\n"  # noqa: E501
                "Then provide your solution following the plan step by step."  # noqa: E501
            ),
            AnswerFormat.COMPLEX_CLASSIFICATION_FORMAT: (
                "CRITICAL: After your thinking, provide your response in the following classification format. "  # noqa: E501
                "Analyze the given text and provide binary classifications with explanations. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<COMPLETENESS_EXPLANATION>Explanation for completeness assessment</COMPLETENESS_EXPLANATION>\n"  # noqa: E501
                "<COMPLETENESS_CLASSIFICATION>TRUE or FALSE</COMPLETENESS_CLASSIFICATION>\n"  # noqa: E501
                "<GRAMMAR_EXPLANATION>Explanation for grammar assessment</GRAMMAR_EXPLANATION>\n"  # noqa: E501
                "<GRAMMAR_CLASSIFICATION>TRUE or FALSE</GRAMMAR_CLASSIFICATION>\n"  # noqa: E501
                "<SCIENTIFIC_EXPLANATION>Explanation for scientific nature assessment</SCIENTIFIC_EXPLANATION>\n"  # noqa: E501
                "<SCIENTIFIC_CLASSIFICATION>TRUE or FALSE</SCIENTIFIC_CLASSIFICATION>\n"  # noqa: E501
                "<PROGRAMMING_EXPLANATION>Explanation for programming nature assessment</PROGRAMMING_EXPLANATION>\n"  # noqa: E501
                "<PROGRAMMING_CLASSIFICATION>TRUE or FALSE</PROGRAMMING_CLASSIFICATION>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_ANALYSIS_WITH_ANSWER: (
                "CRITICAL: After your thinking, provide your response in the following analysis format that ends with a specific answer. "  # noqa: E501
                "Note: The sections should summarize your previous thinking. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<ANALYSIS>Comprehensive analysis of the problem</ANALYSIS>\n"  # noqa: E501
                "<METHODOLOGY>Approach and methods used</METHODOLOGY>\n"  # noqa: E501
                "<FINDINGS>Key findings and insights</FINDINGS>\n"  # noqa: E501
                "<ANSWER>Your final answer in the format: ANSWER IS: [your answer here]</ANSWER>"  # noqa: E501
            ),
            AnswerFormat.COMPLEX_EVALUATION_FORMAT: (
                "CRITICAL: After your thinking, provide your response in the following evaluation format. "  # noqa: E501
                "Note: The sections should summarize your previous thinking and analysis. "  # noqa: E501
                "Format:\n"  # noqa: E501
                "<CRITERIA>Evaluation criteria and standards</CRITERIA>\n"  # noqa: E501
                "<ASSESSMENT>\n<CRITERION_1>Assessment of first criterion</CRITERION_1>\n<SCORE_1>Score for criterion 1 (1-10)</SCORE_1>\n<CRITERION_2>Assessment of second criterion</CRITERION_2>\n<SCORE_2>Score for criterion 2 (1-10)</SCORE_2>\n</ASSESSMENT>\n"  # noqa: E501
                "<OVERALL_SCORE>Overall score (1-10)</OVERALL_SCORE>\n"  # noqa: E501
                "<JUDGMENT>Final judgment and recommendation</JUDGMENT>"  # noqa: E501
            ),
            # Custom delimiters
            AnswerFormat.PIPE_DELIMITED: (
                "CRITICAL: After your thinking, provide your answer enclosed in pipe delimiters. "  # noqa: E501
                "Example: |your answer here|"  # noqa: E501
            ),
            AnswerFormat.HASH_DELIMITED: (
                "CRITICAL: After your thinking, provide your answer enclosed in hash delimiters. "  # noqa: E501
                "Example: #your answer here#"  # noqa: E501
            ),
            AnswerFormat.UNDERSCORE_DELIMITED: (
                "CRITICAL: After your thinking, provide your answer enclosed in underscore delimiters. "  # noqa: E501
                "Example: _your answer here_"  # noqa: E501
            ),
            AnswerFormat.TILDE_DELIMITED: (
                "CRITICAL: After your thinking, provide your answer enclosed in tilde delimiters. "  # noqa: E501
                "Example: ~your answer here~"  # noqa: E501
            ),
            # Programming-style formats
            AnswerFormat.FUNCTION_CALL: (
                "CRITICAL: After your thinking, provide your answer as a function call with quoted string EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example: answer("your response here")'  # noqa: E501
            ),
            AnswerFormat.VARIABLE_ASSIGNMENT: (
                "CRITICAL: After your thinking, provide your answer as a variable assignment with quoted string EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example: answer = "your response here"'  # noqa: E501
            ),
            AnswerFormat.RETURN_STATEMENT: (
                "CRITICAL: After your thinking, provide your answer as a return statement with quoted string EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example: return "your response here"'  # noqa: E501
            ),
            # Additional easy-to-parse formats
            AnswerFormat.EQUALS_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with an equals sign prefix. "
                "Example: = your response here"
            ),
            AnswerFormat.DASH_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with a dash prefix. "
                "Example: - your response here"
            ),
            AnswerFormat.PLUS_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with a plus sign prefix. "
                "Example: + your response here"
            ),
            AnswerFormat.STAR_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with a star prefix. "
                "Example: * your response here"
            ),
            AnswerFormat.PERCENT_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with a percent sign prefix. "
                "Example: % your response here"
            ),
            AnswerFormat.AMPERSAND_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with an ampersand prefix. "
                "Example: & your response here"
            ),
            AnswerFormat.AT_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with an at sign prefix. "
                "Example: @ your response here"
            ),
            AnswerFormat.EXCLAMATION_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with an exclamation mark prefix. "
                "Example: ! your response here"
            ),
            AnswerFormat.QUESTION_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with a question mark prefix. "
                "Example: ? your response here"
            ),
            AnswerFormat.SEMICOLON_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with a semicolon prefix. "
                "Example: ; your response here"
            ),
            AnswerFormat.DOUBLE_COLON_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with double colon prefix. "
                "Example: :: your response here"
            ),
            AnswerFormat.TRIPLE_DASH_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with triple dash prefix. "
                "Example: --- your response here"
            ),
            AnswerFormat.DOUBLE_ARROW_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with double arrow prefix. "
                "Example: >> your response here"
            ),
            AnswerFormat.TRIPLE_ARROW_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with triple arrow prefix. "
                "Example: >>> your response here"
            ),
            AnswerFormat.BACKTICK_FORMAT: (
                "CRITICAL: After your thinking, provide your answer enclosed in single backticks. "
                "Example: `your response here`"
            ),
            AnswerFormat.DOUBLE_BACKTICK_FORMAT: (
                "CRITICAL: After your thinking, provide your answer enclosed in double backticks. "
                "Example: ``your response here``"
            ),
            AnswerFormat.QUOTE_FORMAT: (
                "CRITICAL: After your thinking, provide your answer enclosed in double quotes. "
                'Example: "your response here"'
            ),
            AnswerFormat.SINGLE_QUOTE_FORMAT: (
                "CRITICAL: After your thinking, provide your answer enclosed in single quotes. "
                "Example: 'your response here'"
            ),
            AnswerFormat.TRIPLE_QUOTE_FORMAT: (
                "CRITICAL: After your thinking, provide your answer enclosed in triple quotes. "
                'Example: """your response here"""'
            ),
            AnswerFormat.ANSWER_IS_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with 'ANSWER IS:' prefix in all caps. "  # noqa: E501
                "Example: ANSWER IS: your response here"  # noqa: E501
            ),
            AnswerFormat.SOLUTION_IS_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with 'SOLUTION IS:' prefix in all caps. "  # noqa: E501
                "Example: SOLUTION IS: your response here"  # noqa: E501
            ),
            AnswerFormat.RESULT_IS_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with 'RESULT IS:' prefix in all caps. "  # noqa: E501
                "Example: RESULT IS: your response here"  # noqa: E501
            ),
            AnswerFormat.OUTPUT_IS_FORMAT: (
                "CRITICAL: After your thinking, provide your answer with 'OUTPUT IS:' prefix in all caps. "  # noqa: E501
                "Example: OUTPUT IS: your response here"  # noqa: E501
            ),
            # Code-specific formats (for code datasets)
            AnswerFormat.PYTHON_PRINT: (
                "CRITICAL: After your thinking, provide your answer as a Python print statement EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example: print("your answer here")'  # noqa: E501
            ),
            AnswerFormat.JAVASCRIPT_CONSOLE: (
                "CRITICAL: After your thinking, provide your answer as a JavaScript console.log statement EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example: console.log("your answer here")'  # noqa: E501
            ),
            AnswerFormat.PYTHON_COMMENT: (
                "CRITICAL: After your thinking, provide your answer as a Python comment EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: # your answer here"  # noqa: E501
            ),
            AnswerFormat.JAVASCRIPT_COMMENT: (
                "CRITICAL: After your thinking, provide your answer as a JavaScript comment EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: // your answer here"  # noqa: E501
            ),
            AnswerFormat.C_COMMENT: (
                "CRITICAL: After your thinking, provide your answer as a C-style block comment EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example: /* your answer here */"  # noqa: E501
            ),
            AnswerFormat.SHELL_ECHO: (
                "CRITICAL: After your thinking, provide your answer as a shell echo command EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example: echo "your answer here"'  # noqa: E501
            ),
            AnswerFormat.SHELL_OUTPUT: (
                "CRITICAL: After your thinking, provide your answer as shell command output EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. This format is for code-related content. "  # noqa: E501
                "Example: $ your answer here"  # noqa: E501
            ),
            AnswerFormat.PYTHON_DOCSTRING: (
                "CRITICAL: After your thinking, provide your answer as a Python docstring EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example: """your answer here"""'  # noqa: E501
            ),
            AnswerFormat.INI_FORMAT: (
                "CRITICAL: After your thinking, provide your answer in INI configuration format EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                "Example:\n[section]\nanswer = your response here"  # noqa: E501
            ),
            AnswerFormat.ENV_FORMAT: (
                "CRITICAL: After your thinking, provide your answer as an environment variable EXACTLY ONCE. "  # noqa: E501
                "Use this format only once in your entire response. "  # noqa: E501
                'Example: ANSWER="your response here"'  # noqa: E501
            ),
        }

        # Dynamic formats are commented out - all formats use static instructions
        instruction = format_instructions.get(
            answer_format, "Provide your answer after thinking."
        )

        return f"{base_prompt}\n\n{instruction}\n\nREMEMBER: Use the specified format EXACTLY ONCE and ONLY ONCE. Do not repeat the format or use it multiple times anywhere in your response."  # noqa: E501

    def _generate_dynamic_format_instruction(
        self, answer_format: AnswerFormat
    ) -> Tuple[str, List[str]]:
        """Generate dynamic format instructions by randomly selecting components.

        Returns:
            Tuple of (instruction_text, selected_components)
        """

        # Define component pools
        analysis_components = [
            "restatement",
            "problem_analysis",
            "context_analysis",
            "requirements_analysis",
        ]

        reasoning_components = [
            "reasoning",
            "insights",
            "key_points",
            "considerations",
            "assumptions",
        ]

        planning_components = ["plan", "approach", "methodology", "strategy", "steps"]

        technical_components = [
            "schemas",
            "diagrams",
            "models",
            "architecture",
            "specifications",
        ]

        execution_components = [
            "implementation",
            "solution",
            "code",
            "execution",
            "results",
        ]

        reflection_components = [
            "reflection",
            "evaluation",
            "verification",
            "testing",
            "validation",
        ]

        # Select 3-5 random components based on format type
        if "scratchpad" in answer_format.value:
            # Scratchpad formats get a mix of analysis, reasoning, and planning
            available_components = (
                analysis_components + reasoning_components + planning_components
            )
        elif "analysis" in answer_format.value:
            # Analysis formats focus on analysis and reasoning
            available_components = (
                analysis_components + reasoning_components + reflection_components
            )
        elif "workflow" in answer_format.value:
            # Workflow formats get planning, execution, and reflection
            available_components = (
                planning_components + execution_components + reflection_components
            )
        else:
            # Default: mix of all
            available_components = (
                analysis_components
                + reasoning_components
                + planning_components
                + technical_components
            )

        # Randomly select 3-5 components
        num_components = random.randint(3, 5)
        selected_components = random.sample(
            available_components, min(num_components, len(available_components))
        )

        # Generate format-specific instructions
        if "xml" in answer_format.value:
            instruction = self._generate_xml_format_instruction(selected_components)
        elif "json" in answer_format.value:
            instruction = self._generate_json_format_instruction(selected_components)
        elif "yaml" in answer_format.value:
            instruction = self._generate_yaml_format_instruction(
                selected_components
            )  # noqa
        elif "toml" in answer_format.value:
            instruction = self._generate_toml_format_instruction(
                selected_components
            )  # noqa
        else:
            instruction = f"Please structure your response using the following components: {', '.join(selected_components)}"  # noqa

        # Add component info to instruction for debugging
        component_list = ", ".join(selected_components)

        if self.debug_logging:
            self.logger.debug(
                f"Generated dynamic format {answer_format.value} with components: {component_list}"  # noqa
            )

        return (
            f"{instruction}\n\nSelected components: {component_list}",
            selected_components,
        )

    def _generate_xml_format_instruction(self, components: List[str]) -> str:
        """Generate XML format instruction with selected components."""
        component_examples = []
        for comp in components:
            tag_name = comp.upper()
            if comp in ["reasoning", "insights"]:
                component_examples.append(
                    f"<{tag_name}>\n<POINT_1>First key point</POINT_1>\n<POINT_2>Second key point</POINT_2>\n</{tag_name}>"  # noqa
                )
            elif comp in ["plan", "steps"]:
                component_examples.append(
                    f"<{tag_name}>\n<STEP_1>First step</STEP_1>\n<STEP_2>Second step</STEP_2>\n</{tag_name}>"  # noqa
                )
            elif comp in ["schemas", "models"]:
                component_examples.append(
                    f"<{tag_name}>\n<SCHEMA_1>First schema</SCHEMA_1>\n<SCHEMA_2>Second schema</SCHEMA_2>\n</{tag_name}>"  # noqa
                )
            else:
                component_examples.append(
                    f"<{tag_name}>Content for {comp}</{tag_name}>"
                )

        format_example = "\n".join(component_examples)
        return (
            f"CRITICAL: After your thinking, provide your response using the following XML format with these specific components. "  # noqa
            f"Note: The sections should summarize your previous thinking. "  # noqa
            f"Format:\n{format_example}"
        )

    def _generate_json_format_instruction(self, components: List[str]) -> str:
        """Generate JSON format instruction with selected components."""
        json_fields = []
        for comp in components:
            if comp in ["reasoning", "insights"]:
                json_fields.append(
                    f'"{comp}": {{"point_1": "First key point", "point_2": "Second key point"}}'  # noqa
                )
            elif comp in ["plan", "steps"]:
                json_fields.append(
                    f'"{comp}": {{"step_1": "First step", "step_2": "Second step"}}'  # noqa
                )
            elif comp in ["schemas", "models"]:
                json_fields.append(
                    f'"{comp}": {{"schema_1": "First schema", "schema_2": "Second schema"}}'  # noqa
                )
            else:
                json_fields.append(f'"{comp}": "Content for {comp}"')

        format_example = "{\n  " + ",\n  ".join(json_fields) + "\n}"
        return (
            f"CRITICAL: After your thinking, provide your response as a JSON object with these specific fields. "  # noqa
            f"Note: The sections should summarize your previous thinking. "
            f"Format:\n{format_example}"
        )

    def _generate_yaml_format_instruction(self, components: List[str]) -> str:
        """Generate YAML format instruction with selected components."""
        yaml_fields = []
        for comp in components:
            if comp in ["reasoning", "insights"]:
                yaml_fields.append(
                    f"{comp}:\n  point_1: First key point\n  point_2: Second key point"  # noqa
                )
            elif comp in ["plan", "steps"]:
                yaml_fields.append(
                    f"{comp}:\n  step_1: First step\n  step_2: Second step"  # noqa
                )
            elif comp in ["schemas", "models"]:
                yaml_fields.append(
                    f"{comp}:\n  schema_1: First schema\n  schema_2: Second schema"  # noqa
                )
            else:
                yaml_fields.append(f"{comp}: Content for {comp}")

        format_example = "\n".join(yaml_fields)
        return (
            f"CRITICAL: After your thinking, provide your response in YAML format with these specific fields. "  # noqa
            f"Note: The sections should summarize your previous thinking. "  # noqa
            f"Format:\n{format_example}"
        )

    def _generate_toml_format_instruction(self, components: List[str]) -> str:
        """Generate TOML format instruction with selected components."""
        toml_fields = []
        for comp in components:
            if comp in ["reasoning", "insights"]:
                toml_fields.append(
                    f'[{comp}]\npoint_1 = "First key point"\npoint_2 = "Second key point"'  # noqa
                )
            elif comp in ["plan", "steps"]:
                toml_fields.append(
                    f'[{comp}]\nstep_1 = "First step"\nstep_2 = "Second step"'  # noqa
                )
            elif comp in ["schemas", "models"]:
                toml_fields.append(
                    f'[{comp}]\nschema_1 = "First schema"\nschema_2 = "Second schema"'  # noqa
                )
            else:
                toml_fields.append(f'{comp} = "Content for {comp}"')

        format_example = "\n\n".join(toml_fields)
        return (
            f"CRITICAL: After your thinking, provide your response in TOML format with these specific fields. "  # noqa
            f"Note: The sections should summarize your previous thinking. "  # noqa
            f"Format:\n{format_example}"
        )

    def _get_formats_for_dataset_type(self, dataset_type: str) -> List[AnswerFormat]:
        """Get appropriate formats for a given dataset type."""
        if dataset_type == "math_only":
            # Math datasets can use generic + math-only formats
            available_formats = list(self.generic_formats | self.math_only_formats)
        elif dataset_type == "code_only":
            # Code datasets can use generic + code-only formats
            available_formats = list(self.generic_formats | self.code_only_formats)
        else:  # "generic" or any other type
            # Generic datasets use only generic formats
            available_formats = list(self.generic_formats)

        # Filter by base supported formats
        filtered_formats = [
            f for f in available_formats if f in self.base_supported_formats
        ]

        # Apply equivalent ratio filter if enabled
        final_formats = self._get_balanced_formats(filtered_formats)

        if self.debug_logging:
            self.logger.debug(
                f"Dataset type '{dataset_type}': {len(filtered_formats)} available formats, "
                f"{len(final_formats)} after equivalent ratio filtering"
            )

        return final_formats

    def _get_dynamic_format_patterns(
        self, answer_format: AnswerFormat, stored_components: List[str]
    ) -> List[str]:
        """Generate specific validation patterns for dynamic formats based on selected components."""  # noqa
        if not stored_components:
            return []

        patterns = []

        if "xml" in answer_format.value:
            # Generate specific XML tag patterns for each component
            for component in stored_components:
                tag_name = component.upper()
                patterns.append(f"<{tag_name}>.*?</{tag_name}>")
        elif "json" in answer_format.value:
            # For JSON, we'll validate that it's a valid JSON object with the expected fields
            patterns.append(r"\{.*?\}")  # Basic JSON object pattern
        elif "yaml" in answer_format.value:
            # Generate specific YAML field patterns for each component
            for component in stored_components:
                patterns.append(f"^{component}\\s*:.*$")
        elif "toml" in answer_format.value:
            # Generate specific TOML field patterns for each component
            for component in stored_components:
                patterns.append(f"^{component}\\s*=.*$")

        return patterns

    def _validate_format_appears_exactly_once(
        self, text: str, answer_format: AnswerFormat
    ) -> bool:
        """Validate that the specified format appears exactly once in the response section."""
        # Define patterns for each format to count occurrences
        format_patterns = {
            # Basic structured data formats (answer only)
            AnswerFormat.JSON: [r'\{\s*"answer"\s*:\s*"[^"]*"\s*\}'],
            AnswerFormat.JSON_ARRAY: [r'\[\s*"[^"]*"\s*\]'],
            AnswerFormat.JSON_SIMPLE: [r'"[^"]*"'],
            AnswerFormat.YAML: [r"^answer\s*:\s*.+$"],
            AnswerFormat.YAML_LIST: [r"^-\s*.+$"],
            AnswerFormat.TOML: [r"^answer\s*=\s*.+$"],
            AnswerFormat.TOML_SECTION: [r"^\[response\]", r"^answer\s*=\s*.+$"],
            # Structured data with confidence scores
            AnswerFormat.JSON_CONFIDENCE: [
                r'\{\s*"answer"\s*:\s*"[^"]*"\s*,\s*"confidence"\s*:\s*[0-9.]+\s*\}'
            ],
            AnswerFormat.YAML_CONFIDENCE: [
                r"^answer\s*:\s*.+$",
                r"^confidence\s*:\s*[0-9.]+$",
            ],
            AnswerFormat.TOML_CONFIDENCE: [
                r"^answer\s*=\s*.+$",
                r"^confidence\s*=\s*[0-9.]+$",
            ],
            # XML/HTML tag variations (XML now uses answer tags)
            AnswerFormat.XML: [r"<answer>.*?</answer>"],
            AnswerFormat.XML_FINAL_ANSWER: [r"<answer>Final Answer:.*?</answer>"],
            AnswerFormat.OUTPUT_TAGS: [r"<output>.*?</output>"],
            AnswerFormat.RESULT_TAGS: [r"<result>.*?</result>"],
            AnswerFormat.RESPONSE_TAGS: [r"<response>.*?</response>"],
            AnswerFormat.FINAL_ANSWER_TAGS: [r"<final_answer>.*?</final_answer>"],
            AnswerFormat.SOLUTION_TAGS: [r"<solution>.*?</solution>"],
            AnswerFormat.CONCLUSION_TAGS: [r"<conclusion>.*?</conclusion>"],
            AnswerFormat.REPLY_TAGS: [r"<reply>.*?</reply>"],
            AnswerFormat.NESTED_RESPONSE_ANSWER: [
                r"<response>.*?<answer>.*?</answer></response>"
            ],
            AnswerFormat.NESTED_SOLUTION_ANSWER: [
                r"<solution>.*?<answer>.*?</answer></solution>"
            ],
            AnswerFormat.NESTED_OUTPUT_RESULT: [
                r"<output>.*?<result>.*?</result></output>"
            ],
            AnswerFormat.NESTED_ANALYSIS_CONCLUSION: [
                r"<analysis>.*?<conclusion>.*?</conclusion></analysis>"
            ],
            AnswerFormat.NESTED_REASONING_ANSWER: [
                r"<reasoning>.*?<answer>.*?</answer></reasoning>"
            ],
            # LaTeX formats (text-friendly)
            AnswerFormat.LATEX_BOXED: [r"\\boxed\{[^}]+\}"],
            AnswerFormat.LATEX_TEXTBF: [r"\\textbf\{[^}]+\}"],
            AnswerFormat.LATEX_TEXTIT: [r"\\textit\{[^}]+\}"],
            AnswerFormat.LATEX_UNDERLINE: [r"\\underline\{[^}]+\}"],
            # LaTeX formats (math-only)
            AnswerFormat.LATEX_BOXED_MATH: [r"\$\\boxed\{[^}]+\}\$"],
            AnswerFormat.LATEX_ALIGN: [r"\\begin\{align\}.*?\\end\{align\}"],
            AnswerFormat.LATEX_EQUATION: [r"\\begin\{equation\}.*?\\end\{equation\}"],
            AnswerFormat.LATEX_DISPLAYMATH: [r"\\\\?\[.*?\\\\?\]"],
            AnswerFormat.LATEX_INLINE_MATH: [r"\$[^$]+\$"],
            AnswerFormat.LATEX_TEXT_MATH: [r"\$\\text\{[^}]+\}\$"],
            AnswerFormat.LATEX_MATHRM: [r"\$\\mathrm\{[^}]+\}\$"],
            AnswerFormat.LATEX_THEREFORE: [r"\$\\therefore[^$]+\$"],
            AnswerFormat.LATEX_IMPLIES: [r"\$\\implies[^$]+\$"],
            AnswerFormat.LATEX_EQUIV: [r"\$[^$]*\\equiv[^$]*\$"],
            AnswerFormat.LATEX_MATRIX: [r"\$\\begin\{matrix\}.*?\\end\{matrix\}\$"],
            AnswerFormat.LATEX_PMATRIX: [r"\$\\begin\{pmatrix\}.*?\\end\{pmatrix\}\$"],
            # Markdown formats
            AnswerFormat.MARKDOWN_CODE: [r"```\s*\n.*?\n```"],
            AnswerFormat.MARKDOWN_BOLD: [r"\*\*[^*]+\*\*"],
            AnswerFormat.MARKDOWN_ITALIC: [r"\*[^*]+\*"],
            AnswerFormat.MARKDOWN_HEADER: [r"^##\s*.+?(?:\n|$)"],
            AnswerFormat.MARKDOWN_QUOTE: [r"^>\s*.+?(?:\n|$)"],
            # Bracket and delimiter formats
            AnswerFormat.SQUARE_BRACKETS: [r"\[[^\]]+\]"],
            AnswerFormat.DOUBLE_SQUARE_BRACKETS: [r"\[\[[^\]]+\]\]"],
            AnswerFormat.CURLY_BRACES: [r"\{[^}]+\}"],
            AnswerFormat.PARENTHESES: [r"\([^)]+\)"],
            AnswerFormat.ANGLE_BRACKETS: [r"<[^>]+>"],
            # Natural language patterns
            AnswerFormat.NATURAL_LANGUAGE_ANSWER: [r"The answer is:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_FINAL: [r"Final answer:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_CONCLUSION: [
                r"In conclusion:?\s*.+?(?:\n|$)"
            ],
            AnswerFormat.NATURAL_LANGUAGE_THEREFORE: [r"Therefore:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_RESULT: [r"The result is:?\s*.+?(?:\n|$)"],
            # Additional natural language patterns
            AnswerFormat.NATURAL_LANGUAGE_BEST: [r"The best answer is:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_MY_FINAL: [
                r"My final answer is:?\s*.+?(?:\n|$)"
            ],
            AnswerFormat.NATURAL_LANGUAGE_CORRECT: [
                r"The correct answer is:?\s*.+?(?:\n|$)"
            ],
            AnswerFormat.NATURAL_LANGUAGE_SOLUTION: [
                r"The solution is:?\s*.+?(?:\n|$)"
            ],
            AnswerFormat.NATURAL_LANGUAGE_RESPONSE: [r"My response is:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_ULTIMATELY: [r"Ultimately:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_THUS: [r"Thus:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_HENCE: [r"Hence:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_CONSEQUENTLY: [
                r"Consequently:?\s*.+?(?:\n|$)"
            ],
            AnswerFormat.NATURAL_LANGUAGE_TO_SUMMARIZE: [
                r"To summarize:?\s*.+?(?:\n|$)"
            ],
            AnswerFormat.NATURAL_LANGUAGE_IN_SUMMARY: [r"In summary:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_OVERALL: [r"Overall:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_FINAL_VERDICT: [
                r"Final verdict:?\s*.+?(?:\n|$)"
            ],
            AnswerFormat.NATURAL_LANGUAGE_BOTTOM_LINE: [r"Bottom line:?\s*.+?(?:\n|$)"],
            AnswerFormat.NATURAL_LANGUAGE_KEY_POINT: [
                r"The key point is:?\s*.+?(?:\n|$)"
            ],
            # Special formats
            AnswerFormat.TEXTARENA_FORMAT: [r"\[[A-Da-d]\]"],
            AnswerFormat.COLON_FORMAT: [r"Answer:?\s*.+?(?:\n|$)"],
            AnswerFormat.ARROW_FORMAT: [
                r"=>\s*.+?(?:\n|$)",
                r"->\s*.+?(?:\n|$)",
                r"→\s*.+?(?:\n|$)",
            ],
            # HTML formats
            AnswerFormat.HTML_CODE: [r"<code>.*?</code>"],
            AnswerFormat.HTML_PRE: [r"<pre>.*?</pre>"],
            AnswerFormat.HTML_SPAN: [r"<span>.*?</span>"],
            AnswerFormat.HTML_DIV: [r"<div>.*?</div>"],
            AnswerFormat.HTML_P: [r"<p>.*?</p>"],
            # Multiple structured tags - special case
            AnswerFormat.MULTIPLE_TAGS: [
                r"<theory>.*?</theory>",
                r"<answer>.*?</answer>",
                r"<explanation>.*?</explanation>",
            ],
            # Complex multi-tag formats
            AnswerFormat.COMPLEX_CODING_FORMAT: [
                r"<RESTATEMENT>.*?</RESTATEMENT>",
                r"<REASONING>.*?</REASONING>",
                r"<PLAN>.*?</PLAN>",
                r"<PYDANTIC_SCHEMAS>.*?</PYDANTIC_SCHEMAS>",
                r"<DIAGRAM>.*?</DIAGRAM>",
                r"<REFLECTION>.*?</REFLECTION>",
                r"<SOLUTION>.*?</SOLUTION>",
                r"<EXPLANATION>.*?</EXPLANATION>",
                r"<UNIT_TEST>.*?</UNIT_TEST>",
            ],
            AnswerFormat.COMPLEX_CODING_SIMPLE: [
                r"<RESTATEMENT>.*?</RESTATEMENT>",
                r"<REASONING>.*?</REASONING>",
                r"<PLAN>.*?</PLAN>",
                r"<SOLUTION>.*?</SOLUTION>",
                r"<EXPLANATION>.*?</EXPLANATION>",
            ],
            AnswerFormat.COMPLEX_CODING_MINIMAL: [
                r"<ANALYSIS>.*?</ANALYSIS>",
                r"<APPROACH>.*?</APPROACH>",
                r"<SOLUTION>.*?</SOLUTION>",
                r"<TEST>.*?</TEST>",
            ],
            AnswerFormat.COMPLEX_MATH_FORMAT: [
                r"<RESTATEMENT>.*?</RESTATEMENT>",
                r"<REASONING>.*?</REASONING>",
                r"<APPROACH>.*?</APPROACH>",
                r"<DERIVATION>.*?</DERIVATION>",
                r"<SOLUTION>.*?</SOLUTION>",
                r"<VERIFICATION>.*?</VERIFICATION>",
            ],
            AnswerFormat.COMPLEX_MATH_SIMPLE: [
                r"<PROBLEM_ANALYSIS>.*?</PROBLEM_ANALYSIS>",
                r"<SOLUTION_STEPS>.*?</SOLUTION_STEPS>",
                r"<FINAL_ANSWER>.*?</FINAL_ANSWER>",
            ],
            AnswerFormat.COMPLEX_GENERAL_FORMAT: [
                r"<RESTATEMENT>.*?</RESTATEMENT>",
                r"<ANALYSIS>.*?</ANALYSIS>",
                r"<REASONING>.*?</REASONING>",
                r"<CONCLUSION>.*?</CONCLUSION>",
                r"<REFLECTION>.*?</REFLECTION>",
            ],
            AnswerFormat.COMPLEX_GENERAL_SIMPLE: [
                r"<ANALYSIS>.*?</ANALYSIS>",
                r"<REASONING>.*?</REASONING>",
                r"<CONCLUSION>.*?</CONCLUSION>",
            ],
            AnswerFormat.COMPLEX_RESEARCH_FORMAT: [
                r"<HYPOTHESIS>.*?</HYPOTHESIS>",
                r"<METHODOLOGY>.*?</METHODOLOGY>",
                r"<ANALYSIS>.*?</ANALYSIS>",
                r"<FINDINGS>.*?</FINDINGS>",
                r"<CONCLUSION>.*?</CONCLUSION>",
            ],
            # Advanced scratchpad and classification formats
            AnswerFormat.COMPLEX_SCRATCHPAD_FULL: [
                r"<SCRATCHPAD>.*?</SCRATCHPAD>",
                r"<RESTATEMENT>.*?</RESTATEMENT>",
                r"<REASONING>.*?</REASONING>",
                r"<PLAN>.*?</PLAN>",
                r"<CITATIONS>.*?</CITATIONS>",
                r"<PYDANTIC_SCHEMAS>.*?</PYDANTIC_SCHEMAS>",
                r"<DIAGRAM>.*?</DIAGRAM>",
                r"<REFLECTION>.*?</REFLECTION>",
                r"<REVISED_PLAN>.*?</REVISED_PLAN>",
                r"<SOLUTION>.*?</SOLUTION>",
                r"<EXPLANATION>.*?</EXPLANATION>",
            ],
            AnswerFormat.COMPLEX_SCRATCHPAD_SIMPLE: [
                r"<SCRATCHPAD>.*?</SCRATCHPAD>",
                r"<RESTATEMENT>.*?</RESTATEMENT>",
                r"<REASONING>.*?</REASONING>",
                r"<PLAN>.*?</PLAN>",
                r"<DIAGRAM>.*?</DIAGRAM>",
            ],
            AnswerFormat.COMPLEX_CLASSIFICATION_FORMAT: [
                r"<COMPLETENESS_EXPLANATION>.*?</COMPLETENESS_EXPLANATION>",
                r"<COMPLETENESS_CLASSIFICATION>.*?</COMPLETENESS_CLASSIFICATION>",
                r"<GRAMMAR_EXPLANATION>.*?</GRAMMAR_EXPLANATION>",
                r"<GRAMMAR_CLASSIFICATION>.*?</GRAMMAR_CLASSIFICATION>",
                r"<SCIENTIFIC_EXPLANATION>.*?</SCIENTIFIC_EXPLANATION>",
                r"<SCIENTIFIC_CLASSIFICATION>.*?</SCIENTIFIC_CLASSIFICATION>",
                r"<PROGRAMMING_EXPLANATION>.*?</PROGRAMMING_EXPLANATION>",
                r"<PROGRAMMING_CLASSIFICATION>.*?</PROGRAMMING_CLASSIFICATION>",
            ],
            AnswerFormat.COMPLEX_ANALYSIS_WITH_ANSWER: [
                r"<ANALYSIS>.*?</ANALYSIS>",
                r"<METHODOLOGY>.*?</METHODOLOGY>",
                r"<FINDINGS>.*?</FINDINGS>",
                r"<ANSWER>.*?</ANSWER>",
            ],
            AnswerFormat.COMPLEX_EVALUATION_FORMAT: [
                r"<CRITERIA>.*?</CRITERIA>",
                r"<ASSESSMENT>.*?</ASSESSMENT>",
                r"<OVERALL_SCORE>.*?</OVERALL_SCORE>",
                r"<JUDGMENT>.*?</JUDGMENT>",
            ],
            # Dynamic formats - COMMENTED OUT: These formats are proving too difficult to parse reliably
            # AnswerFormat.DYNAMIC_SCRATCHPAD_XML: [r'<[A-Z_]+>.*?</[A-Z_]+>'],
            # AnswerFormat.DYNAMIC_SCRATCHPAD_JSON: [r'\{.*?\}'],
            # AnswerFormat.DYNAMIC_SCRATCHPAD_YAML: [r'^[a-z_]+:.*$'],
            # AnswerFormat.DYNAMIC_SCRATCHPAD_TOML: [r'^[a-z_]+\s*=.*$', r'^\[[a-z_]+\]$'],
            # AnswerFormat.DYNAMIC_ANALYSIS_XML: [r'<[A-Z_]+>.*?</[A-Z_]+>'],
            # AnswerFormat.DYNAMIC_ANALYSIS_JSON: [r'\{.*?\}'],
            # AnswerFormat.DYNAMIC_WORKFLOW_XML: [r'<[A-Z_]+>.*?</[A-Z_]+>'],
            # AnswerFormat.DYNAMIC_WORKFLOW_JSON: [r'\{.*?\}'],
            # Custom delimiters
            AnswerFormat.PIPE_DELIMITED: [r"\|[^|]+\|"],
            AnswerFormat.HASH_DELIMITED: [r"#[^#]+#"],
            AnswerFormat.UNDERSCORE_DELIMITED: [r"_[^_]+_"],
            AnswerFormat.TILDE_DELIMITED: [r"~[^~]+~"],
            # Programming-style formats
            AnswerFormat.FUNCTION_CALL: [r"answer\([^)]*\)"],
            AnswerFormat.VARIABLE_ASSIGNMENT: [r"answer\s*=\s*[^;\n]+"],
            AnswerFormat.RETURN_STATEMENT: [r"return\s+[^;\n]+"],
            # Additional easy-to-parse formats
            AnswerFormat.EQUALS_FORMAT: [r"^=\s*.+?(?:\n|$)"],
            AnswerFormat.DASH_FORMAT: [r"^-\s*.+?(?:\n|$)"],
            AnswerFormat.PLUS_FORMAT: [r"^\+\s*.+?(?:\n|$)"],
            AnswerFormat.STAR_FORMAT: [r"^\*\s*.+?(?:\n|$)"],
            AnswerFormat.PERCENT_FORMAT: [r"^%\s*.+?(?:\n|$)"],
            AnswerFormat.AMPERSAND_FORMAT: [r"^&\s*.+?(?:\n|$)"],
            AnswerFormat.AT_FORMAT: [r"^@\s*.+?(?:\n|$)"],
            AnswerFormat.EXCLAMATION_FORMAT: [r"^!\s*.+?(?:\n|$)"],
            AnswerFormat.QUESTION_FORMAT: [r"^\?\s*.+?(?:\n|$)"],
            AnswerFormat.SEMICOLON_FORMAT: [r"^;\s*.+?(?:\n|$)"],
            AnswerFormat.DOUBLE_COLON_FORMAT: [r"^::\s*.+?(?:\n|$)"],
            AnswerFormat.TRIPLE_DASH_FORMAT: [r"^---\s*.+?(?:\n|$)"],
            AnswerFormat.DOUBLE_ARROW_FORMAT: [r"^>>\s*.+?(?:\n|$)"],
            AnswerFormat.TRIPLE_ARROW_FORMAT: [r"^>>>\s*.+?(?:\n|$)"],
            AnswerFormat.BACKTICK_FORMAT: [r"`[^`]+`"],
            AnswerFormat.DOUBLE_BACKTICK_FORMAT: [r"``[^`]+``"],
            AnswerFormat.QUOTE_FORMAT: [r'"[^"]+"'],
            AnswerFormat.SINGLE_QUOTE_FORMAT: [r"'[^']+'"],
            AnswerFormat.TRIPLE_QUOTE_FORMAT: [r'"""[^"]+"""'],
            AnswerFormat.ANSWER_IS_FORMAT: [r"ANSWER IS:?\s*.+?(?:\n|$)"],
            AnswerFormat.SOLUTION_IS_FORMAT: [r"SOLUTION IS:?\s*.+?(?:\n|$)"],
            AnswerFormat.RESULT_IS_FORMAT: [r"RESULT IS:?\s*.+?(?:\n|$)"],
            AnswerFormat.OUTPUT_IS_FORMAT: [r"OUTPUT IS:?\s*.+?(?:\n|$)"],
            # Code-specific formats
            AnswerFormat.PYTHON_PRINT: [r'print\s*\(\s*"[^"]*"\s*\)'],
            AnswerFormat.JAVASCRIPT_CONSOLE: [r'console\.log\s*\(\s*"[^"]*"\s*\)'],
            AnswerFormat.PYTHON_COMMENT: [r"^#\s*.+?(?:\n|$)"],
            AnswerFormat.JAVASCRIPT_COMMENT: [r"^//\s*.+?(?:\n|$)"],
            AnswerFormat.C_COMMENT: [r"/\*.*?\*/"],
            AnswerFormat.SHELL_ECHO: [r'echo\s+"[^"]*"'],
            AnswerFormat.SHELL_OUTPUT: [r"^\$\s*.+?(?:\n|$)"],
            AnswerFormat.PYTHON_DOCSTRING: [r'"""[^"]*"""'],
            AnswerFormat.INI_FORMAT: [r"^\[.*?\]", r"^.*?\s*=\s*.+?(?:\n|$)"],
            AnswerFormat.ENV_FORMAT: [r'^[A-Z_]+\s*=\s*"[^"]*"'],
        }

        patterns = format_patterns.get(answer_format, [])

        # Dynamic formats are commented out - no special handling needed

        if not patterns:
            return True  # If no patterns defined, assume valid

        total_matches = 0
        for pattern in patterns:
            matches = re.findall(
                pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE
            )
            total_matches += len(matches)

        # Special cases for formats that require multiple elements
        if answer_format == AnswerFormat.MULTIPLE_TAGS:
            return 2 <= total_matches <= 5
        elif answer_format == AnswerFormat.COMPLEX_CODING_FORMAT:
            return total_matches == 9  # All 9 required tags
        elif answer_format == AnswerFormat.COMPLEX_CODING_SIMPLE:
            return total_matches == 5  # All 5 required tags
        elif answer_format == AnswerFormat.COMPLEX_CODING_MINIMAL:
            return total_matches == 4  # All 4 required tags
        elif answer_format == AnswerFormat.COMPLEX_MATH_FORMAT:
            return total_matches == 6  # All 6 required tags
        elif answer_format == AnswerFormat.COMPLEX_MATH_SIMPLE:
            return total_matches == 3  # All 3 required tags
        elif answer_format == AnswerFormat.COMPLEX_GENERAL_FORMAT:
            return total_matches == 5  # All 5 required tags
        elif answer_format == AnswerFormat.COMPLEX_GENERAL_SIMPLE:
            return total_matches == 3  # All 3 required tags
        elif answer_format == AnswerFormat.COMPLEX_RESEARCH_FORMAT:
            return total_matches == 5  # All 5 required tags
        elif answer_format == AnswerFormat.COMPLEX_SCRATCHPAD_FULL:
            return total_matches == 11  # All 11 required tags
        elif answer_format == AnswerFormat.COMPLEX_SCRATCHPAD_SIMPLE:
            return total_matches == 5  # All 5 required tags
        elif answer_format == AnswerFormat.COMPLEX_CLASSIFICATION_FORMAT:
            return (
                total_matches == 8
            )  # All 8 required tags (4 explanations + 4 classifications)
        elif answer_format == AnswerFormat.COMPLEX_ANALYSIS_WITH_ANSWER:
            return total_matches == 4  # All 4 required tags
        elif answer_format == AnswerFormat.COMPLEX_EVALUATION_FORMAT:
            return total_matches == 4  # All 4 required tags
        # Dynamic formats are commented out - no special handling needed
        elif answer_format in [
            AnswerFormat.YAML_CONFIDENCE,
            AnswerFormat.TOML_CONFIDENCE,
        ]:
            return total_matches == 2  # Need both answer and confidence fields
        elif answer_format in [AnswerFormat.TOML_SECTION, AnswerFormat.INI_FORMAT]:
            return total_matches == 2  # Need both section and key=value
        elif answer_format in [
            AnswerFormat.NESTED_RESPONSE_ANSWER,
            AnswerFormat.NESTED_SOLUTION_ANSWER,
            AnswerFormat.NESTED_OUTPUT_RESULT,
            AnswerFormat.NESTED_ANALYSIS_CONCLUSION,
            AnswerFormat.NESTED_REASONING_ANSWER,
        ]:
            return (
                total_matches == 1
            )  # Each nested format should appear exactly once as a complete unit

        # For all other formats, we want exactly 1 occurrence
        return total_matches == 1

    def _extract_answer_content(
        self,
        text: str,
        answer_format: AnswerFormat,
        dataset_item: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Extract answer content based on the specified format with strict thinking tag validation."""
        if self.debug_logging:
            self.logger.debug(
                f"Extracting {answer_format.value} from response (length: {len(text)})"
            )

        # Ensure text is a string
        if not isinstance(text, str):
            if self.debug_logging:
                self.logger.warning(
                    f"Expected string but got {type(text)}, converting to string"
                )
            text = str(text)

        # 1. Validate thinking tags
        think_tags = re.findall(r"<think>", text, re.IGNORECASE)
        think_close_tags = re.findall(r"</think>", text, re.IGNORECASE)

        if len(think_tags) != 1 or len(think_close_tags) != 1:
            if self.debug_logging:
                self.logger.warning(
                    f"Invalid thinking tag structure: {len(think_tags)} open tags, {len(think_close_tags)} close tags"
                )
            return None

        # Split the text into thinking and response sections
        match_think_block = re.match(
            r"(.*?)(<think>.*?</think>)(.*)", text, re.DOTALL | re.IGNORECASE
        )
        if not match_think_block:
            if self.debug_logging:
                self.logger.warning(
                    "Could not find a complete <think>...</think> block"
                )
            return None

        response_section = match_think_block.group(3)  # Content after </think>

        # Validate that there are no additional thinking tags in the response section
        if "<think>" in response_section.lower():
            if self.debug_logging:
                self.logger.warning(
                    "Found <think> tags in response section after </think>"
                )
            return None

        # 2. Validate that the format appears exactly once in the response section
        if not self._validate_format_appears_exactly_once(
            response_section, answer_format
        ):
            if self.debug_logging:
                self.logger.warning(
                    f"Format {answer_format.value} does not appear exactly once in response section"
                )
            return None

        # 3. Extract based on answer_format from response_section
        extracted_content = None

        # Basic structured data formats (answer only)
        if answer_format == AnswerFormat.JSON:
            # Look for JSON object with "answer" field
            match = re.search(
                r'\{\s*"answer"\s*:\s*"([^"]*)"\s*\}', response_section, re.DOTALL
            )
            if match:
                try:
                    json.loads(match.group(0))
                    extracted_content = match.group(1)  # Extract just the answer value
                except json.JSONDecodeError:
                    pass

        elif answer_format == AnswerFormat.JSON_ARRAY:
            # Look for JSON array with one string element
            match = re.search(r'\[\s*"([^"]*)"\s*\]', response_section, re.DOTALL)
            if match:
                try:
                    json.loads(match.group(0))
                    extracted_content = match.group(1)  # Extract the array element
                except json.JSONDecodeError:
                    pass

        elif answer_format == AnswerFormat.JSON_SIMPLE:
            # Look for simple JSON string
            match = re.search(r'"([^"]*)"', response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.YAML:
            # Look for YAML answer field
            match = re.search(
                r"^answer\s*:\s*(.+?)(?:\n|$)", response_section, re.MULTILINE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.YAML_LIST:
            # Look for YAML list item
            match = re.search(r"^-\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.TOML:
            # Look for TOML answer field
            match = re.search(
                r'^answer\s*=\s*"([^"]*)"', response_section, re.MULTILINE
            )
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.TOML_SECTION:
            # Look for TOML section with answer field
            section_match = re.search(
                r'^\[response\].*?^answer\s*=\s*"([^"]*)"',
                response_section,
                re.MULTILINE | re.DOTALL,
            )
            if section_match:
                extracted_content = section_match.group(1)

        # Structured data with confidence scores
        elif answer_format == AnswerFormat.JSON_CONFIDENCE:
            # Look for JSON with answer and confidence fields
            match = re.search(
                r'\{\s*"answer"\s*:\s*"([^"]*)"\s*,\s*"confidence"\s*:\s*[0-9.]+\s*\}',
                response_section,
                re.DOTALL,
            )
            if match:
                try:
                    json.loads(match.group(0))
                    extracted_content = match.group(1)  # Extract just the answer value
                except json.JSONDecodeError:
                    pass

        elif answer_format == AnswerFormat.YAML_CONFIDENCE:
            # Look for YAML with answer and confidence fields
            answer_match = re.search(
                r"^answer\s*:\s*(.+?)(?:\n|$)", response_section, re.MULTILINE
            )
            confidence_match = re.search(
                r"^confidence\s*:\s*[0-9.]+", response_section, re.MULTILINE
            )
            if answer_match and confidence_match:
                extracted_content = answer_match.group(1).strip()

        elif answer_format == AnswerFormat.TOML_CONFIDENCE:
            # Look for TOML with answer and confidence fields
            answer_match = re.search(
                r'^answer\s*=\s*"([^"]*)"', response_section, re.MULTILINE
            )
            confidence_match = re.search(
                r"^confidence\s*=\s*[0-9.]+", response_section, re.MULTILINE
            )
            if answer_match and confidence_match:
                extracted_content = answer_match.group(1)

        # XML/HTML tag variations (XML now uses answer tags)
        elif answer_format == AnswerFormat.XML:
            match = re.search(
                r"<answer>(.*?)</answer>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.XML_FINAL_ANSWER:
            match = re.search(
                r"<answer>Final Answer:\s*(.*?)</answer>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.OUTPUT_TAGS:
            match = re.search(
                r"<output>(.*?)</output>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.RESULT_TAGS:
            match = re.search(
                r"<result>(.*?)</result>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.RESPONSE_TAGS:
            match = re.search(
                r"<response>(.*?)</response>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.FINAL_ANSWER_TAGS:
            match = re.search(
                r"<final_answer>(.*?)</final_answer>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.SOLUTION_TAGS:
            match = re.search(
                r"<solution>(.*?)</solution>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.CONCLUSION_TAGS:
            match = re.search(
                r"<conclusion>(.*?)</conclusion>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.REPLY_TAGS:
            match = re.search(
                r"<reply>(.*?)</reply>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NESTED_RESPONSE_ANSWER:
            match = re.search(
                r"<response>.*?<answer>(.*?)</answer></response>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NESTED_SOLUTION_ANSWER:
            match = re.search(
                r"<solution>.*?<answer>(.*?)</answer></solution>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NESTED_OUTPUT_RESULT:
            match = re.search(
                r"<output>.*?<result>(.*?)</result></output>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NESTED_ANALYSIS_CONCLUSION:
            match = re.search(
                r"<analysis>.*?<conclusion>(.*?)</conclusion></analysis>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NESTED_REASONING_ANSWER:
            match = re.search(
                r"<reasoning>.*?<answer>(.*?)</answer></reasoning>",
                response_section,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        # LaTeX formats (text-friendly)
        elif answer_format == AnswerFormat.LATEX_BOXED:
            match = re.search(r"\\boxed\{([^}]+)\}", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.LATEX_TEXTBF:
            match = re.search(r"\\textbf\{([^}]+)\}", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.LATEX_TEXTIT:
            match = re.search(r"\\textit\{([^}]+)\}", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.LATEX_UNDERLINE:
            match = re.search(r"\\underline\{([^}]+)\}", response_section)
            if match:
                extracted_content = match.group(1)

        # LaTeX formats (math-only)
        elif answer_format == AnswerFormat.LATEX_BOXED_MATH:
            match = re.search(r"\$\\boxed\{([^}]+)\}\$", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.LATEX_ALIGN:
            match = re.search(
                r"\\begin\{align\}(.*?)\\end\{align\}", response_section, re.DOTALL
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_EQUATION:
            match = re.search(
                r"\\begin\{equation\}(.*?)\\end\{equation\}",
                response_section,
                re.DOTALL,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_DISPLAYMATH:
            match = re.search(r"\\\\?\[(.*?)\\\\?\]", response_section, re.DOTALL)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_INLINE_MATH:
            match = re.search(r"\$([^$]+)\$", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_TEXT_MATH:
            match = re.search(r"\$\\text\{([^}]+)\}\$", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_MATHRM:
            match = re.search(r"\$\\mathrm\{([^}]+)\}\$", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_THEREFORE:
            match = re.search(r"\$\\therefore\s*(.+?)\$", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_IMPLIES:
            match = re.search(r"\$\\implies\s*(.+?)\$", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_EQUIV:
            match = re.search(r"\$(.+?)\\equiv\s*(.+?)\$", response_section)
            if match:
                # Extract the part after the equiv symbol
                extracted_content = match.group(2).strip()

        elif answer_format == AnswerFormat.LATEX_MATRIX:
            match = re.search(
                r"\$\\begin\{matrix\}(.*?)\\end\{matrix\}\$",
                response_section,
                re.DOTALL,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.LATEX_PMATRIX:
            match = re.search(
                r"\$\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}\$",
                response_section,
                re.DOTALL,
            )
            if match:
                extracted_content = match.group(1).strip()

        # Markdown formats
        elif answer_format == AnswerFormat.MARKDOWN_CODE:
            match = re.search(r"```\s*\n(.*?)\n```", response_section, re.DOTALL)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.MARKDOWN_BOLD:
            match = re.search(r"\*\*([^*]+)\*\*", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.MARKDOWN_ITALIC:
            match = re.search(r"\*([^*]+)\*", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.MARKDOWN_HEADER:
            match = re.search(r"^##\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.MARKDOWN_QUOTE:
            match = re.search(r"^>\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        # Bracket and delimiter formats
        elif answer_format == AnswerFormat.SQUARE_BRACKETS:
            match = re.search(r"\[([^\]]+)\]", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.DOUBLE_SQUARE_BRACKETS:
            match = re.search(r"\[\[([^\]]+)\]\]", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.CURLY_BRACES:
            match = re.search(r"\{([^}]+)\}", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.PARENTHESES:
            match = re.search(r"\(([^)]+)\)", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.ANGLE_BRACKETS:
            match = re.search(r"<([^>]+)>", response_section)
            if match:
                extracted_content = match.group(1)

        # Natural language patterns
        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_ANSWER:
            match = re.search(
                r"The answer is:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_FINAL:
            match = re.search(
                r"Final answer:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_CONCLUSION:
            match = re.search(
                r"In conclusion:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_THEREFORE:
            match = re.search(
                r"Therefore:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_RESULT:
            match = re.search(
                r"The result is:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        # Additional natural language patterns
        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_BEST:
            match = re.search(
                r"The best answer is:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_MY_FINAL:
            match = re.search(
                r"My final answer is:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_CORRECT:
            match = re.search(
                r"The correct answer is:?\s*(.+?)(?:\n|$)",
                response_section,
                re.IGNORECASE,
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_SOLUTION:
            match = re.search(
                r"The solution is:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_RESPONSE:
            match = re.search(
                r"My response is:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_ULTIMATELY:
            match = re.search(
                r"Ultimately:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_THUS:
            match = re.search(
                r"Thus:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_HENCE:
            match = re.search(
                r"Hence:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_CONSEQUENTLY:
            match = re.search(
                r"Consequently:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_TO_SUMMARIZE:
            match = re.search(
                r"To summarize:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_IN_SUMMARY:
            match = re.search(
                r"In summary:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_OVERALL:
            match = re.search(
                r"Overall:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_FINAL_VERDICT:
            match = re.search(
                r"Final verdict:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_BOTTOM_LINE:
            match = re.search(
                r"Bottom line:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.NATURAL_LANGUAGE_KEY_POINT:
            match = re.search(
                r"The key point is:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        # Special formats
        elif answer_format == AnswerFormat.TEXTARENA_FORMAT:
            match = re.search(r"\[([A-Da-d])\]", response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.COLON_FORMAT:
            match = re.search(
                r"Answer:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.ARROW_FORMAT:
            patterns = [
                r"=>\s*(.+?)(?:\n|$)",
                r"->\s*(.+?)(?:\n|$)",
                r"→\s*(.+?)(?:\n|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, response_section)
                if match:
                    extracted_content = match.group(1).strip()
                    break

        # HTML formats
        elif answer_format == AnswerFormat.HTML_CODE:
            match = re.search(
                r"<code>(.*?)</code>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.HTML_PRE:
            match = re.search(
                r"<pre>(.*?)</pre>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.HTML_SPAN:
            match = re.search(
                r"<span>(.*?)</span>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.HTML_DIV:
            match = re.search(
                r"<div>(.*?)</div>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.HTML_P:
            match = re.search(
                r"<p>(.*?)</p>", response_section, re.DOTALL | re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        # Multiple structured tags
        elif answer_format == AnswerFormat.MULTIPLE_TAGS:
            tags_found = {}
            tag_patterns = [
                ("theory", r"<theory>(.*?)</theory>"),
                ("answer", r"<answer>(.*?)</answer>"),
                ("explanation", r"<explanation>(.*?)</explanation>"),
                ("reasoning", r"<reasoning>(.*?)</reasoning>"),
                ("conclusion", r"<conclusion>(.*?)</conclusion>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) >= 2:  # At least 2 tags found
                extracted_content = json.dumps(tags_found)

        # Complex multi-tag formats
        elif answer_format == AnswerFormat.COMPLEX_CODING_FORMAT:
            tags_found = {}
            tag_patterns = [
                ("restatement", r"<RESTATEMENT>(.*?)</RESTATEMENT>"),
                ("reasoning", r"<REASONING>(.*?)</REASONING>"),
                ("plan", r"<PLAN>(.*?)</PLAN>"),
                ("schemas", r"<PYDANTIC_SCHEMAS>(.*?)</PYDANTIC_SCHEMAS>"),
                ("diagram", r"<DIAGRAM>(.*?)</DIAGRAM>"),
                ("reflection", r"<REFLECTION>(.*?)</REFLECTION>"),
                ("solution", r"<SOLUTION>(.*?)</SOLUTION>"),
                ("explanation", r"<EXPLANATION>(.*?)</EXPLANATION>"),
                ("unit_test", r"<UNIT_TEST>(.*?)</UNIT_TEST>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 9:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_CODING_SIMPLE:
            tags_found = {}
            tag_patterns = [
                ("restatement", r"<RESTATEMENT>(.*?)</RESTATEMENT>"),
                ("reasoning", r"<REASONING>(.*?)</REASONING>"),
                ("plan", r"<PLAN>(.*?)</PLAN>"),
                ("solution", r"<SOLUTION>(.*?)</SOLUTION>"),
                ("explanation", r"<EXPLANATION>(.*?)</EXPLANATION>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 5:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_CODING_MINIMAL:
            tags_found = {}
            tag_patterns = [
                ("analysis", r"<ANALYSIS>(.*?)</ANALYSIS>"),
                ("approach", r"<APPROACH>(.*?)</APPROACH>"),
                ("solution", r"<SOLUTION>(.*?)</SOLUTION>"),
                ("test", r"<TEST>(.*?)</TEST>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 4:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_MATH_FORMAT:
            tags_found = {}
            tag_patterns = [
                ("restatement", r"<RESTATEMENT>(.*?)</RESTATEMENT>"),
                ("reasoning", r"<REASONING>(.*?)</REASONING>"),
                ("approach", r"<APPROACH>(.*?)</APPROACH>"),
                ("derivation", r"<DERIVATION>(.*?)</DERIVATION>"),
                ("solution", r"<SOLUTION>(.*?)</SOLUTION>"),
                ("verification", r"<VERIFICATION>(.*?)</VERIFICATION>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 6:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_MATH_SIMPLE:
            tags_found = {}
            tag_patterns = [
                ("problem_analysis", r"<PROBLEM_ANALYSIS>(.*?)</PROBLEM_ANALYSIS>"),
                ("solution_steps", r"<SOLUTION_STEPS>(.*?)</SOLUTION_STEPS>"),
                ("final_answer", r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 3:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_GENERAL_FORMAT:
            tags_found = {}
            tag_patterns = [
                ("restatement", r"<RESTATEMENT>(.*?)</RESTATEMENT>"),
                ("analysis", r"<ANALYSIS>(.*?)</ANALYSIS>"),
                ("reasoning", r"<REASONING>(.*?)</REASONING>"),
                ("conclusion", r"<CONCLUSION>(.*?)</CONCLUSION>"),
                ("reflection", r"<REFLECTION>(.*?)</REFLECTION>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 5:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_GENERAL_SIMPLE:
            tags_found = {}
            tag_patterns = [
                ("analysis", r"<ANALYSIS>(.*?)</ANALYSIS>"),
                ("reasoning", r"<REASONING>(.*?)</REASONING>"),
                ("conclusion", r"<CONCLUSION>(.*?)</CONCLUSION>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 3:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_RESEARCH_FORMAT:
            tags_found = {}
            tag_patterns = [
                ("hypothesis", r"<HYPOTHESIS>(.*?)</HYPOTHESIS>"),
                ("methodology", r"<METHODOLOGY>(.*?)</METHODOLOGY>"),
                ("analysis", r"<ANALYSIS>(.*?)</ANALYSIS>"),
                ("findings", r"<FINDINGS>(.*?)</FINDINGS>"),
                ("conclusion", r"<CONCLUSION>(.*?)</CONCLUSION>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 5:  # All required tags found
                extracted_content = json.dumps(tags_found)

        # Advanced scratchpad and classification formats
        elif answer_format == AnswerFormat.COMPLEX_SCRATCHPAD_FULL:
            tags_found = {}
            tag_patterns = [
                ("scratchpad", r"<SCRATCHPAD>(.*?)</SCRATCHPAD>"),
                ("restatement", r"<RESTATEMENT>(.*?)</RESTATEMENT>"),
                ("reasoning", r"<REASONING>(.*?)</REASONING>"),
                ("plan", r"<PLAN>(.*?)</PLAN>"),
                ("citations", r"<CITATIONS>(.*?)</CITATIONS>"),
                ("schemas", r"<PYDANTIC_SCHEMAS>(.*?)</PYDANTIC_SCHEMAS>"),
                ("diagram", r"<DIAGRAM>(.*?)</DIAGRAM>"),
                ("reflection", r"<REFLECTION>(.*?)</REFLECTION>"),
                ("revised_plan", r"<REVISED_PLAN>(.*?)</REVISED_PLAN>"),
                ("solution", r"<SOLUTION>(.*?)</SOLUTION>"),
                ("explanation", r"<EXPLANATION>(.*?)</EXPLANATION>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 11:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_SCRATCHPAD_SIMPLE:
            tags_found = {}
            tag_patterns = [
                ("scratchpad", r"<SCRATCHPAD>(.*?)</SCRATCHPAD>"),
                ("restatement", r"<RESTATEMENT>(.*?)</RESTATEMENT>"),
                ("reasoning", r"<REASONING>(.*?)</REASONING>"),
                ("plan", r"<PLAN>(.*?)</PLAN>"),
                ("diagram", r"<DIAGRAM>(.*?)</DIAGRAM>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 5:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_CLASSIFICATION_FORMAT:
            tags_found = {}
            tag_patterns = [
                (
                    "completeness_explanation",
                    r"<COMPLETENESS_EXPLANATION>(.*?)</COMPLETENESS_EXPLANATION>",
                ),
                (
                    "completeness_classification",
                    r"<COMPLETENESS_CLASSIFICATION>(.*?)</COMPLETENESS_CLASSIFICATION>",
                ),
                (
                    "grammar_explanation",
                    r"<GRAMMAR_EXPLANATION>(.*?)</GRAMMAR_EXPLANATION>",
                ),
                (
                    "grammar_classification",
                    r"<GRAMMAR_CLASSIFICATION>(.*?)</GRAMMAR_CLASSIFICATION>",
                ),
                (
                    "scientific_explanation",
                    r"<SCIENTIFIC_EXPLANATION>(.*?)</SCIENTIFIC_EXPLANATION>",
                ),
                (
                    "scientific_classification",
                    r"<SCIENTIFIC_CLASSIFICATION>(.*?)</SCIENTIFIC_CLASSIFICATION>",
                ),
                (
                    "programming_explanation",
                    r"<PROGRAMMING_EXPLANATION>(.*?)</PROGRAMMING_EXPLANATION>",
                ),
                (
                    "programming_classification",
                    r"<PROGRAMMING_CLASSIFICATION>(.*?)</PROGRAMMING_CLASSIFICATION>",
                ),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 8:  # All required tags found
                extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_ANALYSIS_WITH_ANSWER:
            tags_found = {}
            tag_patterns = [
                ("analysis", r"<ANALYSIS>(.*?)</ANALYSIS>"),
                ("methodology", r"<METHODOLOGY>(.*?)</METHODOLOGY>"),
                ("findings", r"<FINDINGS>(.*?)</FINDINGS>"),
                ("answer", r"<ANSWER>(.*?)</ANSWER>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 4:  # All required tags found
                # Extract the actual answer from the ANSWER tag
                answer_content = tags_found.get("answer", "")
                answer_match = re.search(
                    r"ANSWER IS:\s*(.+)", answer_content, re.IGNORECASE
                )
                if answer_match:
                    extracted_content = answer_match.group(1).strip()
                else:
                    extracted_content = json.dumps(tags_found)

        elif answer_format == AnswerFormat.COMPLEX_EVALUATION_FORMAT:
            tags_found = {}
            tag_patterns = [
                ("criteria", r"<CRITERIA>(.*?)</CRITERIA>"),
                ("assessment", r"<ASSESSMENT>(.*?)</ASSESSMENT>"),
                ("overall_score", r"<OVERALL_SCORE>(.*?)</OVERALL_SCORE>"),
                ("judgment", r"<JUDGMENT>(.*?)</JUDGMENT>"),
            ]
            for tag_name, pattern in tag_patterns:
                match = re.search(pattern, response_section, re.DOTALL | re.IGNORECASE)
                if match:
                    tags_found[tag_name] = match.group(1).strip()

            if len(tags_found) == 4:  # All required tags found
                extracted_content = json.dumps(tags_found)

        # Dynamic formats are commented out - no extraction logic needed

        # Custom delimiters
        elif answer_format == AnswerFormat.PIPE_DELIMITED:
            match = re.search(r"\|([^|]+)\|", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.HASH_DELIMITED:
            match = re.search(r"#([^#]+)#", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.UNDERSCORE_DELIMITED:
            match = re.search(r"_([^_]+)_", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.TILDE_DELIMITED:
            match = re.search(r"~([^~]+)~", response_section)
            if match:
                extracted_content = match.group(1).strip()

        # Programming-style formats
        elif answer_format == AnswerFormat.FUNCTION_CALL:
            patterns = [
                r'answer\("([^"]+)"\)',
                r"answer\(\'([^\']+)\'\)",
                r"answer\(([^)]+)\)",
            ]
            for pattern in patterns:
                match = re.search(pattern, response_section)
                if match:
                    extracted_content = match.group(1).strip()
                    break

        elif answer_format == AnswerFormat.VARIABLE_ASSIGNMENT:
            patterns = [
                r'answer\s*=\s*"([^"]+)"',
                r"answer\s*=\s*\'([^\']+)\'",
                r"answer\s*=\s*([^;\n]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, response_section)
                if match:
                    extracted_content = match.group(1).strip()
                    break

        elif answer_format == AnswerFormat.RETURN_STATEMENT:
            patterns = [
                r'return\s+"([^"]+)"',
                r"return\s+\'([^\']+)\'",
                r"return\s+([^;\n]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, response_section)
                if match:
                    extracted_content = match.group(1).strip()
                    break

        # Additional easy-to-parse formats
        elif answer_format == AnswerFormat.EQUALS_FORMAT:
            match = re.search(r"^=\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.DASH_FORMAT:
            match = re.search(r"^-\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.PLUS_FORMAT:
            match = re.search(r"^\+\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.STAR_FORMAT:
            match = re.search(r"^\*\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.PERCENT_FORMAT:
            match = re.search(r"^%\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.AMPERSAND_FORMAT:
            match = re.search(r"^&\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.AT_FORMAT:
            match = re.search(r"^@\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.EXCLAMATION_FORMAT:
            match = re.search(r"^!\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.QUESTION_FORMAT:
            match = re.search(r"^\?\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.SEMICOLON_FORMAT:
            match = re.search(r"^;\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.DOUBLE_COLON_FORMAT:
            match = re.search(r"^::\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.TRIPLE_DASH_FORMAT:
            match = re.search(r"^---\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.DOUBLE_ARROW_FORMAT:
            match = re.search(r"^>>\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.TRIPLE_ARROW_FORMAT:
            match = re.search(r"^>>>\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.BACKTICK_FORMAT:
            match = re.search(r"`([^`]+)`", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.DOUBLE_BACKTICK_FORMAT:
            match = re.search(r"``([^`]+)``", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.QUOTE_FORMAT:
            match = re.search(r'"([^"]+)"', response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.SINGLE_QUOTE_FORMAT:
            match = re.search(r"'([^']+)'", response_section)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.TRIPLE_QUOTE_FORMAT:
            match = re.search(r'"""([^"]+)"""', response_section, re.DOTALL)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.ANSWER_IS_FORMAT:
            match = re.search(
                r"ANSWER IS:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.SOLUTION_IS_FORMAT:
            match = re.search(
                r"SOLUTION IS:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.RESULT_IS_FORMAT:
            match = re.search(
                r"RESULT IS:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.OUTPUT_IS_FORMAT:
            match = re.search(
                r"OUTPUT IS:?\s*(.+?)(?:\n|$)", response_section, re.IGNORECASE
            )
            if match:
                extracted_content = match.group(1).strip()

        # Code-specific formats
        elif answer_format == AnswerFormat.PYTHON_PRINT:
            match = re.search(r'print\s*\(\s*"([^"]*)"\s*\)', response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.JAVASCRIPT_CONSOLE:
            match = re.search(r'console\.log\s*\(\s*"([^"]*)"\s*\)', response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.PYTHON_COMMENT:
            match = re.search(r"^#\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.JAVASCRIPT_COMMENT:
            match = re.search(r"^//\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.C_COMMENT:
            match = re.search(r"/\*(.*?)\*/", response_section, re.DOTALL)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.SHELL_ECHO:
            match = re.search(r'echo\s+"([^"]*)"', response_section)
            if match:
                extracted_content = match.group(1)

        elif answer_format == AnswerFormat.SHELL_OUTPUT:
            match = re.search(r"^\$\s*(.+?)(?:\n|$)", response_section, re.MULTILINE)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.PYTHON_DOCSTRING:
            match = re.search(r'"""([^"]*)"""', response_section, re.DOTALL)
            if match:
                extracted_content = match.group(1).strip()

        elif answer_format == AnswerFormat.INI_FORMAT:
            # Look for section and key=value pattern
            section_match = re.search(r"^\[.*?\]", response_section, re.MULTILINE)
            value_match = re.search(
                r"^.*?\s*=\s*(.+?)(?:\n|$)", response_section, re.MULTILINE
            )
            if section_match and value_match:
                extracted_content = value_match.group(1).strip()

        elif answer_format == AnswerFormat.ENV_FORMAT:
            match = re.search(
                r'^[A-Z_]+\s*=\s*"([^"]*)"', response_section, re.MULTILINE
            )
            if match:
                extracted_content = match.group(1)

        if extracted_content is not None:
            if self.debug_logging:
                self.logger.debug(
                    f"Successfully extracted {answer_format.value} content: {extracted_content[:100]}{'...' if len(extracted_content) > 100 else ''}"  # noqa
                )
            return extracted_content
        else:
            if self.debug_logging:
                self.logger.warning(
                    f"Failed to extract {answer_format.value} format from response"
                )
            return None

    @classmethod
    def config_init(cls) -> Tuple[AnswerFormatEnvConfig, List[APIServerConfig]]:
        """Initialize configuration for the environment."""
        env_config = AnswerFormatEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=16,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=250,
            batch_size=1024,
            steps_per_eval=20,
            max_token_length=1024 * 12,
            inference_weight=1.0,
            wandb_name="answer_format_adherence",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            debug_logging=True,
            dump_rollouts=False,
            dump_failed_rollouts=False,
            rollout_save_score_threshold=0.0,
            eval_set_percentage=0.1,
            ensure_equivalent_ratios=False,
            format_group_threshold=10,
            suppress_base_env_logs=True,
            ensure_scores_are_not_same=False,
            dataset_configs=[
                {
                    "name": "NousResearch/AcademicMCQA",
                    "split": "train",
                    "sample_size": 50000,
                    "prompt_field": "prompt",
                    "answer_field": "ground_truth",
                    "metadata_fields": ["answer", "options"],
                    "dataset_type": "generic",
                },
                {
                    "name": "gsm8k",
                    "split": "train",
                    "sample_size": 20000,
                    "prompt_field": "question",
                    "answer_field": "answer",
                    "metadata_fields": [],
                    "dataset_type": "math_only",
                },
                # Example of how to add math dataset:
                # {
                #     "name": "hendrycks/competition_math",
                #     "split": "train",
                #     "sample_size": 200,
                #     "prompt_field": "problem",
                #     "answer_field": "solution",
                #     "metadata_fields": ["level", "type"],
                #     "dataset_type": "math_only"
                # },
                # Example of how to add code dataset:
                # {
                #     "name": "openai/humaneval",
                #     "split": "test",
                #     "sample_size": 100,
                #     "prompt_field": "prompt",
                #     "answer_field": "canonical_solution",
                #     "metadata_fields": ["task_id"],
                #     "dataset_type": "code_only"
                # }
            ],
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_max_requests_at_once=32,
                num_requests_for_eval=256,
            ),
        ]
        return env_config, server_configs

    def _extract_prompt_and_answer_from_conversations(
        self, conversations: List[Dict[str, str]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract prompt and answer from conversation format."""
        if not conversations or not isinstance(conversations, list):
            return None, None

        # Find human and assistant messages
        human_msg = None
        assistant_msg = None

        for msg in conversations:
            if isinstance(msg, dict):
                role = msg.get("from", msg.get("role", ""))
                content = msg.get("value", msg.get("content", ""))

                if role in ["human", "user"] and not human_msg:
                    human_msg = content
                elif role in ["gpt", "assistant"] and not assistant_msg:
                    assistant_msg = content

        return human_msg, assistant_msg

    async def setup(self):
        """Load and combine datasets."""
        if self.debug_logging:
            self.logger.info("Starting environment setup")

        all_datasets = []

        for dataset_config in self.config.dataset_configs:
            try:
                dataset_name = dataset_config["name"]
                split = dataset_config.get("split", "train")
                sample_size = dataset_config.get("sample_size", None)

                if self.debug_logging:
                    self.logger.info(f"Loading dataset: {dataset_name}, split: {split}")

                # Load dataset with error handling
                try:
                    if dataset_name == "gsm8k":
                        # Special handling for GSM8K dataset
                        dataset = load_dataset("gsm8k", "main", split=split)
                    else:
                        dataset = load_dataset(dataset_name, split=split)
                except Exception as dataset_load_error:
                    if self.debug_logging:
                        self.logger.error(
                            f"Failed to load dataset {dataset_name}: {dataset_load_error}"
                        )
                    continue  # Skip this dataset and try the next one

                if sample_size and len(dataset) > sample_size:
                    dataset = dataset.shuffle(seed=self.seed).select(range(sample_size))

                # Convert to our standard format
                processed_items = []
                for item in dataset:
                    prompt = None
                    answer = None
                    metadata = {}

                    # Extract prompt and answer based on field configuration
                    prompt_field = dataset_config.get("prompt_field", "prompt")
                    answer_field = dataset_config.get("answer_field", "answer")

                    if (
                        prompt_field == "conversations"
                        and answer_field == "conversations"
                    ):
                        # Handle conversation format
                        conversations = item.get("conversations", [])
                        prompt, answer = (
                            self._extract_prompt_and_answer_from_conversations(
                                conversations
                            )
                        )
                    elif dataset_name == "gsm8k":
                        # Special handling for GSM8K dataset
                        prompt = item.get(prompt_field)
                        raw_answer = item.get(answer_field)
                        # Extract the final numerical answer from GSM8K format
                        if raw_answer and "####" in raw_answer:
                            # GSM8K uses #### to separate explanation from final answer
                            answer = (
                                raw_answer.split("####")[-1].strip().replace(",", "")
                            )
                        elif raw_answer and "#" in raw_answer:
                            # Fallback for other # patterns
                            answer = raw_answer.split("#")[-1].strip().replace(",", "")
                        else:
                            answer = raw_answer
                    elif dataset_name == "NousResearch/AcademicMCQA":
                        # Special handling for AcademicMCQA dataset
                        prompt = item.get(prompt_field)  # "prompt" field
                        correct_answer_index = item.get(
                            "answer"
                        )  # Index of correct answer
                        ground_truth_letter = item.get(
                            "ground_truth"
                        )  # Letter (A, B, C, D)
                        options = item.get("options", [])  # List of answer options

                        # Use the ground truth letter as the answer for format training
                        # The format environment will train on generating the letter in various formats
                        answer = ground_truth_letter

                        # Store additional metadata for MCQA
                        metadata["correct_answer_index"] = correct_answer_index
                        metadata["ground_truth_letter"] = ground_truth_letter
                        metadata["options"] = options
                        if (
                            correct_answer_index is not None
                            and correct_answer_index < len(options)
                        ):
                            metadata["correct_answer_text"] = options[
                                correct_answer_index
                            ]
                    else:
                        # Handle direct field access
                        prompt = item.get(prompt_field)
                        answer = item.get(answer_field)

                    # Extract metadata
                    metadata_fields = dataset_config.get("metadata_fields", [])
                    for field in metadata_fields:
                        if field in item:
                            metadata[field] = item[field]

                    metadata["dataset_name"] = dataset_name

                    # Get dataset type (default to "generic" if not specified)
                    dataset_type = dataset_config.get("dataset_type", "generic")

                    if prompt and answer:
                        processed_items.append(
                            {
                                "prompt": prompt,
                                "answer": answer,
                                "metadata": metadata,
                                "dataset_type": dataset_type,
                            }
                        )

                if self.debug_logging:
                    self.logger.info(
                        f"Processed {len(processed_items)} items from {dataset_name}"
                    )

                all_datasets.extend(processed_items)

            except Exception as e:
                error_msg = f"Error loading dataset {dataset_config.get('name', 'unknown')}: {e}"
                if self.debug_logging:
                    self.logger.error(error_msg)
                print(error_msg)

        if not all_datasets:
            raise ValueError("No datasets could be loaded successfully")

        # Shuffle all items together
        random.shuffle(all_datasets)
        self.dataset_items = all_datasets

        # Split into train and test
        split_idx = int(
            len(self.dataset_items) * (1.0 - self.config.eval_set_percentage)
        )
        self.train_items = self.dataset_items[:split_idx]
        self.test_items = self.dataset_items[split_idx:]

        self.iter = 0

        if self.debug_logging:
            # Log dataset type distribution
            type_counts = {}
            for item in self.dataset_items:
                dataset_type = item.get("dataset_type", "generic")
                type_counts[dataset_type] = type_counts.get(dataset_type, 0) + 1

            self.logger.info(
                f"Setup complete: {len(self.train_items)} training items, {len(self.test_items)} test items"
            )
            self.logger.info(f"Dataset type distribution: {type_counts}")

            # Log format availability by type
            for dataset_type in type_counts.keys():
                available_formats = self._get_formats_for_dataset_type(dataset_type)
                self.logger.info(
                    f"Dataset type '{dataset_type}': {len(available_formats)} available formats"
                )

        print(
            f"AnswerFormatEnv setup complete. {len(self.train_items)} training items, {len(self.test_items)} test items."  # noqa
        )

        # Print dataset type distribution
        type_counts = {}
        for item in self.dataset_items:
            dataset_type = item.get("dataset_type", "generic")
            type_counts[dataset_type] = type_counts.get(dataset_type, 0) + 1
        print(f"Dataset types: {type_counts}")

    async def get_next_item(self) -> Tuple[Tuple[frozenset, ...], Dict[str, Any]]:
        """Get the next training item with randomized format based on dataset type."""
        if not self.train_items:
            raise ValueError("No training items available")

        # Dynamic formats are commented out - no state management needed

        # Get the next item cyclically
        dataset_item = self.train_items[self.iter % len(self.train_items)]
        self.iter += 1

        # Get appropriate formats for this dataset type
        dataset_type = dataset_item.get("dataset_type", "generic")
        available_formats = self._get_formats_for_dataset_type(dataset_type)

        if not available_formats:
            # Fallback to generic formats if no formats available
            available_formats = list(self.generic_formats)
            if self.debug_logging:
                self.logger.warning(
                    f"No formats available for dataset type '{dataset_type}', using generic formats"
                )

        # Randomly select an answer format from available formats with optional weighting
        # Simple formats get higher weight for better training balance
        if len(available_formats) > 10:  # Only apply weighting if we have many formats
            simple_formats = [
                f
                for f in available_formats
                if not f.value.startswith(("complex_", "dynamic_"))
            ]
            complex_formats = [
                f
                for f in available_formats
                if f.value.startswith(("complex_", "dynamic_"))
            ]

            # 70% chance for simple formats, 30% for complex formats
            if random.random() < 0.7 and simple_formats:
                selected_format = random.choice(simple_formats)
            elif complex_formats:
                selected_format = random.choice(complex_formats)
            else:
                selected_format = random.choice(available_formats)
        else:
            selected_format = random.choice(available_formats)

        if self.debug_logging:
            self.logger.debug(
                f"Item {self.iter-1}: dataset_type='{dataset_type}', "
                f"selected_format='{selected_format.value}' from {len(available_formats)} available"
            )

        # Store the selected format in the item for scoring
        dataset_item["selected_format"] = selected_format

        # Dynamic formats are commented out - no component generation needed

        # Generate system prompt for the selected format
        system_prompt = self._generate_system_prompt(selected_format)

        # Create the message structure
        prompt_messages = [
            frozenset({"role": "system", "content": system_prompt}.items()),
            frozenset({"role": "user", "content": dataset_item["prompt"]}.items()),
        ]

        return tuple(prompt_messages), dataset_item

    async def score(
        self,
        rollout_group_data: List[Tuple[Tuple[Dict[str, str], ...], Dict[str, Any]]],
    ) -> Optional[ScoredDataGroup]:
        """Score rollouts based on format adherence."""
        if self.debug_logging:
            self.logger.debug(f"Scoring {len(rollout_group_data)} rollouts")

        scores_obj = ScoredDataGroup()
        scores_obj["tokens"] = list()
        scores_obj["masks"] = list()
        scores_obj["scores"] = list()

        if not rollout_group_data:
            return None

        dataset_item = rollout_group_data[0][1]
        selected_format = dataset_item["selected_format"]
        format_name = selected_format.value

        if self.debug_logging:
            self.logger.debug(f"Scoring for format: {format_name}")

        random.shuffle(rollout_group_data)
        failed_rollouts_this_group = []

        for item_messages, _ in rollout_group_data:
            messages_as_dicts = [dict(fs_message) for fs_message in item_messages]
            model_response_text = messages_as_dicts[-1]["content"]

            # Extract content based on the selected format
            extracted_content = self._extract_answer_content(
                model_response_text, selected_format, dataset_item
            )

            # Score: 1.0 if format is correct, 0.0 if not
            reward = 1.0 if extracted_content is not None else 0.0
            is_failed = reward == 0.0

            # Track format success rates
            self.format_total_counts[format_name] = (
                self.format_total_counts.get(format_name, 0) + 1
            )
            if reward == 1.0:
                self.format_success_counts[format_name] = (
                    self.format_success_counts.get(format_name, 0) + 1
                )

            try:
                # Validate message format for tokenization
                for msg_idx, msg in enumerate(messages_as_dicts):
                    if not isinstance(msg, dict):
                        if self.debug_logging:
                            self.logger.error(
                                f"Message {msg_idx} is not a dict: {type(msg)}"
                            )
                        continue
                    if "role" not in msg or "content" not in msg:
                        if self.debug_logging:
                            self.logger.error(
                                f"Message {msg_idx} missing required keys: {msg.keys()}"
                            )
                        continue
                    if not isinstance(msg["content"], str):
                        msg["content"] = str(msg["content"])

                out_dict = tokenize_for_trainer(
                    self.tokenizer,
                    messages_as_dicts,
                    include_messages=self.config.include_messages,
                )
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]

            except Exception as e:
                if self.debug_logging:
                    self.logger.error(f"Tokenization failed: {e}")
                continue

            # Remove examples with insufficient context
            if len([1 for m_val in masks if m_val != -100]) < 10:
                continue

            scores_obj["tokens"].append(tokens)
            scores_obj["masks"].append(masks)
            scores_obj["scores"].append(reward)

            # Track failed rollouts for debugging
            if is_failed and self.config.dump_failed_rollouts:
                failed_rollouts_this_group.append(
                    {
                        "conversation": messages_as_dicts,
                        "score": reward,
                        "selected_format": format_name,
                        "metadata": dataset_item.get("metadata", {}),
                        "extracted_content": extracted_content,
                        "failure_reason": "format_parsing_failed",
                    }
                )

            self.percent_correct_buffer.append(reward)

            if len(scores_obj["tokens"]) >= self.config.group_size:
                break

        if not scores_obj["tokens"]:
            if self.debug_logging:
                self.logger.warning("No valid examples processed in this batch")
            return None

        # Calculate and log group-level statistics
        group_scores = scores_obj["scores"]
        average_score = sum(group_scores) / len(group_scores) if group_scores else 0.0

        # Update group statistics
        self.group_statistics["total_groups"] += 1
        self.group_statistics["average_scores"].append(average_score)
        self.group_statistics["format_distribution"][format_name] = (
            self.group_statistics["format_distribution"].get(format_name, 0) + 1
        )

        if average_score > 0.0:
            self.group_statistics["successful_groups"] += 1
        else:
            self.group_statistics["failed_groups"] += 1

        # Log group performance with detailed information
        if self.debug_logging:
            correct_count = sum(1 for score in group_scores if score == 1.0)
            incorrect_count = len(group_scores) - correct_count
            correct_percentage = (
                (correct_count / len(group_scores)) * 100 if group_scores else 0.0
            )

            if correct_count == 0:
                self.logger.info(
                    f"Format: {format_name} | Group average score: {average_score:.4f} | {correct_count}/{len(group_scores)} correct ({correct_percentage:.1f}%) (All failures in this group!)"  # noqa
                )
            elif incorrect_count == 0:
                self.logger.info(
                    f"Format: {format_name} | Group average score: {average_score:.4f} | {correct_count}/{len(group_scores)} correct ({correct_percentage:.1f}%) (Perfect group!)"  # noqa
                )
            else:
                self.logger.info(
                    f"Format: {format_name} | Group average score: {average_score:.4f} | {correct_count}/{len(group_scores)} correct ({correct_percentage:.1f}%)"  # noqa
                )

        # Handle failed rollouts dumping
        if failed_rollouts_this_group and self.config.dump_failed_rollouts:
            failed_item_data_to_save = {
                "item_id": f"failed_group_{self.group_statistics['total_groups']}",
                "format": format_name,
                "group_average_score": average_score,
                "rollouts": failed_rollouts_this_group,
                "group_metadata": {
                    "total_rollouts": len(group_scores),
                    "failed_rollouts": len(failed_rollouts_this_group),
                    "dataset_type": dataset_item.get("dataset_type", "unknown"),
                },
            }
            self.failed_rollouts_to_save_buffer.append(failed_item_data_to_save)

            # Save failed rollouts every 50 groups
            if (
                self.config.dump_failed_rollouts
                and len(self.failed_rollouts_to_save_buffer) >= 50
            ):
                if self.debug_logging:
                    self.logger.info(
                        f"Saving batch of {len(self.failed_rollouts_to_save_buffer)} failed rollout groups"
                    )
                await self._save_failed_rollouts_to_jsonl()

        # Check if all scores are the same (no learning signal)
        if all(group_scores[0] == score for score in group_scores):
            if self.debug_logging:
                self.logger.debug(
                    "All scores are identical, returning None for learning signal"
                )
            return None

        # Track successful groups for equivalent ratio enforcement
        if self.ensure_equivalent_ratios:
            # Count this as a successful group if we have any successful examples
            if any(score > 0.0 for score in group_scores):
                self.format_successful_groups[format_name] = (
                    self.format_successful_groups.get(format_name, 0) + 1
                )

                if self.debug_logging:
                    successful_count = self.format_successful_groups[format_name]
                    remaining_until_threshold = (
                        self.format_group_threshold - successful_count
                    )

                    # Always log when a group is added, showing progress toward threshold
                    if remaining_until_threshold > 0:
                        self.logger.info(
                            f"Format {format_name} successful group #{successful_count} - {remaining_until_threshold} more needed to reach threshold ({self.format_group_threshold})"  # noqa
                        )

                    # Log when a format reaches the threshold
                    if successful_count == self.format_group_threshold:
                        self.logger.info(
                            f"Format {format_name} reached threshold of {self.format_group_threshold} successful groups - will be excluded from future rounds until others catch up"  # noqa
                        )

        # Apply length penalty if all responses are correct
        if all(s == 1.0 for s in group_scores):
            avg_len = sum(len(t) for t in scores_obj["tokens"]) / len(
                scores_obj["tokens"]
            )
            if avg_len > self.config.max_token_length * 0.75:
                scores_obj["scores"] = [s * 0.9 for s in scores_obj["scores"]]
                if self.debug_logging:
                    self.logger.debug(f"Applied length penalty: avg_len={avg_len}")

        return scores_obj

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        """Collect trajectories for a given item."""
        prompt_messages_tuple, dataset_item = item

        if self.debug_logging:
            self.logger.debug(
                f"Collecting trajectories for format: {dataset_item['selected_format'].value}"
            )

        # Convert frozensets to dicts for the API call
        messages_for_api = [dict(fs_message) for fs_message in prompt_messages_tuple]

        prompt_str = self.tokenizer.apply_chat_template(
            messages_for_api, add_generation_prompt=True, tokenize=False
        )

        completions = await self.server.completion(
            prompt=prompt_str,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=0.9,
        )

        to_score_list = []
        for choice in completions.choices:
            # Create a full message list for this choice
            current_trajectory_messages = list(prompt_messages_tuple)
            current_trajectory_messages.append(
                frozenset({"role": "assistant", "content": choice.text}.items())
            )
            to_score_list.append((tuple(current_trajectory_messages), dataset_item))

        scored_data = await self.score(to_score_list)

        # Data dumping logic - following reasoning gym pattern exactly
        if scored_data and self.config.dump_rollouts:
            # Only save groups that have at least one rollout with score > threshold
            group_scores = scored_data.get("scores", [])
            threshold = self.config.rollout_save_score_threshold
            if any(score > threshold for score in group_scores):
                if self.debug_logging:
                    self.logger.debug(
                        f"Saving group with scores: {[f'{s:.3f}' for s in group_scores]} (has high-quality rollout, threshold: {threshold})"  # noqa
                    )
                rollouts_with_scores_to_save = []

                num_scored_rollouts = len(group_scores)
                for i in range(num_scored_rollouts):
                    # Get conversation messages directly from to_score_list like reasoning gym does
                    conversation_messages = [
                        dict(fs_msg) for fs_msg in to_score_list[i][0]
                    ]
                    score_for_rollout = group_scores[i]

                    # Only save rollouts that meet the score threshold
                    if score_for_rollout >= threshold:
                        rollouts_with_scores_to_save.append(
                            {
                                "conversation": conversation_messages,
                                "score": score_for_rollout,
                            }
                        )

                if rollouts_with_scores_to_save:
                    item_data_to_save = {
                        "item_id": f"item_{self.iter-1}",
                        "format": dataset_item["selected_format"].value,
                        "rollouts": rollouts_with_scores_to_save,
                        "group_metadata": {
                            "total_rollouts": num_scored_rollouts,
                            "saved_rollouts": len(rollouts_with_scores_to_save),
                            "dataset_type": dataset_item.get("dataset_type", "unknown"),
                            "group_statistics": {
                                "min_score": min(group_scores) if group_scores else 0.0,
                                "max_score": max(group_scores) if group_scores else 0.0,
                                "avg_score": (
                                    sum(group_scores) / len(group_scores)
                                    if group_scores
                                    else 0.0
                                ),
                            },
                        },
                    }
                    self.rollouts_to_save_buffer.append(item_data_to_save)
                    self.processed_item_count += 1

                    # Calculate progress toward next save (like reasoning gym does)
                    current_batch_progress = self.processed_item_count % 100
                    if current_batch_progress == 0:
                        current_batch_progress = 100

                    # Log progress every 10 items or when we hit the save threshold
                    if (
                        current_batch_progress % 10 == 0
                        or current_batch_progress == 100
                    ):
                        if self.debug_logging:
                            self.logger.info(
                                f"Data dump progress: {current_batch_progress}/100 items buffered "
                                f"(Total processed: {self.processed_item_count}, Buffer size: {len(self.rollouts_to_save_buffer)})"  # noqa
                            )

                    # Save batch every 100 processed items
                    if (
                        self.config.dump_rollouts
                        and self.processed_item_count > 0
                        and self.processed_item_count % 100 == 0
                    ):
                        if self.debug_logging:
                            log_msg = (
                                f"Reached {self.processed_item_count} processed items. "
                                f"Triggering save for {len(self.rollouts_to_save_buffer)} rollouts"  # noqa
                            )
                            self.logger.info(log_msg)
                        await self._save_rollouts_to_jsonl()
            else:
                max_score = max(group_scores) if group_scores else 0.0
                if self.debug_logging:
                    self.logger.debug(
                        f"Skipping group save - no high-quality rollouts (max score: {max_score:.3f}, threshold: {threshold})"  # noqa
                    )

        return scored_data, []

    async def _save_rollouts_to_jsonl(self):
        """Save rollouts to JSONL file."""
        if not self.rollouts_to_save_buffer:
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
        except OSError as e:
            if self.debug_logging:
                self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"answer_format_rollouts_{self.run_uuid}_{self.save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")

            if self.debug_logging:
                self.logger.info(
                    f"Saved {len(self.rollouts_to_save_buffer)} rollouts to {file_path}"
                )

            self.rollouts_to_save_buffer.clear()
            self.save_file_batch_num += 1

        except Exception as e:
            if self.debug_logging:
                self.logger.error(f"Error saving rollouts: {e}")

    async def _save_failed_rollouts_to_jsonl(self):
        """Save failed rollouts to JSONL file for debugging."""
        if not self.failed_rollouts_to_save_buffer:
            if self.debug_logging:
                self.logger.info("No failed rollouts in buffer to save.")
            return

        try:
            if not os.path.exists(self.datadumps_dir):
                os.makedirs(self.datadumps_dir)
        except OSError as e:
            if self.debug_logging:
                self.logger.error(f"Error creating directory {self.datadumps_dir}: {e}")
            return

        file_path = os.path.join(
            self.datadumps_dir,
            f"answer_format_FAILED_rollouts_{self.run_uuid}_{self.failed_save_file_batch_num:04d}.jsonl",
        )

        try:
            with open(file_path, "w") as f:
                for rollout_dict in self.failed_rollouts_to_save_buffer:
                    json.dump(rollout_dict, f)
                    f.write("\n")

            if self.debug_logging:
                self.logger.info(
                    f"Successfully saved {len(self.failed_rollouts_to_save_buffer)} FAILED rollouts to {file_path}"
                )

            self.failed_rollouts_to_save_buffer.clear()
            self.failed_save_file_batch_num += 1

        except Exception as e:
            if self.debug_logging:
                self.logger.error(f"Error writing failed rollouts to {file_path}: {e}")
            else:
                print(
                    f"An unexpected error occurred while saving failed rollouts to {file_path}: {e}"
                )

    async def rollout_and_score_eval(self, test_item: Dict[str, Any]) -> float:
        """Evaluate a single test item."""
        # Get appropriate formats for this dataset type
        dataset_type = test_item.get("dataset_type", "generic")
        available_formats = self._get_formats_for_dataset_type(dataset_type)

        if not available_formats:
            # Fallback to generic formats if no formats available
            available_formats = list(self.generic_formats)

        # Randomly select format for evaluation
        selected_format = random.choice(available_formats)
        test_item["selected_format"] = selected_format

        system_prompt = self._generate_system_prompt(selected_format)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": test_item["prompt"]},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        completion = await self.server.completion(
            prompt=prompt,
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.1,
            split="eval",
        )

        model_response = completion.choices[0].text
        extracted_content = self._extract_answer_content(
            model_response, selected_format
        )

        return 1.0 if extracted_content is not None else 0.0

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on the test set."""
        if self.debug_logging:
            self.logger.info("Starting evaluation")

        if not self.test_items:
            self.eval_metrics.append(("eval/percent_correct", 0.0))
            return

        # Use subset for faster evaluation
        items_to_eval = self.test_items[: min(len(self.test_items), 50)]

        eval_results = await tqdm_asyncio.gather(
            *[self.rollout_and_score_eval(item) for item in items_to_eval]
        )

        if eval_results:
            avg_score = sum(eval_results) / len(eval_results)
        else:
            avg_score = 0.0

        self.eval_metrics.append(("eval/percent_correct", avg_score))

        if self.debug_logging:
            self.logger.info(f"Evaluation complete: avg_score={avg_score:.3f}")

    async def add_rollouts_for_wandb(
        self,
        scored_data: Optional[ScoredDataGroup],
        item: Item = None,
    ):
        """Add rollouts to wandb logging."""
        if scored_data is None or not scored_data["tokens"]:
            return

        dataset_item = item[1] if item and len(item) > 1 else {}
        selected_format = dataset_item.get("selected_format", AnswerFormat.JSON)

        num_keep = self.config.num_rollouts_per_group_for_logging
        if num_keep == -1:
            num_keep = len(scored_data["tokens"])
        else:
            num_keep = min(num_keep, len(scored_data["tokens"]))

        rollout_batch = []
        for i in range(num_keep):
            # Get the conversation text by decoding tokens (like reasoning gym does)
            display_convo_text = self.tokenizer.decode(
                scored_data["tokens"][i], skip_special_tokens=True
            )

            rollout_batch.append(
                (
                    display_convo_text,
                    scored_data["scores"][i],
                    selected_format.value,
                    dataset_item.get("metadata", {}).get("dataset_name", "unknown"),
                )
            )

        if rollout_batch:
            self.rollouts_for_wandb.append(rollout_batch)

        if len(self.rollouts_for_wandb) > self.config.num_rollouts_to_keep:
            self.rollouts_for_wandb.pop(0)

    async def create_rollout_table(self, wandb_metrics: Dict) -> Dict:
        """Create wandb table for rollout visualization."""
        if self.rollouts_for_wandb:
            table = wandb.Table(
                columns=[
                    "full_conversation",
                    "score",
                    "selected_format",
                    "dataset_name",
                ]
            )
            for group in self.rollouts_for_wandb:
                for entry in group:
                    table.add_data(*entry)
            wandb_metrics["train/rollouts"] = table

        self.rollouts_for_wandb = []
        return wandb_metrics

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb with clear, intuitive categories."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # === CORE PERFORMANCE METRICS ===
        if self.percent_correct_buffer:
            percent_correct = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
            wandb_metrics["performance/accuracy"] = percent_correct
        else:
            wandb_metrics["performance/accuracy"] = 0.0

        self.percent_correct_buffer = list()

        # Add eval metrics
        for key, value in self.eval_metrics:
            # Rename eval metrics to be clearer
            clean_key = key.replace("eval/percent_correct", "performance/eval_accuracy")
            wandb_metrics[clean_key] = value
        self.eval_metrics = list()

        # === GROUP STATISTICS ===
        if self.group_statistics["total_groups"] > 0:
            total = self.group_statistics["total_groups"]
            successful = self.group_statistics["successful_groups"]
            failed = self.group_statistics["failed_groups"]

            wandb_metrics["groups/total_processed"] = total
            wandb_metrics["groups/successful"] = successful
            wandb_metrics["groups/failed"] = failed
            wandb_metrics["groups/success_rate"] = successful / total

            if self.group_statistics["average_scores"]:
                all_scores = self.group_statistics["average_scores"]
                recent_scores = all_scores[
                    -50:
                ]  # Last 50 groups for recent performance

                wandb_metrics["groups/avg_score_overall"] = sum(all_scores) / len(
                    all_scores
                )
                wandb_metrics["groups/avg_score_recent"] = sum(recent_scores) / len(
                    recent_scores
                )
                wandb_metrics["groups/best_score_recent"] = max(recent_scores)
                wandb_metrics["groups/worst_score_recent"] = min(recent_scores)

        # === FORMAT PERFORMANCE SUMMARY ===
        if self.format_total_counts:
            # Calculate overall format statistics
            total_attempts = sum(self.format_total_counts.values())
            total_successes = sum(self.format_success_counts.values())

            wandb_metrics["formats/total_attempts"] = total_attempts
            wandb_metrics["formats/total_successes"] = total_successes
            wandb_metrics["formats/overall_success_rate"] = (
                total_successes / total_attempts if total_attempts > 0 else 0.0
            )

            # Count formats by performance tier
            format_success_rates = {}
            for format_name, total_count in self.format_total_counts.items():
                if total_count >= 5:  # Only consider formats with enough data
                    success_count = self.format_success_counts.get(format_name, 0)
                    success_rate = success_count / total_count
                    format_success_rates[format_name] = success_rate

            if format_success_rates:
                rates = list(format_success_rates.values())
                wandb_metrics["formats/num_formats_tested"] = len(format_success_rates)
                wandb_metrics["formats/avg_success_rate"] = sum(rates) / len(rates)
                wandb_metrics["formats/best_success_rate"] = max(rates)
                wandb_metrics["formats/worst_success_rate"] = min(rates)

                # Performance tiers
                high_performing = sum(1 for rate in rates if rate >= 0.8)
                medium_performing = sum(1 for rate in rates if 0.5 <= rate < 0.8)
                low_performing = sum(1 for rate in rates if rate < 0.5)

                wandb_metrics["formats/high_performing_count"] = high_performing
                wandb_metrics["formats/medium_performing_count"] = medium_performing
                wandb_metrics["formats/low_performing_count"] = low_performing

        # === BALANCED TRAINING PROGRESS (only if enabled) ===
        if self.ensure_equivalent_ratios and self.format_successful_groups:
            successful_counts = list(self.format_successful_groups.values())
            threshold = self.format_group_threshold

            # Overall progress
            formats_at_threshold = sum(
                1 for count in successful_counts if count >= threshold
            )
            total_formats = len(self.base_supported_formats)
            completion_pct = (formats_at_threshold / total_formats) * 100

            wandb_metrics["balance/completion_percentage"] = completion_pct
            wandb_metrics["balance/formats_completed"] = formats_at_threshold
            wandb_metrics["balance/formats_total"] = total_formats

            # Progress distribution
            wandb_metrics["balance/min_successful_groups"] = min(successful_counts)
            wandb_metrics["balance/max_successful_groups"] = max(successful_counts)
            wandb_metrics["balance/avg_successful_groups"] = sum(
                successful_counts
            ) / len(successful_counts)

            # How many formats are at different progress levels
            quarter_threshold = threshold * 0.25
            half_threshold = threshold * 0.5
            three_quarter_threshold = threshold * 0.75

            at_quarter = sum(
                1 for count in successful_counts if count >= quarter_threshold
            )
            at_half = sum(1 for count in successful_counts if count >= half_threshold)
            at_three_quarter = sum(
                1 for count in successful_counts if count >= three_quarter_threshold
            )

            wandb_metrics["balance/formats_25pct_progress"] = at_quarter
            wandb_metrics["balance/formats_50pct_progress"] = at_half
            wandb_metrics["balance/formats_75pct_progress"] = at_three_quarter

        # === DATASET DISTRIBUTION ===
        if self.group_statistics["format_distribution"]:
            total_uses = sum(self.group_statistics["format_distribution"].values())

            # Just track the most and least used formats for high-level monitoring
            sorted_formats = sorted(
                self.group_statistics["format_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            )

            if sorted_formats:
                most_used_format, most_used_count = sorted_formats[0]
                least_used_format, least_used_count = sorted_formats[-1]

                wandb_metrics["distribution/most_used_format_pct"] = (
                    most_used_count / total_uses
                ) * 100
                wandb_metrics["distribution/least_used_format_pct"] = (
                    least_used_count / total_uses
                ) * 100
                wandb_metrics["distribution/usage_ratio"] = most_used_count / max(
                    least_used_count, 1
                )

        # Create rollout table for detailed inspection
        wandb_metrics = await self.create_rollout_table(wandb_metrics)

        await super().wandb_log(wandb_metrics)

    async def close(self):
        """Clean up and save any remaining rollouts."""
        if self.debug_logging:
            self.logger.info("Closing AnswerFormatEnv")

            # Log final statistics
            if self.group_statistics["total_groups"] > 0:
                success_rate = (
                    self.group_statistics["successful_groups"]
                    / self.group_statistics["total_groups"]
                )
                avg_score = sum(self.group_statistics["average_scores"]) / len(
                    self.group_statistics["average_scores"]
                )
                self.logger.info("Final Statistics:")
                self.logger.info(
                    f"  Total groups processed: {self.group_statistics['total_groups']}"
                )
                self.logger.info(
                    f"  Successful groups: {self.group_statistics['successful_groups']}"
                )
                self.logger.info(
                    f"  Failed groups: {self.group_statistics['failed_groups']}"
                )
                self.logger.info(f"  Overall success rate: {success_rate:.4f}")
                self.logger.info(f"  Overall average score: {avg_score:.4f}")

        # Save any remaining rollouts
        if self.config.dump_rollouts and self.rollouts_to_save_buffer:
            await self._save_rollouts_to_jsonl()

        # Save any remaining failed rollouts
        if self.config.dump_failed_rollouts and self.failed_rollouts_to_save_buffer:
            if self.debug_logging:
                self.logger.info(
                    f"Found {len(self.failed_rollouts_to_save_buffer)} failed rollouts in buffer. Saving now."
                )
            await self._save_failed_rollouts_to_jsonl()
        elif self.debug_logging:
            self.logger.info("No failed rollouts in buffer to save upon closing.")

        # Call superclass close if it exists
        if hasattr(super(), "close"):
            await super().close()


if __name__ == "__main__":
    AnswerFormatEnv.cli()
