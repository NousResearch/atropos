# Answer Format Environment

A comprehensive environment for teaching language models to generate responses in specific structured formats. This environment focuses on **format adherence** rather than answer correctness, using randomized format requirements and corresponding parsers to evaluate models on structured response generation.

## ⚠️ **Important: Rejection Sampling Focused**

**This environment is primarily designed for rejection sampling, not traditional RL training.** Since we only validate format compliance and do not verify answer correctness, the binary scoring (1.0 for correct format, 0.0 for incorrect) makes it less suitable for gradient-based RL methods. Instead, it excels at:

- **Rejection Sampling**: Filter model outputs based on format compliance
- **Format Evaluation**: Assess model capabilities across different structured formats
- **Data Curation**: Generate format-compliant training data
- **Format Benchmarking**: Compare model performance on format adherence tasks

## 🎯 Overview

The Answer Format Environment evaluates models on:
- Generating responses in 150+ different structured formats
- Following strict thinking tag discipline (`<think></think>`)
- Format compliance validation and parsing
- Handling multiple dataset types with appropriate format selection
- Maintaining equivalent evaluation ratios across formats (optional)

**Key Philosophy**: This environment scores based on format compliance (1.0 for correct format, 0.0 for incorrect), not answer accuracy. It teaches models to follow formatting instructions precisely without validating the correctness of the actual answers.

## ✨ Key Features

### 🔄 **Randomized Format Selection**
- 150+ supported answer formats across multiple categories
- Weighted format selection (70% simple, 30% complex)
- Dataset type-aware format filtering (generic, math_only, code_only)
- Dynamic compositor system for complex structured responses

### 🧠 **Thinking Tag Validation**
- Enforces exactly one `<think></think>` section per response
- All reasoning must be contained within thinking tags
- Strict validation prevents multiple thinking sections
- Answer must appear after `</think>` in specified format

### 📊 **Comprehensive Data Management**
- Multi-dataset support with automatic shuffling
- Configurable train/eval splits
- Extensive data dumping with group-level statistics
- Failed rollout tracking and analysis
- WandB integration with detailed metrics

### ⚖️ **Equivalent Ratio Enforcement**
- Optional system to ensure balanced evaluation across formats
- Pauses formats after reaching success threshold
- Prevents format bias in evaluation data
- Comprehensive monitoring and status reporting

### 🔍 **Advanced Monitoring**
- Format success rate tracking
- Group-level performance statistics
- Failure case analysis and logging
- Real-time format balance monitoring

## 📋 Supported Format Categories

### **Basic Structured Data**
```json
{"answer": "content"}                    // JSON
answer: content                          // YAML
answer = "content"                       // TOML
```

### **XML/HTML Tags**
```xml
<answer>content</answer>                 // XML
<answer>Final Answer: content</answer>   // XML with prefix
<output>content</output>                 // Output tags
<result>content</result>                 // Result tags
```

### **LaTeX Formats**
```latex
\boxed{content}                          // Text-friendly boxed
$\boxed{expression}$                     // Math-only boxed
\begin{align} expression \end{align}     // Math alignment
$\text{answer}$                          // Text in math mode
```

### **Natural Language**
```
The answer is: content
Final answer: content
In conclusion: content
Therefore: content
```

### **Programming Formats**
```python
print("answer")                          // Python print
console.log("answer")                    // JavaScript console
# answer                                 // Python comment
return "answer"                          // Return statement
```

### **Complex Multi-Tag Formats**
```xml
<restatement>...</restatement>
<reasoning>...</reasoning>
<solution>...</solution>
<explanation>...</explanation>
```

### **Dynamic Compositor Formats**
Randomly combines 3-6 components in XML, JSON, YAML, or TOML:
- Analysis components (problem_analysis, requirements_analysis, etc.)
- Reasoning components (logical_reasoning, step_by_step, etc.)
- Planning components (approach, methodology, etc.)
- Technical components (implementation, code_structure, etc.)
- Evaluation components (validation, testing, etc.)
- Output components (final_answer, conclusion, etc.)

## 🚀 Quick Start

### Basic Configuration

```python
from atropos.environments.answer_format_environment import AnswerFormatEnv, AnswerFormatEnvConfig

# Simple configuration
config = AnswerFormatEnvConfig(
    dataset_configs=[
        {
            "name": "your_dataset",
            "split": "train",
            "sample_size": 1000,
            "prompt_field": "question",
            "answer_field": "answer",
            "dataset_type": "generic"
        }
    ],
    debug_logging=True,
    dump_rollouts=True,
    eval_set_percentage=0.1
)

# Initialize environment
env = AnswerFormatEnv(config, server_configs)
```

### Multi-Dataset Configuration

```python
config = AnswerFormatEnvConfig(
    dataset_configs=[
        {
            "name": "teknium/OpenHermes-2.5",
            "split": "train",
            "sample_size": 5000,
            "prompt_field": "conversations",
            "answer_field": "conversations",
            "metadata_fields": ["source"],
            "dataset_type": "generic"
        },
        {
            "name": "gsm8k",
            "split": "train",
            "sample_size": 2000,
            "prompt_field": "question",
            "answer_field": "answer",
            "dataset_type": "math_only"
        },
        {
            "name": "NousResearch/AcademicMCQA",
            "split": "train",
            "sample_size": 5000,
            "prompt_field": "prompt",
            "answer_field": "ground_truth",
            "metadata_fields": ["answer", "options"],
            "dataset_type": "generic"
        }
    ],
    ensure_equivalent_ratios=True,
    format_group_threshold=50,
    dump_failed_rollouts=True
)
```

## ⚙️ Configuration Options

### **Dataset Configuration**
```python
dataset_configs: List[Dict[str, Any]] = [
    {
        "name": str,                    # Dataset name or HuggingFace path
        "split": str,                   # Dataset split ("train", "test", etc.)
        "sample_size": int,             # Number of samples to use
        "prompt_field": str,            # Field containing prompts/questions
        "answer_field": str,            # Field containing answers
        "metadata_fields": List[str],   # Additional fields to preserve
        "dataset_type": str             # "generic", "math_only", or "code_only"
    }
]
```

### **Core Settings**
```python
debug_logging: bool = True                    # Enable detailed logging
dump_rollouts: bool = True                    # Save rollouts to JSONL
dump_failed_rollouts: bool = True             # Save failed rollouts separately
rollout_save_score_threshold: float = 0.0    # Minimum score to save rollouts
eval_set_percentage: float = 0.1              # Evaluation set percentage
```

### **Format Control**
```python
supported_formats: Optional[List[AnswerFormat]] = None  # Filter to specific formats
ensure_equivalent_ratios: bool = False                  # Enable ratio enforcement
format_group_threshold: int = 50                        # Success threshold per format
```

## 📊 Dataset Types & Format Selection

### **Generic Datasets** (`dataset_type: "generic"`)
- **Available Formats**: All basic formats (JSON, XML, natural language, brackets, etc.)
- **Use Case**: General conversation, QA, instruction following, MCQA
- **Examples**: OpenHermes, Alpaca, AcademicMCQA, general chat datasets

### **Math-Only Datasets** (`dataset_type: "math_only"`)
- **Available Formats**: Generic formats + LaTeX math expressions
- **Additional Formats**: `$\boxed{}$`, `\begin{align}`, `$\text{}$`, etc.
- **Use Case**: Mathematical problem solving
- **Examples**: GSM8K, MATH, mathematical reasoning datasets

### **Code-Only Datasets** (`dataset_type: "code_only"`)
- **Available Formats**: Generic formats + programming-specific formats
- **Additional Formats**: `print()`, `console.log()`, comments, return statements
- **Use Case**: Code generation, programming problems
- **Examples**: HumanEval, MBPP, coding datasets

## 🎲 Dynamic Compositor System

The dynamic compositor creates complex structured responses by randomly combining components:

### **Component Categories**
- **Analysis**: `problem_analysis`, `requirements_analysis`, `context_analysis`
- **Reasoning**: `logical_reasoning`, `step_by_step`, `causal_reasoning`
- **Planning**: `approach`, `methodology`, `strategy`
- **Technical**: `implementation`, `code_structure`, `algorithm`
- **Evaluation**: `validation`, `testing`, `verification`
- **Output**: `final_answer`, `conclusion`, `summary`

### **Output Formats**
- **XML**: `<component_name>content</component_name>`
- **JSON**: `{"component_name": "content"}`
- **YAML**: `component_name: content`
- **TOML**: `component_name = "content"`

### **Example Dynamic Format**
```xml
<problem_analysis>Understanding the requirements...</problem_analysis>
<logical_reasoning>Step by step analysis...</logical_reasoning>
<approach>My methodology will be...</approach>
<implementation>Here's the solution...</implementation>
<final_answer>42</final_answer>
```

## 📈 Monitoring & Analytics

### **WandB Metrics**
- `train/percent_correct`: Overall format compliance rate
- `train/format_success_rate_{format}`: Per-format success rates
- `train/format_usage_count_{format}`: Usage frequency per format
- `train/equivalent_ratio_paused_formats`: Number of paused formats
- `train/group_success_rate`: Percentage of successful groups
- `train/failed_groups_count`: Number of completely failed groups

### **Group-Level Statistics**
```
Format: json_confidence | Group average score: 0.7500 | 12/16 correct (75.0%)
Format: latex_boxed_math | Group average score: 0.0000 | 0/16 correct (0.0%) (All failures!)
Format: natural_language_answer | Group average score: 1.0000 | 16/16 correct (100.0%) (Perfect group!)
```

### **Data Dumps**
- **Regular Rollouts**: `answer_format_rollouts_{uuid}_{batch}.jsonl`
- **Failed Rollouts**: `answer_format_failed_rollouts_{uuid}_{batch}.jsonl`
- **Metadata**: Format type, scores, conversation history, timestamps
- **Batch Size**: 100 groups per file

## ⚖️ Equivalent Ratio Enforcement

Optional system to ensure balanced evaluation across all formats:

### **How It Works**
1. Tracks successful groups per format
2. Pauses formats that reach threshold (default: 50 successful groups)
3. Continues evaluating other formats until they catch up
4. Resumes paused formats when balance is restored

### **Configuration**
```python
ensure_equivalent_ratios=True,    # Enable the system
format_group_threshold=50,        # Success threshold per format
```

### **Monitoring**
```python
status = env.get_equivalent_ratio_status()
print(f"Paused formats: {status['paused_formats']}")
print(f"Active formats: {status['active_formats']}")
print(f"Progress: {status['format_progress']}")
```

## 🔧 Advanced Usage

### **Custom Format Filtering**
```python
from atropos.environments.answer_format_environment import AnswerFormat

config = AnswerFormatEnvConfig(
    supported_formats=[
        AnswerFormat.JSON,
        AnswerFormat.XML,
        AnswerFormat.LATEX_BOXED,
        AnswerFormat.NATURAL_LANGUAGE_ANSWER
    ]
)
```

### **Evaluation Mode**
```python
# Run evaluation on held-out set
eval_score = await env.evaluate()
print(f"Evaluation format compliance: {eval_score}")
```

### **Supported Dataset Formats**

#### **OpenHermes Conversations**
```python
{
    "name": "teknium/OpenHermes-2.5",
    "prompt_field": "conversations",
    "answer_field": "conversations"
}
```

#### **GSM8K Math Problems**
```python
{
    "name": "gsm8k",
    "prompt_field": "question",
    "answer_field": "answer"
}
```

#### **AcademicMCQA Multiple Choice**
```python
{
    "name": "NousResearch/AcademicMCQA",
    "prompt_field": "prompt",
    "answer_field": "ground_truth",
    "metadata_fields": ["answer", "options"]
}
```

## 📝 Response Format Requirements

### **Thinking Tags**
- **Required**: Exactly one `<think>` opening and one `</think>` closing tag
- **Content**: All reasoning must be inside thinking tags
- **Placement**: Answer must appear after `</think>` in specified format
- **Validation**: No additional thinking tags allowed after first `</think>`

### **Example Valid Response**
```
<think>
Let me analyze this problem step by step.
First, I need to understand what's being asked...
The solution involves calculating...
</think>

{"answer": "42"}
```

### **Example Invalid Response**
```
<think>Some reasoning</think>
{"answer": "42"}
<think>More reasoning</think>  // ❌ Additional thinking tags not allowed
```

## 🚨 Common Issues & Solutions

### **Format Validation Failures**
- **Issue**: Response doesn't match expected format
- **Solution**: Check regex patterns and ensure exact format compliance
- **Debug**: Enable `debug_logging=True` for detailed validation info

### **Thinking Tag Violations**
- **Issue**: Multiple thinking sections or missing tags
- **Solution**: Ensure exactly one `<think></think>` section per response
- **Debug**: Check thinking tag validation logs

### **Dataset Loading Errors**
- **Issue**: Dataset not found or field missing
- **Solution**: Verify dataset name, split, and field names
- **Debug**: Check dataset configuration and field mappings

### **Memory Issues with Large Datasets**
- **Issue**: Out of memory with large sample sizes
- **Solution**: Reduce `sample_size` or use streaming datasets
- **Debug**: Monitor memory usage and adjust batch sizes

## 🔍 Debugging & Development

### **Enable Debug Logging**
```python
config = AnswerFormatEnvConfig(debug_logging=True)
```

### **Check Format Status**
```python
# Get equivalent ratio status
status = env.get_equivalent_ratio_status()

# Check format success rates
for format_name, success_rate in env.format_success_rates.items():
    print(f"{format_name}: {success_rate:.2%}")
```

### **Analyze Failed Rollouts**
```python
# Failed rollouts are automatically saved when dump_failed_rollouts=True
# Check the datadumps directory for analysis files
```

## 📚 Technical Details

### **Scoring System**
- **Format Compliance**: 1.0 for correct format, 0.0 for incorrect
- **No Answer Accuracy**: Content correctness is not evaluated
- **Group Scoring**: Average of all rollouts in a group
- **Success Threshold**: Groups with >0.0 average are considered successful

### **Format Validation**
- **Regex Patterns**: Each format has specific validation patterns
- **Exact Matching**: Strict compliance required for scoring
- **Content Extraction**: Validated content is extracted for consistency

### **Dynamic Format Generation**
- **Component Selection**: Random selection of 3-6 components
- **Format Templates**: XML, JSON, YAML, TOML output formats
- **Validation Storage**: Components stored for precise validation

### **Special Dataset Handling**
- **GSM8K**: Extracts numerical answers from `####` separated format
- **AcademicMCQA**: Uses ground truth letters (A, B, C, D) as answers
- **OpenHermes**: Extracts from conversation format with role-based parsing

## 🤝 Contributing

### **Adding New Formats**
1. Add enum value to `AnswerFormat`
2. Add system prompt instruction
3. Add validation regex pattern
4. Add content extraction logic
5. Test with sample responses

### **Adding New Dataset Types**
1. Define dataset type in configuration
2. Add format filtering logic
3. Update format selection methods
4. Test with representative datasets

### **Adding New Datasets**
1. Add special handling in `setup()` method
2. Define field mappings in configuration
3. Add metadata extraction logic
4. Test dataset loading and processing

## 📄 License

This environment is part of the Atropos training framework. See the main repository for license information.

---

**Need Help?** Check the debug logs, enable verbose logging, or review the comprehensive monitoring metrics for troubleshooting guidance.
