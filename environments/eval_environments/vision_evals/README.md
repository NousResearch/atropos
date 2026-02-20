# Vision Evaluation Benchmarks

This folder contains 27 vision and multimodal benchmarks for evaluating vision-language models. The implementations follow VLMEvalKit patterns where applicable and use the Atropos Eval class for consistent async evaluation.

## Benchmarks

| Benchmark | What it Tests | Dataset | Scoring |
|-----------|---------------|---------|---------|
| MMMU | Multi-discipline academic QA | MMMU/MMMU | MCQ accuracy |
| MMMU-Pro | Harder MMMU with 10 choices | MMMU/MMMU_Pro | MCQ accuracy |
| MMBench | General multimodal understanding | lmms-lab/MMBench | MCQ accuracy |
| MMStar | Expert-level multimodal QA | Lin-Chen/MMStar | MCQ accuracy |
| MMVet | Open-ended VLM capabilities | lmms-lab/MMVet | GPT scoring |
| MMVP | CLIP blindspot detection | MMVP/MMVP | MCQ accuracy |
| AI2D | Scientific diagram understanding | lmms-lab/ai2d | MCQ accuracy |
| BLINK | Visual perception tasks | BLINK-Benchmark/BLINK | MCQ accuracy |
| ChartQA | Chart question answering | ahmed-masry/ChartQA | Relaxed accuracy |
| CharXiv | Scientific chart understanding | princeton-nlp/CharXiv | GPT judge |
| CountBench | Object counting | nielsr/countbench | Numeric match |
| DocVQA | Document understanding | lmms-lab/DocVQA | ANLS score |
| DynaMath | Dynamic math reasoning | DynaMath/DynaMath_Sample | JSON extraction |
| HallusionBench | Visual hallucination detection | lmms-lab/HallusionBench | Yes/No accuracy |
| InfoVQA | Infographic QA | lmms-lab/InfoVQA | ANLS score |
| LogicVista | Visual logical reasoning | Yijia-Xiao/LogicVista | GPT extraction |
| MathVerse | Visual math (multi-version) | AI4Math/MathVerse | GPT extraction + scoring |
| MathVision | Visual math problems | MathLLMs/MathVision | GPT extraction |
| MathVista | Visual math reasoning | AI4Math/MathVista | GPT extraction |
| MMT-Bench | Multi-task multimodal | OpenGVLab/MMT-Bench | MCQ accuracy |
| OCRBench | OCR capabilities | echo840/OCRBench | Substring match |
| POPE | Object hallucination | lmms-lab/POPE | Yes/No accuracy |
| RealWorldQA | Real-world visual QA | xai-org/RealworldQA | Fuzzy match |
| SEED-Bench2 | Visual understanding | lmms-lab/SEED-Bench-2 | MCQ accuracy |
| VisuLogic | Visual logic puzzles | visulogic dataset | MCQ accuracy |
| VLMBlind | Basic visual perception | XAI/vlmsareblind | Task-specific |
| WeMath | Visual math with 4D metrics | We-Math/We-Math | 4D scoring |

## Running an Evaluation

All benchmarks use the same CLI pattern:

```bash
python mmmu_environment.py \
    --model-name "gpt-4o" \
    --server-url "https://api.openai.com/v1"
```

For local models with vLLM or Ollama:

```bash
python mmbench_environment.py \
    --model-name "Qwen/Qwen2-VL-7B-Instruct" \
    --server-url "http://localhost:8000/v1"
```

The evaluations use the `ServerManager` from atroposlib for making API calls.

## Comparison with VLMEvalKit

These implementations are aligned with VLMEvalKit where it makes sense, but simplified for standalone use. Here are the key differences and similarities:

### Scoring Methods

**ChartQA** uses relaxed accuracy with 5% tolerance. Percentages are converted to decimals before comparison (5% becomes 0.05). This matches VLMEvalKit behavior in `vqa_eval.py`.

**DocVQA and InfoVQA** use ANLS (Average Normalized Levenshtein Similarity) with a 0.5 threshold. This is the standard metric from the original papers.

**MathVista** uses GPT-based answer extraction with 5 in-context learning examples. There is a prefetch mechanism that tries regex first before calling GPT. The extraction prompt and ICL examples are taken from VLMEvalKit.

**MathVerse** uses two-stage GPT evaluation. First GPT extracts the answer from the response, then GPT judges whether the extracted answer matches the ground truth. This matches the VLMEvalKit approach.

**WeMath** computes 4-dimensional metrics beyond simple accuracy:
- IK (Insufficient Knowledge): wrong on steps AND wrong on multi
- IG (Inadequate Generalization): right on steps BUT wrong on multi
- CM (Complete Mastery): right on steps AND right on multi
- RM (Rote Memorization): wrong on steps BUT right on multi

**MMVet** uses GPT to score open-ended responses on a 0-1 scale. Without an API key it falls back to substring matching.

**OCRBench** uses category-specific scoring. For handwritten math expressions it compares without spaces. For other categories it does case-insensitive substring matching.

### What We Changed

**Simpler data loading**: We use HuggingFace datasets directly instead of VLMEvalKit's TSV preprocessing. This makes the code easier to understand but may load data slightly differently.

**Async evaluation**: Everything runs async with tqdm progress bars. VLMEvalKit uses synchronous evaluation by default.

**No circular evaluation**: VLMEvalKit supports "circular" MCQ evaluation where options are rotated and the model must get all rotations correct. We do not implement this, which means our MCQ scores may be slightly higher than VLMEvalKit on some benchmarks.

**Unified CLI**: All benchmarks use the same `eval_runner` CLI instead of VLMEvalKit's `run.py` with config files.

### Expected Score Differences

Due to the differences above, you should expect:

- MCQ benchmarks (MMMU, MMBench, MMStar, AI2D): Within 1-2% of VLMEvalKit
- VQA benchmarks (DocVQA, ChartQA): Very close, same scoring methods
- Math benchmarks (MathVista, MathVerse): Within 2-3%, depends on GPT extraction
- Open-ended (MMVet): Can vary more, depends on GPT judge prompts

## Benchmark Details

### General Multimodal Understanding

**MMMU** tests multi-discipline academic knowledge across 30 subjects from accounting to physics. Questions require understanding images and domain knowledge. The validation split has about 900 questions.

**MMMU-Pro** is a harder version with 10 answer choices instead of 4. It has three variants: standard (10 options), standard_4 (4 options), and vision (question in image).

**MMBench** is a comprehensive benchmark covering perception, reasoning, and knowledge. It has English and Chinese versions.

**MMStar** focuses on expert-level questions that require both visual understanding and specialized knowledge.

**SEED-Bench2** tests visual understanding across many categories including scene understanding, instance identity, and spatial relations. The dataset is large (24k samples) so we stream by default and limit to 1000 samples.

**MMT-Bench** is a multi-task benchmark covering 32 different task types. Good for testing breadth of capabilities.

### Document and Chart Understanding

**DocVQA** tests understanding of document images like forms, receipts, and scientific papers. Uses ANLS scoring which allows for minor OCR errors.

**InfoVQA** is similar to DocVQA but focuses on infographics with more complex layouts.

**ChartQA** tests chart reading. Has human and augmented subsets. The human subset is harder. Uses relaxed accuracy (5% tolerance for numbers).

**CharXiv** focuses on scientific charts from arXiv papers. Uses GPT as a judge with grading queries from the dataset.

**OCRBench** tests pure OCR capabilities across 10 categories from regular text to handwritten math expressions.

### Math and Reasoning

**MathVista** is a visual math benchmark with multiple question types (free form, multiple choice) and answer types (integer, float, text, list). Uses the dataset's built-in query prompts.

**MathVerse** has problems at different visual complexity levels from "text dominant" to "vision only". Uses two-stage GPT evaluation.

**MathVision** is another visual math benchmark. Uses GPT extraction with fallback to regex.

**DynaMath** tests dynamic math reasoning with JSON-formatted outputs. Has subject and difficulty level breakdowns.

**WeMath** provides detailed 4D metrics to understand where models fail. Good for diagnosing reasoning vs memorization issues.

**LogicVista** tests visual logical reasoning with 5 skill types. Supports multi-letter answers where multiple options can be correct.

**VisuLogic** tests visual logic with diagram-based puzzles.

### Perception and Hallucination

**POPE** tests object hallucination with yes/no questions about whether objects exist in images. Has random, popular, and adversarial variants.

**HallusionBench** tests visual hallucinations more broadly. Questions are designed to trick models into seeing things that are not there.

**MMVP** tests visual perception on cases where CLIP-based models tend to fail. Useful for understanding encoder limitations.

**BLINK** tests basic visual perception like counting, spatial relations, and similarity. Models often struggle on these "easy" tasks.

**VLMBlind** (VLMs Are Blind) tests very basic visual tasks that humans find trivial but VLMs often fail. Includes counting grid cells, finding circled letters, and counting Olympic rings.

**CountBench** is a simple object counting benchmark.

### Real World

**RealWorldQA** tests understanding of real-world images from XAI. Uses fuzzy matching for answers.

**AI2D** tests understanding of scientific diagrams from AI2 (Allen Institute). Good for testing diagram reasoning.

## GPT Judge Configuration

Several benchmarks use GPT for answer extraction or scoring. To enable this:

```bash
export OPENAI_API_KEY="your-key"
```

You can also configure the judge model when instantiating:

```python
eval_env = MathVista(
    use_gpt_extraction=True,
    judge_model="gpt-4o-mini",
    judge_base_url="https://api.openai.com/v1",
)
asyncio.run(eval_runner(eval_env))
```

Without an API key, benchmarks fall back to regex-based extraction which is less accurate but free.

## Output Format

Results are saved to the eval directory:

```
eval_results/
    metrics.json     # Overall scores
    samples.jsonl    # Per-item predictions
```

The metrics.json file contains accuracy and other metrics depending on the benchmark. The samples.jsonl file has one line per question with the prediction, answer, and whether it was correct.

## Adding New Benchmarks

To add a new vision benchmark:

1. Create a new file like `new_benchmark_environment.py`
2. Inherit from `EvalBase`
3. Implement `setup_data()` to load the dataset
4. Implement `run_item(self, server: ServerManager, data_item: dict)` to process one item
5. Use `await self.chat_completion(server, messages)` for API calls
6. Add image encoding using the standard `encode_image()` pattern

See any existing benchmark for a template. The MMMU implementation is a good starting point for MCQ benchmarks. DocVQA is a good template for VQA benchmarks.

## References

- VLMEvalKit: https://github.com/open-compass/VLMEvalKit
- OpenVLM Leaderboard: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
- MMMU: https://mmmu-benchmark.github.io/
- MathVista: https://mathvista.github.io/
- DocVQA: https://www.docvqa.org/
