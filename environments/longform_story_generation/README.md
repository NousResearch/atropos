# Longform Story Generation with VR-CLI

An implementation of reinforcement learning for long-form story generation using the Next-Chapter Prediction (NCP) task with Verifiable Rewards via Completion Likelihood Improvement (VR-CLI).

Based on ["Learning to Reason for Long-Form Story Generation"](https://arxiv.org/abs/2503.22828) by Alexander Gurung and Mirella Lapata.

## Overview

This system trains language models to generate reasoning and predict the next chapter of long-form stories by:

1. **Processing PDF books** into structured datasets with chapter boundaries
2. **Extracting characters** and generating dynamic character sheets
3. **Creating multi-level summaries** (chapter, plot, story-wide)
4. **Training with VR-CLI rewards** that measure completion likelihood improvement
5. **Generating coherent story continuations** with proper reasoning

## Quick Start

### 1. Setup Environment

```bash
# Navigate to the longform story generation directory
cd environments/longform_story_generation

# Install dependencies (TODO: create requirements.txt)
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Place your PDF books in the data/pdfs/ directory
cp /path/to/your/books/*.pdf data/pdfs/

# Run the data processing pipeline
python scripts/setup_data.sh
```

### 3. Train Model

```bash
# Start with supervised fine-tuning
python training/train_story_sft.py --config configs/sft_training.yaml

# Then apply RL training with VR-CLI
python training/train_story_rl.py --config configs/rl_training.yaml
```

### 4. Generate Stories

```bash
# Generate story continuations
python generation/story_generator.py --model-path /path/to/trained/model --input-story "Chapter 1 content..."
```

## Directory Structure

```
├── ReasoningNCP/           # Reference implementation (submodule)
├── data_pipeline/          # PDF processing and dataset creation
├── training/               # RL and SFT training scripts
├── evaluation/             # Evaluation metrics and interfaces
├── generation/             # Story generation utilities
├── configs/                # Configuration files
├── scripts/                # Setup and utility scripts
├── data/                   # Raw PDFs and processed datasets
├── examples/               # Tutorial notebooks
├── tests/                  # Test suites
└── docs/                   # Documentation
```

## Key Components

### VR-CLI Reward System
The core innovation is the VR-CLI reward function that measures improvement in completion likelihood:

```
improvement = 100 * [(baseline_ppl - policy_ppl) / baseline_ppl]
reward = reward_function(improvement)
```

### Dataset Format
Each training example contains:
- `story_text`: Previous chapter(s)
- `next_chapter`: Target chapter
- `character_sheets`: Dynamic character information
- `prior_plot_summary`: Story summary up to current point
- `next_chapter_synopsis`: Gold synopsis for training

### Multi-Component Processing
1. **PDF → Text**: Extract and clean text from various PDF formats
2. **Chapter Detection**: Automatically identify chapter boundaries
3. **Character Extraction**: Use NER and frequency analysis to identify main characters
4. **Summarization**: Generate chapter, plot, and character summaries using LLMs
5. **Dataset Compilation**: Format everything for training

## Implementation Status

See [TODO.md](TODO.md) for detailed implementation checklist.

**Current Phase**: Initial setup and planning
- ✅ Repository structure created
- ✅ Reference implementation integrated as submodule
- ⏳ PDF processing pipeline (Phase 1.1)
- ⏳ Chapter boundary detection (Phase 1.2)
- ⏳ Character extraction system (Phase 1.3)

## Citation

Original work:
```bibtex
@misc{gurung2025learningreasonlongformstory,
      title={Learning to Reason for Long-Form Story Generation},
      author={Alexander Gurung and Mirella Lapata},
      year={2025},
      eprint={2503.22828},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.22828},
}
```

## Contributing

1. Review the [TODO.md](TODO.md) for current implementation needs
2. Follow the existing code structure and documentation standards
3. Add comprehensive tests for new components
4. Update documentation for any API changes

## License

This implementation is part of the Nous Research Atropos project.
