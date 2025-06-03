# Longform Story Generation - Implementation TODO

This document outlines the complete implementation plan for the longform story generation RL trainer, based on the ReasoningNCP approach with VR-CLI (Verifiable Rewards via Completion Likelihood Improvement).

## Overview

The system trains models to predict useful plans for the next chapter of a story using:
- **Next-Chapter Prediction (NCP)** task for training
- **VR-CLI rewards** based on completion likelihood improvement over baseline
- **Multi-component dataset** with stories, summaries, and character information

## Phase 1: Data Pipeline Setup

### 1.1 PDF Processing Infrastructure
- [ ] Create `pdf_processor.py` for extracting text from PDF files
  - [ ] Support for various PDF formats (text-based, OCR for scanned)
  - [ ] Handle multi-column layouts, headers, footers
  - [ ] Extract chapter titles and page numbers
  - [ ] Clean up formatting artifacts

### 1.2 Chapter Boundary Detection
- [ ] Implement `chapter_detector.py` for automatic chapter detection
  - [ ] Pattern matching for chapter headers (Chapter X, Part Y, etc.)
  - [ ] ML-based detection for non-standard formats
  - [ ] Manual override/correction interface
  - [ ] Validation and quality checks

### 1.3 Character Extraction & Biography Generation
- [ ] Create `character_extractor.py` for identifying main characters
  - [ ] Named Entity Recognition (NER) for character names
  - [ ] Character mention frequency analysis
  - [ ] Character relationship mapping
- [ ] Implement `character_bio_generator.py` using LLM
  - [ ] Generate character sheets per chapter
  - [ ] Track character development over time
  - [ ] Summarize character information efficiently

### 1.4 Content Summarization
- [ ] Create `summarizer.py` for multi-level summarization
  - [ ] Chapter-level summaries
  - [ ] Progressive plot summaries (up to chapter N)
  - [ ] High-level story summaries
  - [ ] Character sheet summarization
- [ ] Implement prompt templates for consistent summarization

### 1.5 Dataset Compilation
- [ ] Create `dataset_compiler.py` to format training data
  - [ ] Combine all components into NCP format
  - [ ] Generate training/validation/test splits
  - [ ] Apply filtering criteria (word limits, quality checks)
  - [ ] Export to JSONL format compatible with training

## Phase 2: Training Infrastructure

### 2.1 Data Loading & Preprocessing
- [ ] Implement `ncp_dataset.py` for PyTorch data loading
  - [ ] Efficient tokenization and batching
  - [ ] Support for variable-length sequences
  - [ ] Memory-efficient character sheet handling
- [ ] Create prompt templates in `prompt_utils.py`
  - [ ] System prompts for different model types
  - [ ] Reasoning chain generation
  - [ ] Flexible component inclusion (summaries, character sheets)

### 2.2 VR-CLI Reward System
- [ ] Implement `vr_cli_rewards.py` for reward computation
  - [ ] Baseline perplexity computation using reference model
  - [ ] Policy model perplexity evaluation
  - [ ] Completion likelihood improvement calculation
  - [ ] Reward function variants (clipped, binary, continuous)
- [ ] Create `baseline_computer.py` for precomputing baselines
  - [ ] Parallel processing for efficiency
  - [ ] Caching and persistence

### 2.3 RL Training Integration
- [ ] Adapt OpenRLHF or similar library for our use case
  - [ ] Custom experience maker for VR-CLI rewards
  - [ ] Story-specific evaluation metrics
  - [ ] Checkpoint management and model saving
- [ ] Create training scripts:
  - [ ] `train_story_rl.py` - Main RL training script
  - [ ] `train_story_sft.py` - Supervised fine-tuning baseline
  - [ ] Shell scripts for different model sizes

### 2.4 Evaluation Framework
- [ ] Implement `story_evaluator.py` for comprehensive evaluation
  - [ ] Automated metrics (perplexity, BLEU, etc.)
  - [ ] Story coherence evaluation
  - [ ] Character consistency checking
  - [ ] Plot progression analysis
- [ ] Create human evaluation interface
  - [ ] Web interface for annotators
  - [ ] Quality scoring rubrics
  - [ ] Inter-annotator agreement tracking

## Phase 3: Generation & Testing

### 3.1 Story Generation Interface
- [ ] Create `story_generator.py` for inference
  - [ ] Support for trained and pretrained models
  - [ ] Interactive generation with user input
  - [ ] Batch generation for evaluation
  - [ ] Configurable generation parameters

### 3.2 Quality Assurance
- [ ] Implement comprehensive testing suite
  - [ ] Unit tests for all components
  - [ ] Integration tests for full pipeline
  - [ ] Data quality validation
  - [ ] Model output validation

### 3.3 Documentation & Examples
- [ ] Create comprehensive documentation
  - [ ] API documentation for all modules
  - [ ] Tutorial notebooks with examples
  - [ ] Configuration guides
  - [ ] Troubleshooting guide

## Phase 4: Integration & Optimization

### 4.1 Atropos Backend Integration
- [ ] Adapt for Atropos training infrastructure
  - [ ] SLURM job submission integration
  - [ ] Distributed training support
  - [ ] Wandb logging integration
  - [ ] Resource management optimization

### 4.2 Performance Optimization
- [ ] Profile and optimize bottlenecks
  - [ ] Memory usage optimization
  - [ ] Training speed improvements
  - [ ] Inference acceleration
  - [ ] Distributed processing where applicable

### 4.3 Configuration Management
- [ ] Create comprehensive config system
  - [ ] YAML-based configuration files
  - [ ] Environment-specific overrides
  - [ ] Hyperparameter sweep support
  - [ ] Reproducibility guarantees

## Directory Structure Plan

```
environments/longform_story_generation/
├── ReasoningNCP/                    # Reference implementation (submodule)
├── data_pipeline/
│   ├── pdf_processor.py
│   ├── chapter_detector.py
│   ├── character_extractor.py
│   ├── character_bio_generator.py
│   ├── summarizer.py
│   └── dataset_compiler.py
├── training/
│   ├── ncp_dataset.py
│   ├── prompt_utils.py
│   ├── vr_cli_rewards.py
│   ├── baseline_computer.py
│   ├── train_story_rl.py
│   └── train_story_sft.py
├── evaluation/
│   ├── story_evaluator.py
│   └── human_eval_interface.py
├── generation/
│   └── story_generator.py
├── configs/
│   ├── default.yaml
│   ├── training.yaml
│   └── evaluation.yaml
├── scripts/
│   ├── setup_data.sh
│   ├── train_3b.sh
│   └── train_7b.sh
├── tests/
├── docs/
├── data/                           # Raw PDFs and processed datasets
│   ├── pdfs/
│   ├── processed/
│   └── datasets/
└── examples/
    └── tutorial.ipynb
```

## Key Implementation Notes

### Data Requirements (per ReasoningNCP format):
- `story_text`: Previous chapter(s) content
- `next_chapter`: Target chapter to predict
- `chapter_index`: Current position in story
- `prior_plot_summary`: Summary of story up to current point
- `high_level_plot_summary`: Overall story summary
- `character_sheets`: Dictionary of character information
- `next_chapter_synopsis`: Gold synopsis for training
- `last_n_chapters`: Number of previous chapters included
- `next_chapter_header`: Chapter title/header information

### VR-CLI Reward Function:
```
improvement = 100 * [(baseline_ppl - policy_ppl) / baseline_ppl]
reward = reward_function(improvement)  # e.g., min(improvement, 0) or binary
```

### Quality Thresholds:
- Chapter word limits: 200-5000 words
- Maximum message words: 10,000
- Minimum chapter index: 2 (to have prior context)
- Maximum chapter offset: 2 (leave final chapters for validation)

## Dependencies to Add

- PyPDF2/pdfplumber for PDF processing
- spaCy/transformers for NLP tasks
- OpenRLHF or similar for RL training
- VLLM for efficient LLM inference
- Ray for distributed processing
- Additional visualization libraries for evaluation

## Success Metrics

- [ ] Successfully process PDFs into structured datasets
- [ ] Generate high-quality character sheets and summaries
- [ ] Train models showing VR-CLI reward improvement
- [ ] Achieve coherent multi-chapter story generation
- [ ] Match or exceed ReasoningNCP baseline performance
- [ ] Integration with Atropos infrastructure
- [ ] Comprehensive evaluation suite operational
