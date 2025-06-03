# Data Pipeline for Longform Story Generation

This module provides tools for processing PDF books into structured datasets for training longform story generation models. It uses **pdfplumber** for high-quality text extraction with excellent layout preservation and table handling.

## Features

- **High-quality PDF text extraction** using pdfplumber
- **Layout preservation** for better text structure
- **Table extraction** for complex documents
- **Progress tracking** with JSON persistence for resumption
- **Robust error handling** with detailed logging
- **Text cleaning and normalization**

## Installation

```bash
# Install the required dependency
pip install -r data_pipeline/requirements.txt
```

## Quick Start

### 1. Add PDF Files

**Note**: PDF files are gitignored to save repository space. You need to add your own PDF book files locally.

Place your PDF book files in the `data_pipeline/pdfs/` directory:

```bash
# The directory structure is already created, just add your PDFs
cp /path/to/your/books/*.pdf environments/longform_story_generation/data_pipeline/pdfs/

# Example: adding some novels
cp ~/Documents/Books/*.pdf environments/longform_story_generation/data_pipeline/pdfs/
```

### 2. Process PDFs

Run the processing script:

```bash
# Basic usage (run from the longform_story_generation directory)
python -m data_pipeline.process_pdfs \
    --pdf-dir data_pipeline/pdfs \
    --output-dir data_pipeline/output

# With options
python -m data_pipeline.process_pdfs \
    --pdf-dir data_pipeline/pdfs \
    --output-dir data_pipeline/output \
    --max-files 5 \
    --verbose
```

### 3. Monitor Progress

The system automatically tracks progress in `processing_progress.json`. You can:

```bash
# View current progress
python -m data_pipeline.process_pdfs --show-summary

# Reset failed files for retry
python -m data_pipeline.process_pdfs --reset-failed

# Continue interrupted processing
python -m data_pipeline.process_pdfs --pdf-dir data_pipeline/pdfs --output-dir data_pipeline/output
```

## What's Gitignored

To keep the repository lightweight, the following are excluded from git:

- **PDF files**: `data_pipeline/pdfs/*.pdf` - Users provide their own
- **Processed output**: `data_pipeline/output/` - Generated files
- **Progress tracking**: `processing_progress.json` - Local state
- **Log files**: `pdf_processing.log` - Runtime logs

This means each user can work with their own collection of books without bloating the shared repository.

## Command Line Options

| Option | Description |
|--------|-------------|
| `--pdf-dir` | Directory containing PDF files (required) |
| `--output-dir` | Directory for processed output (required) |
| `--progress-file` | JSON file for tracking progress (default: processing_progress.json) |
| `--max-files` | Maximum number of files to process (for testing) |
| `--verbose` | Enable verbose logging |
| `--show-summary` | Show progress summary and exit |
| `--reset-failed` | Reset failed files to retry processing |
| `--no-layout-preservation` | Disable layout preservation (faster but less accurate) |
| `--no-tables` | Disable table extraction |

## Output Files

For each processed PDF, the system creates:

- `{filename}_raw.txt` - Raw extracted text
- `{filename}_cleaned.txt` - Cleaned and normalized text
- Progress tracking in JSON with metadata

## Progress Tracking

The system tracks files through these stages:

1. **PDF_EXTRACTION** - Text extraction from PDF
2. **CHAPTER_DETECTION** - Identify chapter boundaries (coming next)
3. **CHARACTER_EXTRACTION** - Extract characters and create bios (coming next)
4. **SUMMARIZATION** - Generate summaries (coming next)
5. **DATASET_COMPILATION** - Final dataset creation (coming next)
6. **COMPLETED** - All processing complete

## Configuration

Customize processing through `PDFProcessingConfig`:

```python
from data_pipeline import PDFProcessor, PDFProcessingConfig

config = PDFProcessingConfig(
    # Text cleaning options
    remove_headers_footers=True,
    remove_page_numbers=True,
    normalize_whitespace=True,
    fix_line_breaks=True,

    # Output options
    save_raw_text=True,
    save_cleaned_text=True,

    # pdfplumber options
    layout_preservation=True,
    extract_tables=True
)

processor = PDFProcessor(config=config)
```

## Error Handling

The system handles various error conditions gracefully:

- **Missing PDF files**: Skipped with warning
- **Corrupted PDFs**: Logged as failed with error details
- **Empty text extraction**: Marked as low confidence
- **Processing interruption**: Can be resumed from last successful stage

## Quality Assessment

Each extracted text receives a confidence score (0.0-1.0) based on:

- Text length and completeness
- Character distribution and special characters
- Sentence structure and word patterns
- Line length distribution

Files with low confidence scores may need manual review.

## Next Steps

After PDF processing, the next pipeline stages will be:

1. **Chapter Detection** - Automatically identify chapter boundaries
2. **Character Extraction** - Use NER to find main characters
3. **Summarization** - Generate plot and character summaries using LLMs
4. **Dataset Compilation** - Format everything for NCP training

## Troubleshooting

### Common Issues

**"pdfplumber not found"**
```bash
pip install pdfplumber>=0.9.0
```

**"No PDF files found"**
- Make sure you've added PDF files to `data_pipeline/pdfs/`
- Check that files have `.pdf` or `.PDF` extensions
- Verify file permissions are readable

**"Permission denied" errors**
- Check file permissions on PDF directory
- Ensure output directory is writable

**Poor text extraction quality**
- Try with `--no-layout-preservation` for faster processing
- Some PDFs may be scanned images (need OCR preprocessing)
- Check confidence scores in progress file

**Processing very slow**
- Use `--no-tables` to skip table extraction
- Process in smaller batches with `--max-files`
- Consider using faster hardware for large document sets

## Example Usage

```python
from pathlib import Path
from data_pipeline import PDFProcessor, ProgressTracker

# Initialize components
tracker = ProgressTracker("my_progress.json")
processor = PDFProcessor(progress_tracker=tracker)

# Process a single PDF
result = processor.process_pdf(
    pdf_path=Path("data_pipeline/pdfs/novel.pdf"),
    output_dir=Path("data_pipeline/output/")
)

print(f"Extracted {len(result.cleaned_text)} characters")
print(f"Confidence: {result.confidence_score:.2f}")

# Process directory
results = processor.process_directory(
    pdf_dir=Path("data_pipeline/pdfs/"),
    output_dir=Path("data_pipeline/output/")
)
```
