"""
Data Pipeline for Longform Story Generation

This module handles the complete data processing pipeline from PDF books to
structured datasets ready for training, including:

- PDF text extraction and cleaning
- Chapter boundary detection
- Character extraction and biography generation
- Multi-level summarization
- Dataset compilation in NCP format

Components:
- pdf_processor: Extract and clean text from PDF files
- chapter_detector: Identify chapter boundaries automatically
- character_extractor: Extract characters and generate biographies
- summarizer: Create chapter and plot summaries
- dataset_compiler: Format data for training
"""

# Import core components that are available
try:
    from .pdf_processor import (  # noqa: F401
        ExtractedText,
        PDFProcessingConfig,
        PDFProcessor,
    )
    from .progress_tracker import ProcessingStage, ProgressTracker  # noqa: F401

    __all__ = [
        "ProgressTracker",
        "ProcessingStage",
        "PDFProcessor",
        "PDFProcessingConfig",
        "ExtractedText",
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Some data pipeline components unavailable: {e}")
    __all__ = []
