#!/usr/bin/env python3
"""
PDF Processing Pipeline Script

Example script for processing PDF files with progress tracking and resumption.
Uses pdfplumber for high-quality text extraction with layout preservation.

Usage:
    python -m data_pipeline.process_pdfs --pdf-dir data_pipeline/pdfs --output-dir data_pipeline/output

Features:
- Processes all PDFs in the specified directory using pdfplumber
- Tracks progress in JSON file for resumption
- Handles errors gracefully with detailed logging
- Extracts tables and preserves text layout
"""

import argparse
import logging
import sys
from pathlib import Path

from .pdf_processor import PDFProcessingConfig, PDFProcessor
from .progress_tracker import ProgressTracker


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("pdf_processing.log")],
    )


def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description="Process PDF files for story generation using pdfplumber"
    )
    parser.add_argument(
        "--pdf-dir", required=True, type=Path, help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for processed output"
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default="processing_progress.json",
        help="JSON file for tracking progress",
    )
    parser.add_argument(
        "--max-files", type=int, help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--reset-failed",
        action="store_true",
        help="Reset failed files to retry processing",
    )
    parser.add_argument(
        "--show-summary", action="store_true", help="Show progress summary and exit"
    )
    parser.add_argument(
        "--no-layout-preservation",
        action="store_true",
        help="Disable layout preservation (faster but less accurate)",
    )
    parser.add_argument(
        "--no-tables", action="store_true", help="Disable table extraction"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Initialize progress tracker
    progress_tracker = ProgressTracker(args.progress_file)

    # Handle special commands
    if args.show_summary:
        progress_tracker.print_summary()
        return

    if args.reset_failed:
        progress_tracker.reset_failed_files()
        logger.info("Reset all failed files")

    # Validate directories
    if not args.pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {args.pdf_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Configure PDF processing
    config = PDFProcessingConfig(
        save_raw_text=True,
        save_cleaned_text=True,
        remove_headers_footers=True,
        normalize_whitespace=True,
        layout_preservation=not args.no_layout_preservation,
        extract_tables=not args.no_tables,
    )

    # Initialize PDF processor
    try:
        processor = PDFProcessor(config=config, progress_tracker=progress_tracker)
    except ImportError as e:
        logger.error(f"Failed to initialize PDF processor: {e}")
        logger.error("Please install pdfplumber with: pip install pdfplumber>=0.9.0")
        sys.exit(1)

    # Process PDFs
    try:
        logger.info("Starting PDF processing with pdfplumber...")
        logger.info(f"PDF directory: {args.pdf_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Progress file: {args.progress_file}")
        logger.info(f"Layout preservation: {config.layout_preservation}")
        logger.info(f"Table extraction: {config.extract_tables}")

        results = processor.process_directory(
            pdf_dir=args.pdf_dir, output_dir=args.output_dir, max_files=args.max_files
        )

        logger.info("Processing complete!")
        logger.info(f"Successfully processed {len(results)} files")

        # Show final summary
        progress_tracker.print_summary()

        # Show detailed statistics
        if results:
            total_pages = sum(r.page_count for r in results)
            total_chars = sum(len(r.cleaned_text) for r in results)
            total_words = sum(len(r.cleaned_text.split()) for r in results)
            avg_confidence = sum(r.confidence_score for r in results) / len(results)

            logger.info("Processing Statistics:")
            logger.info(f"  Total files processed: {len(results)}")
            logger.info(f"  Total pages processed: {total_pages}")
            logger.info(f"  Total characters extracted: {total_chars:,}")
            logger.info(f"  Total words extracted: {total_words:,}")
            logger.info(f"  Average confidence score: {avg_confidence:.2f}")
            logger.info(f"  Average pages per file: {total_pages / len(results):.1f}")
            logger.info(f"  Average words per file: {total_words / len(results):,.0f}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
