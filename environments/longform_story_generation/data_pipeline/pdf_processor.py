"""
PDF Processing for Longform Story Generation

Handles extraction and cleaning of text from PDF files using pdfplumber,
which excels at handling complex layouts and preserving text structure.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError(
        "pdfplumber is required. Install with: pip install pdfplumber>=0.9.0"
    )

from .progress_tracker import ProcessingStage, ProgressTracker


@dataclass
class ExtractedText:
    """Container for extracted text and metadata"""

    raw_text: str
    cleaned_text: str
    page_count: int
    confidence_score: float
    metadata: Dict[str, Any]


@dataclass
class PDFProcessingConfig:
    """Configuration for PDF processing"""

    # Text cleaning options
    remove_headers_footers: bool = True
    remove_page_numbers: bool = True
    normalize_whitespace: bool = True
    fix_line_breaks: bool = True

    # Output options
    save_raw_text: bool = True
    save_cleaned_text: bool = True

    # pdfplumber specific options
    layout_preservation: bool = True
    extract_tables: bool = True


class PDFProcessor:
    """
    PDF text extraction and processing using pdfplumber.

    Features:
    - High-quality text extraction with layout preservation
    - Table extraction for complex documents
    - Text cleaning and normalization
    - Progress tracking and resumption
    - Robust error handling
    """

    def __init__(
        self,
        config: Optional[PDFProcessingConfig] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ):
        """
        Initialize PDF processor.

        Args:
            config: Processing configuration
            progress_tracker: Progress tracker instance
        """
        self.config = config or PDFProcessingConfig()
        self.progress_tracker = progress_tracker

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("PDF processor initialized with pdfplumber")

    def extract_text(self, pdf_path: Path) -> ExtractedText:
        """
        Extract text from PDF using pdfplumber.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractedText object with extracted content
        """
        try:
            text_parts = []
            metadata = {}

            with pdfplumber.open(pdf_path) as pdf:
                metadata["total_pages"] = len(pdf.pages)
                metadata["pdf_info"] = pdf.metadata if hasattr(pdf, "metadata") else {}

                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text with layout preservation
                        if self.config.layout_preservation:
                            page_text = page.extract_text(layout=True)
                        else:
                            page_text = page.extract_text()

                        if page_text and page_text.strip():
                            text_parts.append(page_text)

                        # Try table extraction if regular text is sparse and tables enabled
                        if self.config.extract_tables and (
                            not page_text or len(page_text.strip()) < 50
                        ):
                            tables = page.extract_tables()
                            for table in tables:
                                if table:  # Check if table is not None/empty
                                    # Convert table to text format
                                    table_text = "\n".join(
                                        [
                                            "\t".join([cell or "" for cell in row])
                                            for row in table
                                            if row
                                        ]
                                    )
                                    if table_text.strip():
                                        text_parts.append(table_text)

                    except Exception as e:
                        self.logger.warning(
                            f"Error extracting page {page_num + 1}: {e}"
                        )
                        continue

            # Combine all text parts
            raw_text = "\n\n".join(text_parts)

            # Clean the text
            cleaned_text = self._clean_text(raw_text)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(cleaned_text)

            return ExtractedText(
                raw_text=raw_text,
                cleaned_text=cleaned_text,
                page_count=metadata["total_pages"],
                confidence_score=confidence_score,
                metadata=metadata,
            )

        except Exception as e:
            self.logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""

        cleaned = text

        # Remove excessive whitespace
        if self.config.normalize_whitespace:
            # Normalize spaces but preserve intentional line breaks
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            # Limit consecutive newlines to maximum of 2
            cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

        # Fix line breaks in sentences (join hyphenated words and broken sentences)
        if self.config.fix_line_breaks:
            # Join hyphenated words split across lines
            cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
            # Join lines that seem to be broken mid-sentence (lowercase to lowercase)
            cleaned = re.sub(r"([a-z,])\n([a-z])", r"\1 \2", cleaned)

        # Remove headers/footers (basic patterns)
        if self.config.remove_headers_footers:
            # Remove lines that are likely headers/footers
            lines = cleaned.split("\n")
            filtered_lines = []
            for line in lines:
                line_stripped = line.strip()
                # Skip likely header/footer patterns
                if (
                    len(line_stripped) < 3
                    or re.match(r"^\d+$", line_stripped)  # Very short lines
                    or re.match(  # Just page numbers
                        r"^Chapter \d+$", line_stripped.title()
                    )
                    or line_stripped.lower()  # Standalone chapter headers
                    in ["", " ", "page"]
                ):  # Empty or page markers
                    continue
                filtered_lines.append(line)
            cleaned = "\n".join(filtered_lines)

        # Remove standalone page numbers
        if self.config.remove_page_numbers:
            # Remove lines that are just numbers (likely page numbers)
            cleaned = re.sub(r"^\s*\d+\s*$", "", cleaned, flags=re.MULTILINE)
            # Remove page numbers in the middle of text
            cleaned = re.sub(r"\n\s*\d+\s*\n", "\n", cleaned)

        # Final cleanup
        cleaned = cleaned.strip()
        # Ensure consistent paragraph separation
        cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)

        return cleaned

    def _calculate_confidence_score(self, text: str) -> float:
        """Calculate confidence score for extracted text quality"""
        if not text:
            return 0.0

        score = 1.0

        # Check text length (very short might indicate extraction issues)
        if len(text) < 100:
            score -= 0.5

        # Penalize excessive special characters (might indicate extraction errors)
        special_char_ratio = len(re.findall(r'[^\w\s.,!?;:\'"()-]', text)) / len(text)
        if special_char_ratio > 0.15:
            score -= min(0.4, special_char_ratio * 2)

        # Penalize very short lines (might indicate extraction issues)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            short_lines = sum(1 for line in lines if len(line) < 20)
            short_line_ratio = short_lines / len(lines)
            if short_line_ratio > 0.7:
                score -= 0.3

        # Reward proper sentence structure
        sentences = re.split(r"[.!?]+", text)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(
                sentences
            )
            if 5 <= avg_sentence_length <= 30:  # Reasonable sentence length
                score += 0.1
            elif avg_sentence_length < 3:  # Very short sentences might indicate issues
                score -= 0.2

        # Check for reasonable word distribution
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 3 <= avg_word_length <= 8:  # Reasonable average word length
                score += 0.1

        return max(0.0, min(1.0, score))

    def process_pdf(
        self, pdf_path: Path, output_dir: Optional[Path] = None
    ) -> ExtractedText:
        """
        Process a single PDF file with progress tracking.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted text files

        Returns:
            ExtractedText object
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        filename = pdf_path.name
        self.logger.info(f"Processing {filename}")

        # Register with progress tracker if available
        if self.progress_tracker:
            status = self.progress_tracker.register_pdf(pdf_path)
            if status.current_stage != ProcessingStage.PDF_EXTRACTION:
                self.logger.info(
                    f"Skipping {filename} - already at stage {status.current_stage.value}"
                )
                return None

        try:
            # Extract text
            result = self.extract_text(pdf_path)

            # Save output files if requested
            if output_dir and (
                self.config.save_raw_text or self.config.save_cleaned_text
            ):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                base_name = pdf_path.stem

                if self.config.save_raw_text:
                    raw_file = output_dir / f"{base_name}_raw.txt"
                    with open(raw_file, "w", encoding="utf-8") as f:
                        f.write(result.raw_text)

                if self.config.save_cleaned_text:
                    cleaned_file = output_dir / f"{base_name}_cleaned.txt"
                    with open(cleaned_file, "w", encoding="utf-8") as f:
                        f.write(result.cleaned_text)

            # Update progress tracker
            if self.progress_tracker:
                metadata = {
                    "text_length": len(result.cleaned_text),
                    "page_count": result.page_count,
                    "confidence_score": result.confidence_score,
                    "word_count": len(result.cleaned_text.split()),
                }
                self.progress_tracker.update_stage(
                    filename, ProcessingStage.CHAPTER_DETECTION, metadata=metadata
                )

            self.logger.info(
                f"Successfully processed {filename} - "
                f"{len(result.cleaned_text):,} characters, "
                f"confidence: {result.confidence_score:.2f}"
            )
            return result

        except Exception as e:
            error_msg = f"Failed to process {filename}: {str(e)}"
            self.logger.error(error_msg)

            # Update progress tracker with error
            if self.progress_tracker:
                self.progress_tracker.update_stage(
                    filename, ProcessingStage.PDF_EXTRACTION, error_message=error_msg
                )

            raise

    def process_directory(
        self, pdf_dir: Path, output_dir: Path, max_files: Optional[int] = None
    ) -> List[ExtractedText]:
        """
        Process all PDF files in a directory with progress tracking.

        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory for output files
            max_files: Maximum number of files to process (for testing)

        Returns:
            List of ExtractedText objects for successfully processed files
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)

        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        # Get pending files (not yet processed or failed)
        if self.progress_tracker:
            pending_files = self.progress_tracker.get_pending_files(pdf_dir)
        else:
            pending_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))

        if max_files:
            pending_files = pending_files[:max_files]

        self.logger.info(f"Found {len(pending_files)} PDF files to process")

        if self.progress_tracker:
            self.progress_tracker.print_summary()

        results = []
        for i, pdf_file in enumerate(pending_files):
            try:
                self.logger.info(
                    f"Processing file {i+1}/{len(pending_files)}: {pdf_file.name}"
                )
                result = self.process_pdf(pdf_file, output_dir)
                if result:
                    results.append(result)

            except Exception as e:
                self.logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue

        self.logger.info(f"Successfully processed {len(results)} files")

        if self.progress_tracker:
            self.progress_tracker.print_summary()

        return results
