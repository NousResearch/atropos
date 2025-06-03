"""
Chapter Boundary Detection using LLM Classification

Uses a language model to identify chapter boundaries in extracted text,
with data collection for training a faster BERT classifier.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import dotenv
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from .progress_tracker import ProcessingStage, ProgressTracker


class ContentType(Enum):
    """Types of content in a book"""

    STORY = "story"  # Actual narrative chapters
    FOREWORD = "foreword"  # Preface, introduction, foreword
    CONTENTS = "contents"  # Table of contents
    PROLOGUE = "prologue"  # Story prologue
    EPILOGUE = "epilogue"  # Story epilogue
    APPENDIX = "appendix"  # Supplementary material
    INDEX = "index"  # Index, bibliography
    NOTES = "notes"  # Editorial notes, text notes
    OTHER = "other"  # Unknown/miscellaneous


class BoundaryType(Enum):
    """Types of boundaries detected"""

    CHAPTER_START = "chapter_start"  # Beginning of a new chapter
    CHAPTER_END = "chapter_end"  # End of a chapter
    SECTION_START = "section_start"  # Beginning of a major section
    SECTION_END = "section_end"  # End of a major section
    BOOK_START = "book_start"  # Beginning of a book (multi-book works)
    BOOK_END = "book_end"  # End of a book
    NONE = "none"  # No boundary detected


@dataclass
class ContentBoundary:
    """Represents a detected content boundary with rich metadata"""

    start_position: int
    end_position: int
    content_type: ContentType
    boundary_type: BoundaryType
    boundary_text: Optional[str]  # Exact text that marks the boundary
    chapter_title: Optional[str]  # Title if it's a story chapter
    confidence: float
    is_story_content: bool  # Whether to include in training data
    text_chunk: str
    reasoning: str


@dataclass
class ChapterDetectionConfig:
    """Configuration for LLM-based chapter detection"""

    chunk_size: int = 1000  # Words per chunk
    chunk_overlap: int = 200  # Overlapping words between chunks
    api_delay: float = 0.5  # Delay between API calls to avoid rate limits
    max_retries: int = 3

    # Dataset collection
    save_training_data: bool = True
    training_data_file: str = "chapter_detection_training_data.jsonl"

    # Content filtering
    story_content_types: List[ContentType] = None  # Types to include in training

    def __post_init__(self):
        if self.story_content_types is None:
            self.story_content_types = [
                ContentType.STORY,
                ContentType.PROLOGUE,
                ContentType.EPILOGUE,
            ]


class ChapterDetector:
    """
    LLM-based chapter boundary detection with training data collection.

    Features:
    - Uses DeepHermes API for robust chapter detection
    - Classifies content types (story vs. metadata)
    - Handles various chapter formatting styles
    - Collects labeled data for training faster BERT classifier
    - Chunked processing for large texts
    - Error handling and retry logic
    """

    def __init__(
        self,
        config: Optional[ChapterDetectionConfig] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ):
        """
        Initialize chapter detector.

        Args:
            config: Detection configuration
            progress_tracker: Progress tracker instance
        """
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI library required. Install with: pip install openai python-dotenv"
            )

        self.config = config or ChapterDetectionConfig()
        self.progress_tracker = progress_tracker

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        dotenv.load_dotenv()
        api_key = os.getenv("HERMES_API_KEY")
        if not api_key:
            raise ValueError("HERMES_API_KEY not found in environment variables")

        # Initialize OpenAI client with custom endpoint
        self.client = OpenAI(
            api_key=api_key, base_url="https://inference-api.nousresearch.com/v1"
        )

        # Training data collection
        self.training_data = []

        self.logger.info("Chapter detector initialized with DeepHermes API")

    def _create_system_prompt(self) -> str:
        """Create the system prompt for enhanced content analysis"""
        return """
You are an expert text analyzer specializing in identifying and classifying content in books and novels.

Your task is to analyze a chunk of text and determine:
1. What TYPE of content this is (story vs. metadata)
2. Whether it contains boundaries between sections
3. The exact text that marks any boundaries

CONTENT TYPES:
- story: Actual narrative chapters with plot, characters, dialogue
- foreword: Preface, introduction, foreword, author's note
- contents: Table of contents, chapter listings
- prologue: Story prologue or opening setup
- epilogue: Story epilogue or conclusion
- appendix: Supplementary material, appendices, notes
- index: Index, bibliography, references
- notes: Editorial notes, publication information
- other: Unknown or miscellaneous content

BOUNDARY TYPES:
- chapter_start: Beginning of a new story chapter
- chapter_end: End of a story chapter
- section_start: Beginning of major section (Book I, Part 1, etc.)
- section_end: End of major section
- book_start: Beginning of a book in multi-book work
- book_end: End of a book
- none: No significant boundary

Look for these boundary indicators:
- Chapter titles (e.g., "Chapter 1", "THE SHADOW OF THE PAST", "A Long-expected Party")
- Section headers (e.g., "BOOK ONE", "PART II", "THE FELLOWSHIP OF THE RING")
- Clear narrative breaks or transitions
- Formatting that suggests divisions

Respond with your analysis in this exact XML format:
<analysis>
<content_type>story|foreword|contents|prologue|epilogue|appendix|index|notes|other</content_type>
<contains_boundary>true/false</contains_boundary>
<boundary_type>chapter_start|chapter_end|section_start|section_end|book_start|book_end|none</boundary_type>
<boundary_text>exact text that marks the boundary, or null</boundary_text>
<chapter_title>title if it's a story chapter, or null</chapter_title>
<confidence>0.0-1.0</confidence>
<reasoning>brief explanation of your analysis</reasoning>
</analysis>

Be conservative with boundaries - only mark as true if you're confident there's a clear division."""

    def _classify_chunk(self, text_chunk: str) -> Dict:
        """
        Classify a text chunk using the LLM API.

        Args:
            text_chunk: Text to analyze

        Returns:
            Dictionary with classification results
        """
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="DeepHermes-3-Llama-3-8B-Preview",
                    messages=[
                        {"role": "system", "content": self._create_system_prompt()},
                        {
                            "role": "user",
                            "content": f"Analyze this text chunk:\n\n{text_chunk}",
                        },
                    ],
                    max_tokens=512,
                    temperature=0.1,  # Low temperature for consistent classification
                )

                # Parse the response
                response_text = response.choices[0].message.content
                return self._parse_llm_response(response_text, text_chunk)

            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(
                        self.config.api_delay * (attempt + 1)
                    )  # Exponential backoff
                else:
                    raise

        raise RuntimeError(
            f"Failed to classify chunk after {self.config.max_retries} attempts"
        )

    def _parse_llm_response(self, response_text: str, original_chunk: str) -> Dict:
        """
        Parse the LLM response and extract classification data.

        Args:
            response_text: Raw LLM response
            original_chunk: Original text chunk

        Returns:
            Parsed classification data
        """
        # Extract XML content
        analysis_match = re.search(
            r"<analysis>(.*?)</analysis>", response_text, re.DOTALL
        )
        if not analysis_match:
            self.logger.warning("Could not parse LLM response, using defaults")
            return {
                "content_type": ContentType.OTHER,
                "contains_boundary": False,
                "boundary_type": BoundaryType.NONE,
                "boundary_text": None,
                "chapter_title": None,
                "confidence": 0.0,
                "reasoning": "Parse error",
                "raw_response": response_text,
            }

        analysis_content = analysis_match.group(1)

        # Parse individual fields
        try:
            content_type_str = self._extract_xml_field(
                analysis_content, "content_type", "other"
            )
            content_type = ContentType(content_type_str)
        except ValueError:
            content_type = ContentType.OTHER

        contains_boundary = (
            self._extract_xml_field(analysis_content, "contains_boundary") == "true"
        )

        try:
            boundary_type_str = self._extract_xml_field(
                analysis_content, "boundary_type", "none"
            )
            boundary_type = BoundaryType(boundary_type_str)
        except ValueError:
            boundary_type = BoundaryType.NONE

        boundary_text = self._extract_xml_field(analysis_content, "boundary_text")
        chapter_title = self._extract_xml_field(analysis_content, "chapter_title")
        confidence = float(
            self._extract_xml_field(analysis_content, "confidence", "0.0")
        )
        reasoning = self._extract_xml_field(
            analysis_content, "reasoning", "No reasoning provided"
        )

        # Clean up null values
        if boundary_text and boundary_text.lower() in ["null", "none", ""]:
            boundary_text = None
        if chapter_title and chapter_title.lower() in ["null", "none", ""]:
            chapter_title = None

        # Determine if this is story content for training
        is_story_content = content_type in self.config.story_content_types

        result = {
            "content_type": content_type,
            "contains_boundary": contains_boundary,
            "boundary_type": boundary_type,
            "boundary_text": boundary_text,
            "chapter_title": chapter_title,
            "confidence": confidence,
            "is_story_content": is_story_content,
            "reasoning": reasoning,
            "raw_response": response_text,
            "text_chunk": original_chunk,
        }

        # Collect training data
        if self.config.save_training_data:
            training_sample = {
                "text": original_chunk,
                "content_type": content_type.value,
                "contains_boundary": contains_boundary,
                "boundary_type": boundary_type.value,
                "boundary_text": boundary_text,
                "chapter_title": chapter_title,
                "confidence": confidence,
                "is_story_content": is_story_content,
                "reasoning": reasoning,
            }
            self.training_data.append(training_sample)

        return result

    def _extract_xml_field(
        self, xml_content: str, field_name: str, default: str = ""
    ) -> str:
        """Extract a field from XML content"""
        pattern = f"<{field_name}>(.*?)</{field_name}>"
        match = re.search(pattern, xml_content, re.DOTALL)
        return match.group(1).strip() if match else default

    def _create_text_chunks(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks for analysis.

        Args:
            text: Full text to chunk

        Returns:
            List of (chunk_text, start_position) tuples
        """
        words = text.split()
        chunks = []

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 50:  # Skip very short chunks
                break

            chunk_text = " ".join(chunk_words)
            start_pos = len(" ".join(words[:i]))
            chunks.append((chunk_text, start_pos))

        return chunks

    def detect_content_boundaries(
        self, text: str, filename: Optional[str] = None
    ) -> List[ContentBoundary]:
        """
        Detect content boundaries and classify content types using LLM.

        Args:
            text: Full text to analyze
            filename: Optional filename for progress tracking

        Returns:
            List of detected content boundaries
        """
        if filename and self.progress_tracker:
            # Check if already processed
            if filename in self.progress_tracker.progress_data:
                status = self.progress_tracker.progress_data[filename]
                if status.current_stage != ProcessingStage.CHAPTER_DETECTION:
                    self.logger.info(
                        f"Skipping {filename} - already at stage {status.current_stage.value}"
                    )
                    return []

        self.logger.info(
            f"Starting content boundary detection for {filename or 'text'}"
        )

        # Create chunks
        chunks = self._create_text_chunks(text)
        self.logger.info(f"Created {len(chunks)} text chunks for analysis")

        boundaries = []
        story_boundaries = []

        try:
            # Process each chunk
            for i, (chunk_text, start_pos) in enumerate(chunks):
                self.logger.info(f"Analyzing chunk {i+1}/{len(chunks)}")

                # Classify chunk
                result = self._classify_chunk(chunk_text)

                # If boundary detected with sufficient confidence
                if result["contains_boundary"] and result["confidence"] > 0.5:
                    boundary = ContentBoundary(
                        start_position=start_pos,
                        end_position=start_pos + len(chunk_text),
                        content_type=result["content_type"],
                        boundary_type=result["boundary_type"],
                        boundary_text=result["boundary_text"],
                        chapter_title=result["chapter_title"],
                        confidence=result["confidence"],
                        is_story_content=result["is_story_content"],
                        text_chunk=(
                            chunk_text[:200] + "..."
                            if len(chunk_text) > 200
                            else chunk_text
                        ),
                        reasoning=result["reasoning"],
                    )
                    boundaries.append(boundary)

                    # Track story content separately
                    if boundary.is_story_content:
                        story_boundaries.append(boundary)

                    boundary_desc = f"{boundary.content_type.value}"
                    if boundary.chapter_title:
                        boundary_desc += f": {boundary.chapter_title}"
                    elif boundary.boundary_text:
                        boundary_desc += f": {boundary.boundary_text}"

                    self.logger.info(
                        f"Found {boundary.boundary_type.value} - {boundary_desc}"
                    )

                # Rate limiting
                time.sleep(self.config.api_delay)

            # Save training data
            if self.config.save_training_data and self.training_data:
                self._save_training_data()

            # Update progress tracker
            if filename and self.progress_tracker:
                metadata = {
                    "total_boundaries_detected": len(boundaries),
                    "story_boundaries_detected": len(story_boundaries),
                    "chunks_analyzed": len(chunks),
                    "training_samples_collected": len(self.training_data),
                    "content_types_found": list(
                        set(b.content_type.value for b in boundaries)
                    ),
                }
                self.progress_tracker.update_stage(
                    filename, ProcessingStage.CHARACTER_EXTRACTION, metadata=metadata
                )

            self.logger.info(
                f"Content analysis complete. Found {len(boundaries)} total boundaries, "
                f"{len(story_boundaries)} story boundaries"
            )

            return boundaries

        except Exception as e:
            error_msg = f"Content boundary detection failed: {str(e)}"
            self.logger.error(error_msg)

            if filename and self.progress_tracker:
                self.progress_tracker.update_stage(
                    filename, ProcessingStage.CHAPTER_DETECTION, error_message=error_msg
                )

            raise

    def _save_training_data(self):
        """Save collected training data to JSONL file"""
        if not self.training_data:
            return

        output_file = Path(self.config.training_data_file)

        # Append to existing file or create new one
        with open(output_file, "a", encoding="utf-8") as f:
            for sample in self.training_data:
                f.write(json.dumps(sample) + "\n")

        self.logger.info(
            f"Saved {len(self.training_data)} training samples to {output_file}"
        )
        self.training_data.clear()  # Clear to avoid duplicates

    def process_file(self, text_file: Path, output_dir: Path) -> List[ContentBoundary]:
        """
        Process a text file and detect content boundaries.

        Args:
            text_file: Path to cleaned text file
            output_dir: Directory for output files

        Returns:
            List of detected content boundaries
        """
        filename = text_file.name

        # Read text
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Detect boundaries
        boundaries = self.detect_content_boundaries(text, filename)

        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save all boundaries
        boundaries_file = output_dir / f"{text_file.stem}_content_analysis.json"
        boundaries_data = [
            {
                "start_position": b.start_position,
                "end_position": b.end_position,
                "content_type": b.content_type.value,
                "boundary_type": b.boundary_type.value,
                "boundary_text": b.boundary_text,
                "chapter_title": b.chapter_title,
                "confidence": b.confidence,
                "is_story_content": b.is_story_content,
                "reasoning": b.reasoning,
                "text_preview": b.text_chunk,
            }
            for b in boundaries
        ]

        with open(boundaries_file, "w", encoding="utf-8") as f:
            json.dump(boundaries_data, f, indent=2, ensure_ascii=False)

        # Save story-only boundaries for backward compatibility
        story_boundaries = [b for b in boundaries if b.is_story_content]
        story_file = output_dir / f"{text_file.stem}_story_chapters.json"
        story_data = [
            {
                "start_position": b.start_position,
                "end_position": b.end_position,
                "chapter_title": b.chapter_title,
                "boundary_text": b.boundary_text,
                "confidence": b.confidence,
                "text_preview": b.text_chunk,
            }
            for b in story_boundaries
        ]

        with open(story_file, "w", encoding="utf-8") as f:
            json.dump(story_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved content analysis to {boundaries_file}")
        self.logger.info(
            f"Saved {len(story_boundaries)} story chapters to {story_file}"
        )

        return boundaries
