#!/usr/bin/env python3
"""
Test script for LLM-based content boundary detection.

Tests the enhanced content detection system on processed PDF text files.
"""

import json
import logging
from pathlib import Path

from .progress_tracker import ProgressTracker
from .text_classifier import ChapterDetectionConfig, ChapterDetector


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("content_detection_test.log"),
        ],
    )


def create_test_subset(input_file: Path, output_file: Path, num_lines: int = 3000):
    """Create a small test subset from the input file"""
    logger = logging.getLogger(__name__)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            lines.append(line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(
        f"Created test subset: {len(lines)} lines from {input_file.name} -> {output_file.name}"
    )
    return output_file


def test_content_detection():
    """Test content boundary detection on processed text files"""

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    # Configuration for testing
    config = ChapterDetectionConfig(
        chunk_size=800,  # Smaller chunks for testing
        chunk_overlap=150,
        api_delay=1.0,  # Slower to be respectful to API
        save_training_data=True,
        training_data_file="content_detection_training_data.jsonl",
    )

    # Initialize components
    progress_tracker = ProgressTracker("content_detection_progress.json")
    detector = ChapterDetector(config, progress_tracker)

    # Find processed text files
    output_dir = Path("data_pipeline/output")
    if not output_dir.exists():
        logger.error("Output directory not found. Run PDF processing first.")
        return

    # Look for text files
    text_files = list(output_dir.glob("*_cleaned.txt"))
    if not text_files:
        logger.error("No cleaned text files found in output directory")
        return

    logger.info(f"Found {len(text_files)} text files to process")

    # Process each file (create subset first for testing)
    for text_file in text_files:
        logger.info(f"Processing {text_file.name}")

        try:
            # Create a test subset first
            subset_file = output_dir / f"{text_file.stem}_test_subset.txt"
            create_test_subset(text_file, subset_file, num_lines=3000)

            # Detect boundaries on subset
            boundaries = detector.process_file(
                subset_file, output_dir / "content_analysis"
            )

            # Report results
            logger.info(f"Detection complete for {subset_file.name}")
            logger.info(f"Found {len(boundaries)} total boundaries")

            # Group by content type
            content_types = {}
            story_boundaries = []

            for boundary in boundaries:
                content_type = boundary.content_type.value
                if content_type not in content_types:
                    content_types[content_type] = []
                content_types[content_type].append(boundary)

                if boundary.is_story_content:
                    story_boundaries.append(boundary)

            # Report content type breakdown
            logger.info("Content type breakdown:")
            for content_type, boundaries_of_type in content_types.items():
                logger.info(f"  {content_type}: {len(boundaries_of_type)} boundaries")
                for i, boundary in enumerate(boundaries_of_type[:3]):  # Show first 3
                    title = (
                        boundary.chapter_title or boundary.boundary_text or "Untitled"
                    )
                    logger.info(
                        f"    {i+1}. {boundary.boundary_type.value}: {title} "
                        f"(confidence: {boundary.confidence:.2f})"
                    )
                if len(boundaries_of_type) > 3:
                    logger.info(f"    ... and {len(boundaries_of_type) - 3} more")

            logger.info(f"Story content boundaries: {len(story_boundaries)}")
            for i, boundary in enumerate(story_boundaries):
                title = boundary.chapter_title or boundary.boundary_text or "Untitled"
                logger.info(
                    f"  Story {i+1}: {title} (confidence: {boundary.confidence:.2f})"
                )

            # Clean up test subset
            subset_file.unlink()

        except Exception as e:
            logger.error(f"Failed to process {text_file.name}: {e}")
            continue

    logger.info("Content detection testing complete")

    # Check training data
    training_file = Path(config.training_data_file)
    if training_file.exists():
        # Count lines in JSONL file and analyze content types
        content_type_counts = {}
        boundary_type_counts = {}
        total_samples = 0

        with open(training_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line)
                        total_samples += 1

                        content_type = sample.get("content_type", "unknown")
                        boundary_type = sample.get("boundary_type", "unknown")

                        content_type_counts[content_type] = (
                            content_type_counts.get(content_type, 0) + 1
                        )
                        boundary_type_counts[boundary_type] = (
                            boundary_type_counts.get(boundary_type, 0) + 1
                        )
                    except (json.JSONDecodeError, KeyError):
                        continue

        logger.info(f"Collected {total_samples} training samples in {training_file}")
        logger.info("Training data content types:")
        for content_type, count in content_type_counts.items():
            logger.info(f"  {content_type}: {count}")
        logger.info("Training data boundary types:")
        for boundary_type, count in boundary_type_counts.items():
            logger.info(f"  {boundary_type}: {count}")
    else:
        logger.warning("No training data file found")


if __name__ == "__main__":
    test_content_detection()
