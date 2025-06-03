"""
Progress Tracker for PDF Processing Pipeline

Handles tracking of processed PDFs and pipeline stages to enable resumption
of interrupted processing runs.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProcessingStage(Enum):
    """Stages of PDF processing pipeline"""

    PDF_EXTRACTION = "pdf_extraction"
    CHAPTER_DETECTION = "chapter_detection"
    CHARACTER_EXTRACTION = "character_extraction"
    SUMMARIZATION = "summarization"
    DATASET_COMPILATION = "dataset_compilation"
    COMPLETED = "completed"


@dataclass
class ProcessingStatus:
    """Status information for a single PDF"""

    filename: str
    file_path: str
    file_size: int
    file_hash: str
    current_stage: ProcessingStage
    completed_stages: List[ProcessingStage]
    start_time: str
    last_updated: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ProgressTracker:
    """
    Tracks progress of PDF processing pipeline with resumption capabilities.

    Features:
    - JSON-based persistence for progress tracking
    - File integrity checking via hashing
    - Stage-based processing tracking
    - Error handling and logging
    - Resumption of interrupted processing
    """

    def __init__(self, progress_file: str = "processing_progress.json"):
        """
        Initialize progress tracker.

        Args:
            progress_file: Path to JSON file for storing progress
        """
        self.progress_file = Path(progress_file)
        self.progress_data: Dict[str, ProcessingStatus] = {}
        self.load_progress()

    def load_progress(self) -> None:
        """Load existing progress from JSON file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert dict back to ProcessingStatus objects
                for filename, status_dict in data.items():
                    # Convert stage strings back to enum
                    status_dict["current_stage"] = ProcessingStage(
                        status_dict["current_stage"]
                    )
                    status_dict["completed_stages"] = [
                        ProcessingStage(stage)
                        for stage in status_dict["completed_stages"]
                    ]
                    self.progress_data[filename] = ProcessingStatus(**status_dict)

                print(f"Loaded progress for {len(self.progress_data)} files")
            except Exception as e:
                print(f"Error loading progress file {self.progress_file}: {e}")
                print("Starting with empty progress tracker")
                self.progress_data = {}

    def save_progress(self) -> None:
        """Save current progress to JSON file"""
        try:
            # Convert ProcessingStatus objects to dict for JSON serialization
            data = {}
            for filename, status in self.progress_data.items():
                status_dict = asdict(status)
                # Convert enums to strings for JSON serialization
                status_dict["current_stage"] = status.current_stage.value
                status_dict["completed_stages"] = [
                    stage.value for stage in status.completed_stages
                ]
                data[filename] = status_dict

            # Write to temporary file first, then rename for atomic operation
            temp_file = self.progress_file.with_suffix(".json.tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            temp_file.replace(self.progress_file)

        except Exception as e:
            print(f"Error saving progress file {self.progress_file}: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA-256 hash of file for integrity checking"""
        import hashlib

        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error hashing file {file_path}: {e}")
            return ""

    def register_pdf(self, file_path: Path) -> ProcessingStatus:
        """
        Register a PDF file for processing.

        Args:
            file_path: Path to PDF file

        Returns:
            ProcessingStatus object for the file
        """
        filename = file_path.name
        file_hash = self._get_file_hash(file_path)
        current_time = datetime.now().isoformat()

        # Check if file is already registered
        if filename in self.progress_data:
            existing_status = self.progress_data[filename]

            # Check if file has changed (different hash)
            if existing_status.file_hash != file_hash:
                print(f"File {filename} has changed, resetting progress")
                # File changed, reset progress
                status = ProcessingStatus(
                    filename=filename,
                    file_path=str(file_path),
                    file_size=file_path.stat().st_size,
                    file_hash=file_hash,
                    current_stage=ProcessingStage.PDF_EXTRACTION,
                    completed_stages=[],
                    start_time=current_time,
                    last_updated=current_time,
                )
            else:
                # File unchanged, return existing status
                return existing_status
        else:
            # New file, create new status
            status = ProcessingStatus(
                filename=filename,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                file_hash=file_hash,
                current_stage=ProcessingStage.PDF_EXTRACTION,
                completed_stages=[],
                start_time=current_time,
                last_updated=current_time,
            )

        self.progress_data[filename] = status
        self.save_progress()
        return status

    def update_stage(
        self,
        filename: str,
        new_stage: ProcessingStage,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update processing stage for a file.

        Args:
            filename: Name of the file
            new_stage: New processing stage
            error_message: Error message if stage failed
            metadata: Additional metadata to store
        """
        if filename not in self.progress_data:
            raise ValueError(f"File {filename} not registered")

        status = self.progress_data[filename]

        # Mark current stage as completed if successful
        if (
            error_message is None
            and status.current_stage not in status.completed_stages
        ):
            status.completed_stages.append(status.current_stage)

        # Update to new stage
        status.current_stage = new_stage
        status.last_updated = datetime.now().isoformat()
        status.error_message = error_message

        # Update metadata
        if metadata:
            status.metadata.update(metadata)

        self.save_progress()

    def get_files_by_stage(self, stage: ProcessingStage) -> List[ProcessingStatus]:
        """Get all files currently at a specific processing stage"""
        return [
            status
            for status in self.progress_data.values()
            if status.current_stage == stage and status.error_message is None
        ]

    def get_failed_files(self) -> List[ProcessingStatus]:
        """Get all files that have failed processing"""
        return [
            status
            for status in self.progress_data.values()
            if status.error_message is not None
        ]

    def get_completed_files(self) -> List[ProcessingStatus]:
        """Get all files that have completed all processing stages"""
        return [
            status
            for status in self.progress_data.values()
            if status.current_stage == ProcessingStage.COMPLETED
        ]

    def get_pending_files(self, pdf_directory: Path) -> List[Path]:
        """
        Get list of PDF files that haven't been processed yet.

        Args:
            pdf_directory: Directory containing PDF files

        Returns:
            List of PDF file paths not yet processed
        """
        if not pdf_directory.exists():
            return []

        # Get all PDF files in directory
        pdf_files = list(pdf_directory.glob("*.pdf")) + list(
            pdf_directory.glob("*.PDF")
        )

        # Filter out already processed files
        pending_files = []
        for pdf_file in pdf_files:
            if pdf_file.name not in self.progress_data:
                pending_files.append(pdf_file)
            else:
                # Check if file needs reprocessing (different hash)
                status = self.progress_data[pdf_file.name]
                current_hash = self._get_file_hash(pdf_file)
                if status.file_hash != current_hash:
                    pending_files.append(pdf_file)
                elif status.error_message is not None:
                    # Include failed files for retry
                    pending_files.append(pdf_file)

        return pending_files

    def print_summary(self) -> None:
        """Print a summary of processing progress"""
        if not self.progress_data:
            print("No files tracked yet")
            return

        stage_counts = {}
        error_count = 0

        for status in self.progress_data.values():
            if status.error_message:
                error_count += 1
            else:
                stage = status.current_stage
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        print("\nProcessing Progress Summary:")
        print(f"Total files: {len(self.progress_data)}")
        print(f"Failed files: {error_count}")

        for stage in ProcessingStage:
            count = stage_counts.get(stage, 0)
            print(f"{stage.value}: {count}")

        print()

    def reset_failed_files(self) -> None:
        """Reset all failed files to retry processing"""
        for status in self.progress_data.values():
            if status.error_message:
                status.error_message = None
                status.current_stage = ProcessingStage.PDF_EXTRACTION
                status.completed_stages = []
                status.last_updated = datetime.now().isoformat()

        self.save_progress()
        print("Reset all failed files for retry")

    def remove_file(self, filename: str) -> None:
        """Remove a file from tracking"""
        if filename in self.progress_data:
            del self.progress_data[filename]
            self.save_progress()
            print(f"Removed {filename} from tracking")
