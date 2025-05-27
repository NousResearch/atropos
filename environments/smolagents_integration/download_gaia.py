#!/usr/bin/env python3
"""
Script to download the GAIA benchmark dataset from HuggingFace.
This script follows a strict success/fail policy with no fallbacks to mock data.
"""

import argparse
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Download GAIA benchmark dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/gaia",
        help="Directory to store GAIA dataset",
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use raw dataset instead of annotated version",
    )
    return parser.parse_args()


def download_gaia_dataset(output_dir, use_raw=False):
    """
    Download the GAIA benchmark dataset from HuggingFace.
    Fails explicitly if download is unsuccessful - no fallbacks to mock data.
    """
    try:
        # Check for required packages
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error(
                "Required packages not installed. Run: pip install huggingface_hub"
            )
            return False

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Determine which repo to use
        repo_id = "gaia-benchmark/GAIA" if use_raw else "smolagents/GAIA-annotated"
        logger.info(f"Downloading GAIA dataset from {repo_id}...")

        # Download the dataset (which is gated/private)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            ignore_patterns=[".gitattributes", "README.md"],
        )

        # Create a minimal GAIA.py that loads directly from metadata.jsonl using absolute paths
        with open(os.path.join(output_dir, "GAIA.py"), "w") as f:
            f.write(
                f'''
"""
GAIA benchmark dataset loader.
Loads data directly from metadata.jsonl files.
"""

import os
import json
import datasets

# Define absolute path to the dataset directory - crucial for correct operation
DATASET_PATH = "{os.path.abspath(output_dir)}"

class GAIA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2023.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="2023_all",
            version=VERSION,
            description="GAIA 2023 benchmark",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="GAIA benchmark dataset",
            features=datasets.Features({{
                "Question": datasets.Value("string"),
                "Final answer": datasets.Value("string"),
                "Level": datasets.Value("string"),
                "task_id": datasets.Value("string"),
                "file_name": datasets.Value("string"),
            }}),
            homepage="https://huggingface.co/datasets/gaia-benchmark/GAIA",
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={{"split": "validation"}},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={{"split": "test"}},
            ),
        ]

    def _generate_examples(self, split):
        """Read data from the metadata.jsonl file."""
        # Use absolute path to the metadata.jsonl file
        metadata_path = os.path.join(DATASET_PATH, "2023", split, "metadata.jsonl")
        print(f"Loading GAIA data from: {{metadata_path}}")

        with open(metadata_path, "r") as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                if "file_name" in example and example["file_name"]:
                    # Ensure file paths include the 2023 directory and absolute path
                    example["file_name"] = os.path.join("2023", split, example["file_name"])
                yield i, example
'''
            )

        # Verify the download worked by checking for key files
        validation_path = os.path.join(
            output_dir, "2023", "validation", "metadata.jsonl"
        )
        if not os.path.exists(validation_path):
            logger.error(f"Download appears incomplete. Missing: {validation_path}")
            return False

        logger.info(f"Dataset downloaded successfully to {output_dir}")
        return True

    except Exception as e:
        logger.error(f"Error downloading GAIA dataset: {e}")
        return False


def main():
    args = parse_args()
    success = download_gaia_dataset(args.output_dir, use_raw=args.use_raw)

    if success:
        logger.info("GAIA dataset setup completed successfully")
        logger.info(
            "You can now run: python -m environments.smolagents_integration.run_gaia_single_task"
        )
    else:
        logger.error("GAIA dataset setup failed - Aborting")
        logger.error(
            "You need to download the GAIA dataset to use the SmolaGents integration."
        )
        logger.error(
            "Please ensure you have access to the GAIA dataset on HuggingFace and try again."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
