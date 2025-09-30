#!/usr/bin/env python

import os

from smolagents import tool


@tool
def read_file(file_path: str) -> str:
    """
    Read content from a file. Prints the contents.

    Args:
        file_path: Path to the file to read

    Returns:
        Content of the file as a string
    """
    with open(file_path, "r") as f:
        content = f.read()
        print(content)
        return content


@tool
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        file_path: Path to the file to write
        content: Content to write to the file

    Returns:
        Confirmation message
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)
    return f"Content written to {file_path}"


@tool
def append_to_file(file_path: str, content: str) -> str:
    """
    Append content to a file.

    Args:
        file_path: Path to the file to append to
        content: Content to append to the file

    Returns:
        Confirmation message
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f:
        f.write(content)
    return f"Content appended to {file_path}"
