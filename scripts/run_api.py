#!/usr/bin/env python3
"""
A script to run the FastAPI server for the research pipeline API
"""

import os
import sys
from pathlib import Path

import uvicorn

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def main():
    # Importing here to leverage the updated sys.path
    from backend_services.main import app

    # Run the FastAPI server
    print("Starting FastAPI server...")
    uvicorn.run("backend_services.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
