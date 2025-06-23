#!/usr/bin/env python3
"""
Setup script for 2,500-Agent LLM Society Simulation
"""

import os

from setuptools import find_packages, setup


# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_path, "r") as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


# Read long description from README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "2,500-Agent LLM-Driven Society Simulation"


setup(
    name="llm-society-simulation",
    version="0.1.0",
    description="A 2,500-agent, fully 3D, LLM-driven society simulation using Mesa-frames, FLAME GPU, and Atropos",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Research Team",
    author_email="research@example.com",
    url="https://github.com/research-team/llm-society-simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "nvidia-ml-py>=11.0.0",
        ],
        "cloud": [
            "google-cloud-storage>=2.10.0",
            "google-cloud-aiplatform>=1.30.0",
            "google-cloud-logging>=3.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    entry_points={
        "console_scripts": [
            "llm-society=src.main:main",
            "llm-society-dashboard=src.dashboard:main",
            "llm-society-monitor=src.monitoring:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
