# AI Research Pipeline

This project implements an automated AI research pipeline on Google Cloud Platform (GCP).

The existing **Padres Spatial RL Environment** will serve as a core component for running experiments.

## Project Vision

The goal is to systematically conduct experiments, analyze results, generate insights, and iteratively improve AI models (particularly LLMs) for complex tasks like spatial reasoning. The pipeline will automate many aspects of this research lifecycle.

## Key Planned Components & Technologies:

*   **Core Infrastructure (GCP):**
    *   Docker & Cloud Run: For containerizing and orchestrating Padres and other services.
    *   BigQuery & Firestore: For data storage (trial results, configurations).
    *   Pub/Sub: For event-driven architecture.
    *   Cloud Functions: For serverless tasks.
    *   Cloud Storage & Artifact Registry: For artifacts and images.
*   **Backend Services (Python, FastAPI):
    *   Central FastAPI API: For managing and controlling the pipeline.
    *   MCP (Multi-Controller Piper) Framework: For modular interaction with LLMs and other tools.
    *   Experiment Orchestrator: To manage the execution of experiment trials.
*   **Research Automation (Python, Cloud Functions):
    *   Research Assistant: To analyze results and generate hypotheses.
    *   Analysis Engine: For in-depth data analysis.
    *   Report Generator: To automate report creation and notifications.

## Directory Structure

- `padres_container/`: Dockerization for the Padres environment.
- `backend_services/`: Central FastAPI service and MCP framework integration.
  - `mcp_servers/`: Specific MCP server implementations.
- `research_automation/`: Components for research assistance, analysis, and reporting.
- `config/`: Configuration files for experiments, services, etc.
- `scripts/`: Utility scripts for deployment, data management, etc.
- `docs/`: Project documentation.
- `spatial_rl_mvp/`: Contains the existing Padres Spatial RL Environment code.
  - `visualization/`: Frontend for Padres environment.
  - `spatial_env.py`: Core Padres backend logic.

*(More details on each component can be found in the respective directories as they are developed.)* 