# CloudVR-PerfGuard

This project implements CloudVR-PerfGuard, a system for continuous AI-driven research on VR performance data.

## Prerequisites

- Docker installed and running.
- A Gemini API key.

## Setup

1.  **Environment Variables**:
    Create a `.env` file in the `cloudvr_perfguard` directory with your Gemini API key:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

2.  **Configuration (Optional)**:
    The continuous research pipeline uses `research_config.json` for its settings. A default configuration will be created if one doesn't exist. You can customize this file as needed.

## Building and Running the Docker Container

1.  **Build the Docker image**:
    Navigate to the root of the NOUS project directory (the one containing the `cloudvr_perfguard` folder) and run:
    ```bash
    docker build -t cloudvr-perfguard -f cloudvr_perfguard/Dockerfile .
    ```

2.  **Run the Docker container**:
    ```bash
    docker run -d --name cloudvr-perfguard-container -v $(pwd)/cloudvr_perfguard/database:/app/database -v $(pwd)/cloudvr_perfguard/.env:/app/.env cloudvr-perfguard
    ```
    This command will:
    - Run the container in detached mode (`-d`).
    - Mount the local `cloudvr_perfguard/database` directory into the container at `/app/database` to persist the database.
    - Mount the local `.env` file into the container for API key access.

## Managing the Pipeline

The `continuous_research_pipeline.py` script provides a command-line interface for managing the pipeline. You can interact with it using `docker exec`.

**Examples**:

-   **Check status**:
    ```bash
    docker exec cloudvr-perfguard-container python cloudvr_perfguard/ai_integration/continuous_research_pipeline.py status
    ```

-   **Trigger a daily run manually**:
    ```bash
    docker exec cloudvr-perfguard-container python cloudvr_perfguard/ai_integration/continuous_research_pipeline.py run-daily
    ```

-   **Stop the pipeline gracefully**:
    ```bash
    docker exec cloudvr-perfguard-container python cloudvr_perfguard/ai_integration/continuous_research_pipeline.py stop
    ```
    (Note: The `stop` command currently signals the pipeline to stop. The container itself will need to be stopped using `docker stop cloudvr-perfguard-container` if it was started with a command that keeps it running, like the default CMD in the Dockerfile.)

## Directory Structure (within `cloudvr_perfguard`)

-   `ai_integration/`: Core AI integration modules, including the continuous research pipeline.
-   `core/`: Core functionalities like database management, performance testing, etc.
-   `database/`: Stores the SQLite database (`cloudvr_perfguard.db`).
-   `logs/`: Stores pipeline logs.
-   `research_outputs/`: Stores generated papers and functions.
-   `scripts/`: Utility scripts, e.g., for populating test data.
-   `api/`: FastAPI application for potential API interactions (if developed further).
-   `Dockerfile`: For building the Docker image.
-   `requirements.txt`: Python dependencies.
-   `.env`: For storing environment variables (e.g., API keys).
-   `research_config.json`: Configuration for the continuous research pipeline.

## Development

For local development outside of Docker:

1.  Ensure you have Python 3.10+ installed.
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r cloudvr_perfguard/requirements.txt
    ```
4.  Set up your `.env` file as described in the Setup section.
5.  You can then run scripts directly, e.g.:
    ```bash
    python cloudvr_perfguard/ai_integration/continuous_research_pipeline.py start
    ```
