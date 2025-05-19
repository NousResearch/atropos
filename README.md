# Spatial RL Environment MVP for Nous Hackathon

## 1. Overview

This project is a Minimum Viable Product (MVP) for a spatial reinforcement learning (RL) environment, designed for the Nous RL Hackathon. It features:
- A Python backend using PyBullet for 3D physics simulation.
- An Atropos-style environment structure (`SpatialEnvironmentMVP`) for managing tasks, actions, and scoring.
- Real-time communication of simulation state via WebSockets.
- A 3D frontend visualization built with Three.js.

The primary goal is to create an environment where an LLM can learn to perform spatial reasoning tasks.

## 2. Architecture

- **Backend (`spatial_env.py`):**
    - Manages PyBullet physics.
    - Implements `SpatialEnvironmentMVP` for RL interaction loop.
    - Runs a WebSocket server (`ws://localhost:8765`) to stream scene updates.
    - Includes a CLI `process` mode to generate trajectory data.
- **Frontend (`visualization/` directory):**
    - `index.html`, `main.js`, `style.css`.
    - Renders the 3D scene using Three.js.
    - Connects to the backend WebSocket server for live updates.

## 3. Setup Instructions

This project uses Conda to manage dependencies, which has been effective in resolving PyBullet installation challenges, particularly on macOS systems.

1.  **Prerequisites:**
    *   Miniforge3 (or Anaconda/Miniconda) installed and the `conda` command accessible in your terminal.

2.  **Create and Activate Conda Environment:**
    Open a terminal and run:
    ```bash
    conda create -n pybullet_hack_env python=3.10 -y
    conda activate pybullet_hack_env
    ```

3.  **Install Dependencies:**
    Within the activated `pybullet_hack_env` environment:
    ```bash
    conda install -c conda-forge pybullet websockets numpy -y
    pip install --upgrade pip 
    # Add any other pip-only dependencies if they arise (e.g., specific LLM SDKs)
    ```

## 4. Running the Application (Live Visualization Mode)

1.  **Start the Backend Server:**
    In your terminal, with the `pybullet_hack_env` conda environment activated, navigate to the `spatial_rl_mvp/` project root directory and run:
    ```bash
    python spatial_env.py
    ```
    This will start the PyBullet simulation, the WebSocket server on `ws://localhost:8765`, and automatically run a few interactive demo turns.

2.  **Start the Frontend HTTP Server:**
    Open a **new, separate terminal**. Navigate to the `spatial_rl_mvp/visualization/` directory and run:
    ```bash
    python3 -m http.server 8080 
    # (Or any available port, e.g., 8000. Python 3 is needed.)
    ```

3.  **View in Browser:**
    Open your web browser and navigate to `http://localhost:8080` (or the port you used for the HTTP server). You should see the 3D visualization connect and update in real-time as the backend demo turns run.

## 5. Generating Trajectory Data (CLI `process` mode)

To generate example trajectory data using the mock LLM agent:
1. Ensure the `pybullet_hack_env` conda environment is activated.
2. Navigate to the `spatial_rl_mvp/` project root directory.
3. Run the following command:
   ```bash
   python spatial_env.py process --num_turns 5 --output_file trajectories.jsonl
   ```
   - `--num_turns`: Specify the number of trajectory episodes to generate. Defaults to 5.
   - `--output_file`: Specify the file to save the data. Defaults to `trajectories.jsonl`.

   Each line in the output file will be a JSON object representing the data collected for one turn/episode.
   **Note:** In this mode, the WebSocket server for live visualization is NOT started. Physics simulation runs in the background to generate data. 