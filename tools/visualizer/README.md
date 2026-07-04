# Atropos Interactive Trajectory Visualizer & Debugger Dashboard

A modern visual debugging tool designed to load, step-through, and play back reinforcement learning trajectories in real-time.

## Features
- **Visual Trajectory Player**: Play forward, backward, or click to any step.
- **Thinking Analyzer**: Renders deep `<think>` chain-of-thought blocks with glowing visual cues.
- **Environment Hub**: Select any active community environment and simulate runs.
- **Live Metrics Charting**: Visually trace rewards and token lengths using integrated chart renderers.

## Running the Dashboard
```bash
pip install fastapi uvicorn chart.js
uvicorn tools.visualizer.main:app --reload --port 8500
```
Then navigate to `http://localhost:8500` in your web browser.
