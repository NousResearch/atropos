# Padres: Spatial RL Environment with PyBullet, LLM, and W&B

## Video Demo
[Watch the demo video](https://youtu.be/uuSur31U1Pc)

## Project Overview
This project implements a spatial reasoning environment using PyBullet for physics simulation, integrated with LLM-based task generation and W&B for experiment tracking.

## Key Features
- PyBullet-based 3D physics environment
- LLM-powered task generation and execution
- Real-time visualization
- W&B integration for experiment tracking

## Setup
1. Install dependencies:
```bash
conda env create -f environment.yml
conda activate spatial_rl_conda_env
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

3. Run the environment:
```bash
python -m spatial_rl_mvp.spatial_env
```

4. Open visualization:
```bash
cd spatial_rl_mvp/visualization
python3 -m http.server 8080
```
Then visit http://localhost:8080 in your browser.

## W&B Integration
The project uses Weights & Biases for experiment tracking. View the latest run [here](https://wandb.ai/carlosgarcia/spatial_rl_mvp/runs/1q2w3e4r5t6y7u8i9o0p).

## Project Structure
- `spatial_rl_mvp/`: Core environment implementation
- `visualization/`: Web-based visualization
- `environment.yml`: Conda environment specification
- `.env.example`: Environment variables template 