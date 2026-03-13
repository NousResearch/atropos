# Capital City Quiz Environment

A simple reinforcement learning environment that trains LLMs to correctly
identify capital cities of countries around the world.

## Overview
- **Task**: Given a country name, output the correct capital city
- **Reward**: Binary — 1.0 for correct, 0.0 for incorrect
- **Dataset**: 30 countries built-in (no external dataset needed)

## Usage
```bash
python environments/community/capital_city_env/capital_city_env.py serve
```

## Author
Enzamk
