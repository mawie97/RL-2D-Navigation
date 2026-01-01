# MuJoCo Navigation with Adaptive Reward Switching

This repository contains a reinforcement learning framework for continuous 2D navigation in MuJoCo. The project studies how procedurally generated environment layouts and adaptive reward mechanisms affect an agent’s ability to navigate toward a goal while handling challenging structures such as obstacles and dead-ends.

Environments are generated as grid-based layouts and converted into MuJoCo XML models. A PPO agent is trained using ray-based distance sensing to reach a target position. When the agent becomes stuck, an adaptive reward switching mechanism activates alternative reward signals, including openness-based and backtracking rewards, to encourage recovery behavior.

The codebase supports training, evaluation, and analysis across multiple environment layouts and is designed to facilitate controlled experiments on navigation generalization and curriculum-based scenario generation. This repository accompanies an academic thesis and is intended for research and reproducibility purposes.

---

## How to Run

### 1. Install dependencies

Create a virtual environment and install required packages using the provided `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Configure MuJoCo rendering

For local execution with visualization, no additional configuration is required.

### 3. Prepare environment layouts

Training and evaluation environments are provided as MuJoCo XML files and are expected in the following directories:


`navppo/layouts/train/`  - training environments
`navppo/layouts/eval/`          - evaluation environments


Ensure these directories exist and contain valid .xml files.

### 4. Train the agent
`python run_train_env.py`


This script iterates over training environments and saves trained models, logs, and normalization statistics under the runs/ directory.

### 5. Evaluate a trained model
`python evaluate_trained_model.py`


The evaluation script loads a trained policy from runs/ and evaluates it on held-out environments.

### 6. Scenario Generation

Environment layouts are procedurally generated on a discrete grid and compiled into MuJoCo XML models before training and evaluation. Multiple generation strategies are implemented to support different levels of structural complexity.

`xml_generator.py`
High-level interface for generating grid-based environment layouts.

`xml_writer.py`
Converts grid-based layouts into MuJoCo XML models by mapping grid cells to continuous object poses.

`generator_bresenham.py`
Baseline generator using random obstacle placement with Bresenham-style path constraints.

`generator_hybrid.py`
Hybrid generator combining structured constraints (e.g., corridors, dead-ends) with controlled randomness.

`generator_solver.py`
Constraint-based generator that enforces higher-level structural properties during layout generation.

These generators are used to create training and evaluation environments with varying degrees of topological complexity.

