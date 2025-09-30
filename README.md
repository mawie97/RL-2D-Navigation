# Multi-Agent SAR with RLlib and MuJoCo

This repository demonstrates how to train a centralized-critic PPO agent on an interactive multi-agent search-and-rescue (SAR) simulation built in **MuJoCo**.  

The environment contains two agents that move in a plane, cast rays to detect obstacles/targets, and share a **team-level reward**.  

Training uses **Ray’s RLlib** library with a centralized value function (a.k.a. CTDE / MAPPO style) while each agent’s actor observes only its own ray-casting readings.

---

## Directory structure
```
Thesis_Project/
├── env/
│   ├── mappo_mujoco_env.py   # MuJoCo environment implementing the multi‑agent SAR task
├── cc_model.py               # Custom PyTorch model implementing a centralized critic and decentralized actors
├── env_rllib_wrapper.py      # Helper for registering the environment with RLlib
├── train_rllib_mappo.py      # Script to configure and run PPO training with RLlib
├── layouts/
│   └── base_layout.xml       # MuJoCo XML model describing the map, agents and targets
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview (this file)
```


---

## `env/mappo_mujoco_env.py`

Defines the underlying multi-agent environment.  
This file builds a `MultiAgentSAR` class derived from RLlib’s `MultiAgentEnv`. It wraps a MuJoCo world described by `layouts/base_layout.xml` and provides:

- **Multi-agent observations**  
  Each agent sees a 12-dimensional vector of ray-casting distances to nearby obstacles and targets.  
  Observations are keyed by agent names (`"agent_1"`, `"agent_2"`).

- **Continuous actions**  
  Agents control two slide joints (x and y).  
  Actions are 2-D vectors bounded by ±0.1.

- **Global state**  
  At every step, the environment concatenates all agents’ observations into  
  `info["common"]["global_state"]` for the centralized critic.

- **Rewards & termination**  
  - Positive reward when any agent’s rays detect a new target.  
  - Episode ends when all targets are found, max steps are reached, or a collision occurs.

---

## `env_rllib_wrapper.py`

Registers the environment with RLlib.  

RLlib expects envs to be created through a **factory** that accepts a config dict. This file defines:

- `make_env_creator(default_csv=None, default_xmls=None)`  
  Returns a closure that instantiates `MultiAgentSAR` with values from `env_config`.

- `register_sar_env(name="MultiAgentSAR-v0", csv=None, xmls=None)`  
  Registers the closure with RLlib’s registry and returns the name.

The training script calls `register_sar_env()` before running PPO.

---

## `cc_model.py`

Custom Torch model implementing a centralized critic and decentralized actors.  

Defines `TorchCCModel` (inherits from `TorchModelV2`) with two sub-networks:

- **Actor head**  
  - Consumes *per-agent* observations.  
  - Outputs action logits (wrapped by RLlib into Gaussian distributions).  
  - Shared across all agents, but each uses only its own obs.

- **Critic head**  
  - Consumes `global_state` (`info["common"]["global_state"]`).  
  - Outputs a scalar value → centralized training with decentralized actors.

The model is registered with RLlib’s `ModelCatalog` under `"torch_cc_model"`.

---

## `train_rllib_mappo.py`

Configures and runs PPO training using RLlib.  

### Training Setup Steps

1. **Initialise Ray**  
   Calls `ray.init()` to start a local Ray cluster.

2. **Register the environment**  
   Calls `register_sar_env()` from `env_rllib_wrapper` to register `MultiAgentSAR` under the name `"MultiAgentSAR-v0"`.

3. **Define observation/action spaces**  
   For multi-agent RLlib training you need to specify the shapes of per-agent observation and action spaces:  
   - `single_obs_space` → 12-D `Box`  
   - `single_act_space` → 2-D `Box`

4. **Build a PPOConfig**  
   Creates a `PPOConfig()` and sets:

   - `.framework("torch")` → use PyTorch  
   - `.environment()` → use the registered environment with `env_config` (e.g., XML path, log file)  
   - `render_enabled=True` will render the environment
   - `.multi_agent()` → one shared policy for all agents (map every agent id to the same policy)  
   - `.training()` → PPO hyper-parameters + custom model (`"torch_cc_model"`) with `global_state_dim`  
   - `.resources()` → allocate CPU/GPU resources  
   - `.env_runners()` → set number of parallel environment processes & rollout fragment length  
   - `.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)` → use the **classic RLlib API stack** (required when using ModelV2 custom models)
---
# Training and Usage

## Training Loop

- Builds the PPO algorithm via `config.build_algo()`  
- Runs training for **200 iterations**  
- Prints the **mean episode reward**  
- Periodically saves **checkpoints**

RLlib also supports periodic evaluation via `.evaluation()` on the config to run episodes with rendering on the driver process.

---

## Installation

Install Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
---

## Running Training

After installing dependencies, run the training script:

```bash
python train_rllib_mappo.py
```
The script will start **Ray**, build the PPO agent, and begin training.  
Rewards and checkpoints will be printed to the console.  

⚠️ Rendering notes:  
- If `render_enabled=True` and `num_env_runners > 1`, multiple MuJoCo windows may open.  
- For a **single window**, either:  
  - Set `num_env_runners=0` or `1`, or  
  - Use RLlib’s **evaluation feature** to render separate episodes.

---
