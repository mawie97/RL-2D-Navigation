# Thesis_Project

# Setup
In another folder:
git clone https://github.com/marlbenchmark/on-policy.git
pip install -e .

pip install -r requirements.txt


# 1. Environment (MultiAgentSAR)

Talks directly to MuJoCo.

Defines physics, obstacles, agent positions, rewards, etc.

Gives you per-agent obs, rewards, dones, info.

This is your “world”.

# 2. Wrapper (SAROnPolicyWrapper)

Purpose: translate env I/O into the exact format the algorithm expects.

Why needed? MultiAgentSAR returns dicts like:

```python
{
    "agent_0": obs0,
    "agent_1": obs1
}, info = {"global_state": ...}
```

but the MAPPO code expects:

```python
obs = np.array([[...], [...]])               # shape (n_agents, obs_dim)
share_obs = np.array([[...], [...]])         # shape (n_agents, state_dim)
action_space = [Box(...), Box(...)]          # list of spaces
```

So the wrapper:

Stacks dicts into arrays.

Repeats the global state across agents.

Exposes observation_space, share_observation_space, and action_space as lists.

Think of it as a translator.

# 3. Runner (Runner / sar_runner.py)

Purpose: drive the training loop.

It’s the “coach”:

Calls env.reset(), env.step(actions) in a loop.

Collects rollouts into a buffer.

Computes advantages/returns.

Calls the algorithm’s train() to update networks.

Handles logging + saving checkpoints.

Without the runner, you’d have to write the full PPO training loop yourself.

# 4. Algorithm (MAPPO)

The actual brain: neural nets + PPO update.

Gets data from the runner (obs, actions, rewards, advantages).

Updates policy and critic networks.

Pure math + optimization — doesn’t know about MuJoCo.

🔗 So the flow is:
MultiAgentSAR (raw MuJoCo) 
   ↓
SAROnPolicyWrapper (translates dict → arrays, exposes spaces)
   ↓
Runner (collect rollouts, compute returns/advantages, call train)
   ↓
MAPPO (actor-critic networks, PPO updates in PyTorch)



Wrapper = translates env data into algorithm-ready format.

Runner = defines the training loop and connects algorithm ↔ env.
