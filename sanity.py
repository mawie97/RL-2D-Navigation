import numpy as np
from env.mappo_mujoco_env import MultiAgentSAR


env = MultiAgentSAR(csv_log_path=None, xml_paths=["layouts/base_layout.xml"], render_enabled=True)
obs, info = env.reset()
for t in range(10000):
    # small constant action for both agents
    action = {name: np.array([0.1, 0.1], dtype=np.float32) for name in env.agent_names}
    obs, rew, term, trunc, info = env.step(action)
    env.render()
env.close()
