# quick_test.py
from env.MultiAgentSAR import MultiAgentSAR
from sar_adapter.sar_env import SAROnPolicyWrapper
from pathlib import Path
import numpy as np

XML_DIR = Path(__file__).parent / "layouts"
xml_path = str((XML_DIR / "base_layout.xml").resolve())

env = MultiAgentSAR(None, [xml_path], agent_names=("agent_0","agent_1"))

wrap = SAROnPolicyWrapper(env, ["agent_0","agent_1"])

obs, share_obs, _ = wrap.reset()
print("obs", obs.shape, "share_obs", share_obs.shape)

act_dim = env.action_space["agent_0"].shape[0]
a = np.zeros((2, act_dim), np.float32)
obs, rew, dones, infos, share_obs, _ = wrap.step(a)
print("step ok", obs.shape, rew.shape, dones.shape, share_obs.shape)
