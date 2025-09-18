# Translater between the env and MAPPO
import numpy as np

class SAROnPolicyWrapper:
    def __init__(self, base_env, agent_names):
        self.env = base_env
        self.agent_names = list(agent_names)
        self.n_agents = len(self.agent_names)
        obs_dict, info = self.env.reset()
        self.obs_dim = obs_dict[self.agent_names[0]].shape[0]
        self.state_dim = info["global_state"].shape[0]

    def reset(self):
        obs_dict, info = self.env.reset()
        obs = np.stack([obs_dict[n] for n in self.agent_names], axis=0).astype(np.float32)
        share_obs = np.repeat(info["global_state"][None, :], self.n_agents, axis=0).astype(np.float32)
        return obs, share_obs, None

    def step(self, actions):
        act_dict = {n: actions[i] for i, n in enumerate(self.agent_names)}
        obs_dict, rew_dict, terminated, truncated, info = self.env.step(act_dict)
        done = bool(any(terminated.values()) or any(truncated.values()))
        obs = np.stack([obs_dict[n] for n in self.agent_names], axis=0).astype(np.float32)
        share_obs = np.repeat(info["global_state"][None, :], self.n_agents, axis=0).astype(np.float32)
        rew = np.array([rew_dict[n] for n in self.agent_names], dtype=np.float32)
        dones = np.array([done] * self.n_agents, dtype=np.bool_)
        infos = [{} for _ in range(self.n_agents)]
        return obs, rew, dones, infos, share_obs, None

    def close(self):
        self.env.close()
