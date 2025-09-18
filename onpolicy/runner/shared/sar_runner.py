import numpy as np
import torch

from .base_runner import Runner as BaseRunner

def _np(x):  # ensure numpy
    return x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()

class Runner(BaseRunner):
    """
    Minimal MAPPO runner for single-thread, fully cooperative envs.
    Assumes your env wrapper returns:
      reset() -> (obs:(N,obs_dim), share_obs:(N,state_dim), None)
      step(a:(N,act_dim)) -> (obs, rew:(N,), dones:(N,), infos:[N*{}], share_obs, None)
    Works with use_recurrent_policy=False (no RNN), CTDE on/off.
    """

    def run(self):
        self.warmup()
        total_num_steps = 0
        episode = 0

        while total_num_steps < self.num_env_steps:
            ep_env_info = {}

            # collect one full rollout
            for step in range(self.episode_length):
                data = self.collect(step)   # asks policy for actions, steps env
                self.insert(data)           # writes into SharedReplayBuffer
                total_num_steps += self.n_rollout_threads * self.num_agents

                # optional: aggregate env stats here if you push them via infos

            # compute returns (GAE) with critic bootstrap & train PPO
            self.compute()
            train_infos = self.train()

            # logging
            if (episode % self.log_interval) == 0:
                self.log_train(train_infos, total_num_steps)
                if ep_env_info:
                    self.log_env(ep_env_info, total_num_steps)

            # checkpoint
            if (episode % self.save_interval) == 0:
                self.save(episode)

            episode += 1

        # final save
        self.save(episode)

    def warmup(self):
        """
        Seeds buffer[t=0] with initial obs/state and masks.
        Shapes in buffer are (T+1, n_threads, n_agents, ...). We use n_threads=1.
        """
        obs, share_obs, _ = self.envs.reset()         # (N,obs_dim), (N,state_dim)
        obs = obs[None, ...]                          # -> (1,N,obs_dim)
        share_obs = share_obs[None, ...]              # -> (1,N,state_dim)

        # masks: 1.0 = alive, 0.0 = done (we start alive)
        masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

        # rnn states placeholders (unused if use_recurrent_policy=False)
        rnn_states = np.zeros(
            (1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        rnn_states_critic = np.zeros_like(rnn_states)

        self.buffer.share_obs[0] = share_obs
        self.buffer.obs[0] = obs
        self.buffer.rnn_states[0] = rnn_states
        self.buffer.rnn_states_critic[0] = rnn_states_critic
        self.buffer.masks[0] = masks

    @torch.no_grad()
    def collect(self, step):
        """
        Uses current policy to produce actions, steps env once,
        and returns everything needed for buffer.insert().
        """
        self.trainer.prep_rollout()

        # fetch last obs & states from buffer (thread dim=1)
        obs = self.buffer.obs[step]                  # (1,N,obs_dim)
        share_obs = self.buffer.share_obs[step]      # (1,N,state_dim)
        rnn_states = self.buffer.rnn_states[step]    # (1,N,rnn,hidden)
        rnn_states_critic = self.buffer.rnn_states_critic[step]
        masks = self.buffer.masks[step]              # (1,N,1)

        # policy calls expect flat batch
        B = obs.shape[0] * obs.shape[1]              # = 1 * N
        obs_flat = obs.reshape(B, -1)
        share_flat = share_obs.reshape(B, -1)
        rnn_flat = rnn_states.reshape(B, rnn_states.shape[-2], rnn_states.shape[-1])
        mask_flat = masks.reshape(B, 1)

        # critic values for bootstrap
        if self.algorithm_name in ["mat", "mat_dec"]:
            values = self.policy.get_values(share_flat, obs_flat, rnn_flat, mask_flat)
        else:
            values = self.policy.get_values(share_flat, rnn_flat, mask_flat)

        # actions & logprobs from actor
        actions, logprobs = self.policy.get_actions(obs_flat, rnn_flat, mask_flat)

        # reshape back to (1,N,dim)
        values = _np(values).reshape(1, self.num_agents, -1)
        actions = _np(actions).reshape(1, self.num_agents, -1)
        logprobs = _np(logprobs).reshape(1, self.num_agents, -1)

        # step environment with per-agent actions (strip thread dim)
        act_np = actions[0]                          # (N, act_dim)
        obs_next, rewards, dones, infos, share_obs_next, _ = self.envs.step(act_np)

        # wrap env outputs with thread dim
        obs_next = obs_next[None, ...]               # (1,N,obs_dim)
        share_obs_next = share_obs_next[None, ...]   # (1,N,state_dim)
        rewards = np.asarray(rewards, dtype=np.float32)[None, :, None]  # (1,N,1)

        # masks for next step
        # PPO expects masks=0 where episode ended so RNN can reset; we broadcast team-done to all agents
        done_flag = bool(np.any(dones))
        next_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        if done_flag:
            next_masks[:] = 0.0

        data = {
            "obs": obs_next,
            "share_obs": share_obs_next,
            "rewards": rewards,
            "dones": next_masks,            # used both as masks & active_masks
            "values": values,
            "actions": actions,
            "logprobs": logprobs,
            "rnn_states": np.zeros_like(rnn_states),
            "rnn_states_critic": np.zeros_like(rnn_states_critic),
        }
        return data

    def insert(self, data):
        """
        Writes one timestep into SharedReplayBuffer.
        Expected order:
          insert(share_obs, obs, rnn_states, rnn_states_critic,
                 actions, value_preds, action_log_probs, rewards,
                 masks, active_masks)
        """
        self.buffer.insert(
            data["share_obs"],
            data["obs"],
            data["rnn_states"],
            data["rnn_states_critic"],
            data["actions"],
            data["values"],
            data["logprobs"],
            data["rewards"],
            data["dones"],          # masks
            data["dones"],          # active masks (coop => same as masks)
        )
