# cc_rlmodule.py
import torch
import torch.nn as nn
from typing import Dict
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.policy.sample_batch import SampleBatch

class CCPPOConfig(RLModuleConfig):
    def __init__(self, local_dim: int, global_dim: int, act_dim: int,
                 pi_sizes=(128, 128), vf_sizes=(256, 256), activation="relu", **kw):
        super().__init__(**kw)
        self.local_dim = int(local_dim)
        self.global_dim = int(global_dim)
        self.act_dim = int(act_dim)
        self.pi_sizes = tuple(pi_sizes)
        self.vf_sizes = tuple(vf_sizes)
        self.activation = activation

def _act(name: str):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "gelu": nn.GELU}[name]()

class CCPPOTorchModule(TorchRLModule):
    """
    PPO RLModule:
      - obs is a Dict: {"obs": [B, local_dim], "state": [B, global_dim]}
      - Actor sees only local obs -> mean
      - Global learned log_std (free log std)
      - Critic sees only global state -> value
      - Outputs:
          action_dist_inputs = concat(mean, log_std)  [B, 2*act_dim]
          vf_preds           = value                  [B]
    """
    def __init__(self, config: CCPPOConfig):
        super().__init__(config)
        self.cfg: CCPPOConfig = config

        A = _act(self.cfg.activation)

        # Actor on local obs
        pi = []
        last = self.cfg.local_dim
        for h in self.cfg.pi_sizes:
            pi += [nn.Linear(last, h), A]
            last = h
        self.actor = nn.Sequential(*pi)
        self.mean_head = nn.Linear(last, self.cfg.act_dim)
        # free log std parameter (global, not state-dependent)
        self.log_std = nn.Parameter(torch.zeros(self.cfg.act_dim))

        # Critic on global state
        vf = []
        last = self.cfg.global_dim
        for h in self.cfg.vf_sizes:
            vf += [nn.Linear(last, h), A]
            last = h
        self.critic = nn.Sequential(*vf, nn.Linear(last, 1))

    # ---- three paths required by RLModule ----
    def forward_inference(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._forward_common(batch)

    def forward_exploration(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._forward_common(batch)

    def forward_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._forward_common(batch)

    # ---- shared head ----
    def _forward_common(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = batch[SampleBatch.OBS]         # dict with keys "obs", "state"
        o_local  = obs["obs"].float()
        o_global = obs["state"].float()

        h = self.actor(o_local)
        mean = self.mean_head(h)
        log_std = self.log_std.expand_as(mean)
        logits = torch.cat([mean, log_std], dim=-1)   # [B, 2*act_dim]

        value = self.critic(o_global).squeeze(-1)     # [B]

        return {
            "action_dist_inputs": logits,
            "vf_preds": value,
        }
