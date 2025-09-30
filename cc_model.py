# This defines a PyTorch network that separates the actor 
# and critic. The actor network uses only the agent’s 
# local observation, whereas the critic network uses the 
# global state. The model caches the global state passed 
# through the info dictionary and exposes it to RLlib’s 
# value function.
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.catalog import ModelCatalog
from collections import OrderedDict

class TorchCCModel(TorchModelV2, nn.Module):
    """
    Actor: uses own obs (obs).
    Critic: uses infos["global_state"].
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Use np.prod (NumPy 2.0+ removed some aliases)
        obs_size = int(np.prod(obs_space.shape))

        # Read from custom_model_config (DO NOT accept as kwarg)
        cmc = model_config.get("custom_model_config", {})
        self.gs_dim = cmc.get("global_state_dim")
        assert self.gs_dim is not None, "Set custom_model_config.global_state_dim in PPO config"

        # Actor head (from local obs)
        self.actor = nn.Sequential(
            nn.Linear(obs_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        # Critic head (from global_state)
        self.critic = nn.Sequential(
            nn.Linear(self.gs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        self._vf_in = None  # cached global_state for value_function()

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]

        # Flatten if dict-like (safety)
        if isinstance(x, (dict, OrderedDict)):
            parts = []
            for v in x.values():
                t = v if torch.is_tensor(v) else torch.as_tensor(v)
                parts.append(t.float())
            x = torch.cat(parts, dim=-1)
        else:
            x = x.float()

        logits = self.actor(x)

        # ---- Robust global_state fetch ----
        gs = None
        infos = input_dict.get("infos", None)

        # (A) If RLlib already materialized it to a top-level column
        if "global_state" in input_dict:
            gs = input_dict["global_state"]

        # (B) Classic path: dict of tensors
        if gs is None and isinstance(infos, dict):
            gs = infos.get("global_state", None)

        # (C) Packed tensor path: infos is a Tensor (assume it's our global_state)
        if gs is None and torch.is_tensor(infos):
            if infos.dim() >= 2 and infos.shape[-1] == self.gs_dim:
                gs = infos

        # Fallback for eval or missing wiring
        if gs is None:
            gs = torch.zeros((x.shape[0], self.gs_dim), device=x.device)

        self._vf_in = gs.float()
        return logits, state

    def value_function(self):
        return self.critic(self._vf_in).squeeze(-1)

# Important: register AFTER class definition
ModelCatalog.register_custom_model("torch_cc_model", TorchCCModel)
