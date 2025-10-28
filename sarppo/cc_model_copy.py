import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Dict as DictSpace, Box
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.catalog import ModelCatalog

def _np_str(a):
    # consistent formatting for comparison with env prints
    return np.array2string(
        a, precision=4, suppress_small=True, max_line_width=160, separator=", "
    )

class TorchCCModel(TorchModelV2, nn.Module):
    """
    Actor: uses local obs.
    Critic: uses global state.
    Works with Dict({"obs":(local,), "state":(global,)}) OR flattened Box(local+global).
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Determine dims and whether we expect a Dict or a flat Box:
        self._expects_dict = isinstance(obs_space, DictSpace)
        if self._expects_dict:
            self.local_dim  = int(np.prod(obs_space["obs"].shape))
            self.global_dim = int(np.prod(obs_space["state"].shape))
        elif isinstance(obs_space, Box):
            # RLlib gave us a flattened vector. Get split sizes from custom_model_config.
            cmc = model_config.get("custom_model_config", {})

            self._debug = bool(cmc.get("debug_dump", True))  # turn on/off from PPO config
            self._actor_prints = 0
            self._critic_prints = 0

            if "local_dim" not in cmc or "global_dim" not in cmc:
                raise ValueError(
                    "Flattened Box obs requires custom_model_config with "
                    "`local_dim` and `global_dim` (e.g., 12 and 24)."
                )
            self.local_dim  = int(cmc["local_dim"])
            self.global_dim = int(cmc["global_dim"])
            flat = int(np.prod(obs_space.shape))
            if flat != self.local_dim + self.global_dim:
                raise ValueError(
                    f"Flat obs size ({flat}) != local_dim+global_dim "
                    f"({self.local_dim}+{self.global_dim})."
                )
        else:
            raise ValueError(f"Unsupported obs_space: {obs_space}")

        self.act_dim = int(np.prod(action_space.shape))
        # if num_outputs != self.act_dim:
        #     # Not fatal, but helpful to know.
        #     print(f"[TorchCCModel] Warning: num_outputs({num_outputs}) != action_dim({self.act_dim}).")

        # Actor (local obs)
        self.actor = nn.Sequential(
            nn.Linear(self.local_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
        # Critic (global state)
        self.critic = nn.Sequential(
            nn.Linear(self.global_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self._vf_in = None
        self._printed_once = False

    def forward(self, input_dict, state, seq_lens):
        obs_in = input_dict["obs"]
        if self._expects_dict:
            assert isinstance(obs_in, dict), "Obs must be a Dict with 'obs' and 'state'"
            o_local  = obs_in["obs"].float()                  # [B, local_dim]
            o_global = obs_in["state"].float()                # [B, global_dim]
            if not self._printed_once:
                # print(f"[CCModel] local shape={tuple(o_local.shape)} dtype={o_local.dtype}")
                # print(f"[CCModel] global shape={tuple(o_global.shape)} dtype={o_global.dtype}")
                self._printed_once = True
        else:
            x = obs_in.float()                                # [B, local+global]
            o_local  = x[..., :self.local_dim]
            o_global = x[..., self.local_dim:]
            assert o_local.shape[-1]  == self.local_dim
            assert o_global.shape[-1] == self.global_dim
        
        if self._debug and self._actor_prints < 3:
            if not torch.allclose(o_local, torch.zeros_like(o_local)) or \
               not torch.allclose(o_global, torch.zeros_like(o_global)):
                loc0 = o_local[0].detach().cpu().numpy()
                glo0 = o_global[0].detach().cpu().numpy()
                # print(f"[CCMODEL/ACTOR] local[0]={_np_str(loc0)}  sum={loc0.sum():.6f}")
                # print(f"[MODEL/CRITIC-feed] global[0]={_np_str(glo0)}  sum={glo0.sum():.6f}")
                # print(" ")
                self._actor_prints += 1


        self._vf_in = o_global
        logits = self.actor(o_local)
        return logits, state

    def value_function(self):
        if self._debug and self._vf_in is not None and self._critic_prints < 2:
            glo0 = self._vf_in[0].detach().cpu().numpy()
            # print(f"[_CCMODEL/CRITIC] cached global[0]={_np_str(glo0)}  sum={glo0.sum():.6f}")
            self._critic_prints += 1

        if self._vf_in is None:
            device = next(self.parameters()).device
            self._vf_in = torch.zeros((1, self.global_dim), device=device, dtype=torch.float32)
        return self.critic(self._vf_in).squeeze(-1)

ModelCatalog.register_custom_model("torch_cc_model", TorchCCModel)
