

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
        
        cmc = model_config.get("custom_model_config", {})
        self._debug = bool(cmc.get("debug_dump", True)) 
        self._actor_prints = 0
        self._critic_prints = 0
        self._printed_once = False

        # Determine dims and whether we expect a Dict or a flat Box:
        self._expects_dict = isinstance(obs_space, DictSpace)
        if self._expects_dict:
            self.local_dim  = int(np.prod(obs_space["obs"].shape))
            self.global_dim = int(np.prod(obs_space["state"].shape))
        elif isinstance(obs_space, Box):
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

        # Centralized state split parameters
        # Separate global state into: [agent features | grid  | extras(3)]
        self.grid_size = 20
        self.grid_dim = self.grid_size * self.grid_size
        self.extra_dim = 3
        self.flat_global_dim = self.global_dim - (self.grid_dim + self.extra_dim)
        assert self.flat_global_dim > 0, (
            f"global_dim={self.global_dim} must equal flat+{self.grid_dim}+{self.extra_dim}"
        )

        # Coverage encoder: 1x5x5 -> 64
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),  
            nn.Flatten(),                                          
            nn.Linear(8 * self.grid_size * self.grid_size, 64), nn.ReLU()                          # [B, 64]
        )

        # Critic over [flat_global | cov_feat(64) | extras(3)]
        critic_in = self.flat_global_dim + 64 + self.extra_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_in, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Actor (local obs)
        self.actor = nn.Sequential(
            nn.Linear(self.local_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

        self._vf_in = None
        self._critic_in_dim = critic_in  # for the (rare) fallback path
        assert self.flat_global_dim % self.local_dim == 0, \
            f"flat_global_dim={self.flat_global_dim} should be n_agents*local_dim (local_dim={self.local_dim})."

    def forward(self, input_dict, state, seq_lens):
        obs_in = input_dict["obs"]
        
        if self._expects_dict:
            assert isinstance(obs_in, dict), "Obs must be a Dict with 'obs' and 'state'"
            o_local  = obs_in["obs"].float()                 # [B, local_dim]
            o_global = obs_in["state"].float()               # [B, global_dim]
        else:
            x = obs_in.float()                             # [B, local+global]
            o_local  = x[..., :self.local_dim]       # from start up to self.local_dim (exclusive)
            o_global = x[..., self.local_dim:]       # from self.local_dim to the end
            assert o_local.shape[-1]  == self.local_dim
            assert o_global.shape[-1] == self.global_dim
        
        if self._debug and not self._printed_once:
            print(f"[CCMODEL] local shape={tuple(o_local.shape)}, global shape={tuple(o_global.shape)}")
            self._printed_once = True
        
        # Strict split: [flat | coverage(100) | extras(3)]
        B = o_global.shape[0]
        assert o_global.shape[-1] == self.global_dim, "Global state length mismatch."

        flat_end   = self.flat_global_dim
        grid_end   = flat_end + self.grid_dim

        flat_global = o_global[..., :flat_end]                 # [B, flat]
        grid_flat   = o_global[..., flat_end:grid_end]         # [B, 100]
        extras      = o_global[..., grid_end:]                 # [B, 3]
        
        assert grid_flat.shape[-1] == self.grid_dim, f"Coverage slice wrong size: {grid_flat.shape[-1]} != {self.grid_dim}"
        assert extras.shape[-1]    == self.extra_dim, f"Extras slice wrong size: {extras.shape[-1]} != {self.extra_dim}"

        # grid_image  = grid_flat.view(B, 1, self.grid_size, self.grid_size)  # [B,1,5,5]
        grid_image  = grid_flat.reshape(B, 1, self.grid_size, self.grid_size)
        grid_feat   = self.grid_encoder(grid_image)                          # [B,64]
        self._vf_in = torch.cat([flat_global, grid_feat, extras], dim=-1)    # [B, flat+64+3]

        logits = self.actor(o_local)
        return logits, state

    def value_function(self):
        if self._debug and self._vf_in is not None and self._critic_prints < 2:
            glo0 = self._vf_in[0].detach().cpu().numpy()
            # print(f"[_CCMODEL/CRITIC] cached global[0]={_np_str(glo0)}  sum={glo0.sum():.6f}")
            self._critic_prints += 1

        if self._vf_in is None:
            device = next(self.parameters()).device
            self._vf_in = torch.zeros((1, self._critic_in_dim), device=device, dtype=torch.float32)
        return self.critic(self._vf_in).squeeze(-1)

ModelCatalog.register_custom_model("torch_cc_model", TorchCCModel)
