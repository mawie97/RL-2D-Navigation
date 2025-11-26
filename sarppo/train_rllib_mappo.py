import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env_rllib_wrapper import register_sar_env
import cc_model as cc_model
# simport cc_model_copy as cc_model
import numpy as np
from gymnasium.spaces import Dict as DictSpace, Box
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from pathlib import Path
import datetime as dt
from ray import tune
from ray.tune import RunConfig, CheckpointConfig
from pathlib import Path
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np

from ray import tune
from ray.tune.stopper import Stopper, CombinedStopper, MaximumIterationStopper
import numpy as np

def grab(m, nested_path, flat_key):
    d = m
    for k in nested_path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            d = None
            break
    if d is not None and not isinstance(d, dict):
        return d
    return m.get(flat_key)

class MyCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.total_eps = 0
        self._prev_hist_len = 0

    def on_train_result(self, *, algorithm, result, **kwargs):    
        eps_this  = grab(result, ("env_runners","num_episodes"), "env_runners/num_episodes")          
        self.total_eps += int(eps_this)
        result["my_cumulative_episodes"] = self.total_eps
        print(f"[train_result] iter={result.get('training_iteration')} eps_this_iter={eps_this} cumulative_episodes={self.total_eps}")

def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"

if __name__ == "__main__":

    ray.init(ignore_reinit_error=True)
    
    HERE = Path(__file__).resolve().parent
    project_root = HERE.parent
    xmls = [str(project_root / "layouts")]   # xml path
    csv_path = str(project_root / "train_log.csv")    
    # Register env
    env_name = register_sar_env(
        name="MultiAgentSAR-v0",
        csv=csv_path,
        xmls=xmls
    )

    OBS_DIM, N_AGENTS = 18, 2
    COV_DIM = 400
    EXTRA_DIM = 3
    GLOBAL_DIM = OBS_DIM * N_AGENTS + COV_DIM + EXTRA_DIM

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=env_name,
            env_config={
                "csv_log_path": csv_path,
                "xml_paths": xmls,
                "render_enabled": True,    
            },
        )
        .multi_agent(
            policies={
                "shared_policy": (None, None, None, {"framework": "torch"})
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .training(
            model={
                "custom_model": "torch_cc_model",
                "vf_share_layers": False,
                "_disable_preprocessor_api": True,
                "free_log_std": True,
                "custom_model_config": {
                    "local_dim": OBS_DIM,
                    "global_dim": GLOBAL_DIM,
                    "debug_dump": True,
                },
            },
            gamma=0.995, lr=3e-4, lambda_=0.95,
            train_batch_size=1000, minibatch_size=256, num_epochs=10,
            clip_param=0.2, vf_clip_param=100.0, entropy_coeff=0.05,
        )
        .callbacks(MyCallbacks)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=1, rollout_fragment_length=500, batch_mode="complete_episodes")
        .api_stack(enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False)
    )

    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "ray_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    tuner = tune.Tuner(
        "PPO",
        run_config=RunConfig(
            stop={"my_cumulative_episodes": 3000},              
            storage_path=str(results_dir),                     
            name="sar_ppo_run",
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,                          # always save at the end
                checkpoint_frequency=5,                          # and every 5 iters (optional)
                num_to_keep=3,                                   # keep latest 3 (optional)
            ),
        ),
        param_space=config.to_dict(),
    )

    result_grid = tuner.fit()
 
    for r in result_grid:
        eps_this = grab(r.metrics, ("env_runners","episodes_this_iter"), "env_runners/episodes_this_iter")
        num_eps  = grab(r.metrics, ("env_runners","num_episodes"), "env_runners/num_episodes")
        my_cum_epis = grab(r.metrics, ("my_cumulative_episodes"),"my_cumulative_episodes")
        print({"iter": r.metrics.get("training_iteration"),
            "eps_this_iter": eps_this, "num_episodes": num_eps, "My_episo":  my_cum_epis})


    