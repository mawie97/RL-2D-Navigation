import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env_rllib_wrapper import register_sar_env
import cc_model as cc_model
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


class PeekResultKeys(DefaultCallbacks):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        ks = [k for k in result if k.startswith("env_runners/")]
        print("[Peek] iter:", result.get("training_iteration"),
              "| env_runners keys:", ks,
              "| eps_this_iter via hist:", len(result.get("env_runners/hist_stats/episode_lengths", [])))

class MyCallbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.total_eps = 0          # cumulative episodes we expose
        self._prev_num_eps = 0      # last seen cumulative from RLlib

    def on_learn_on_batch_begin(self, *, policy, train_batch, **kwargs):
        rewards   = train_batch[SampleBatch.REWARDS]
        agent_idx = train_batch.get("agent_index", None)
        agent_ids = train_batch.get("agent_id", None)
        n = min(8, len(rewards))
        print("\n[BATCH] --- sample rows ---")
        for i in range(n):
            who = f"id={agent_ids[i]}" if agent_ids is not None else f"idx={int(agent_idx[i])}"
            print(f"  row {i}: {who}  reward={float(rewards[i]):.3f}")

    # Do NOT accumulate in on_episode_end (runs on workers)

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Prefer RLlib's per-iter metric if available
        eps_this = result.get("env_runners/episodes_this_iter", None)

        # Fallback: compute per-iter episodes by diffing the cumulative metric
        if not eps_this:
            cum = result.get("env_runners/num_episodes", 0) or 0
            try:
                cum = int(cum)
            except Exception:
                cum = int(np.asarray(cum).item())
            eps_this = max(0, cum - self._prev_num_eps)
            self._prev_num_eps = cum

        try:
            eps_this = int(eps_this)
        except Exception:
            eps_this = int(np.asarray(eps_this).item())

        self.total_eps += eps_this
        result["cumulative_episodes"] = self.total_eps

        print(f"[train_result] iter={result.get('training_iteration')} "
              f"eps_this_iter(computed)={eps_this} "
              f"num_episodes(cum)={result.get('env_runners/num_episodes')} "
              f"cumulative_episodes={self.total_eps}")

TARGET_EPS = 20

def stop_by_episodes(trial_id, result):
    total = (result.get("env_runners/num_episodes")
             or result.get("episodes_total")
             or 0)
    try:
        return int(total) >= TARGET_EPS
    except Exception:
        return False  # be defensive if metric is weird for the very first iter


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"

if __name__ == "__main__":
    # Used to parallelise environment rollouts and training
    ray.init(ignore_reinit_error=True)
    
    HERE = Path(__file__).resolve().parent
    project_root = HERE.parent
    xmls = [str(project_root / "layouts" / "base_layout.xml")]   # absolute path
    csv_path = str(project_root / "train_log.csv")    
    # Register env
    env_name = register_sar_env(
        name="MultiAgentSAR-v0",
        csv=csv_path,
        xmls=xmls
    )

    OBS_DIM, N_AGENTS = 12, 2
    COVER_K = 4
    COV_DIM = (20 // COVER_K) * (20 // COVER_K)  # 25
    GLOBAL_DIM = OBS_DIM * N_AGENTS + COV_DIM

    dict_obs_space = DictSpace({
        "obs":   Box(-np.inf, np.inf, shape=(OBS_DIM,),    dtype=np.float32),
        "state": Box(-np.inf, np.inf, shape=(GLOBAL_DIM,), dtype=np.float32),
    })
    act_space = Box(-0.1, 0.1, shape=(2,), dtype=np.float32)

    config = (
        PPOConfig()
        .framework("torch")
        .environment(
            env=env_name,
            env_config={
                "csv_log_path": csv_path,
                "xml_paths": xmls,
                "render_enabled": False,      # render only in eval
            },
        )
        .multi_agent(
            policies={
                "shared_policy": (None, dict_obs_space, act_space, {"framework": "torch"})
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .training(
            model={
                "custom_model": "torch_cc_model",
                "vf_share_layers": False,
                "_disable_preprocessor_api": True,
                "free_log_std": True,   # try to keep Dict intact
                "custom_model_config": {             # but be robust if flattened
                    "local_dim": OBS_DIM,
                    "global_dim": GLOBAL_DIM,
                    "debug_dump": True,
                },
            },
            gamma=0.995, lr=3e-4, lambda_=0.95,
            train_batch_size=200, minibatch_size=200, num_epochs=10,
            clip_param=0.2, vf_clip_param=100.0, entropy_coeff=0.001,
        )
        .callbacks(MyCallbacks)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=1, rollout_fragment_length=200, batch_mode="complete_episodes")
        .api_stack(enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False)
    )

        # Choose where Tune writes results/checkpoints
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "ray_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tuner = tune.Tuner(
        "PPO",
        run_config=RunConfig(
            stop={"episodes_total": 5},
            # stop={"training_iteration": 1},
            # stop={"cumulative_episodes": 3},                
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
        print({
            "iter": r.metrics.get("training_iteration"),
            "cum_eps": r.metrics.get("env_runners/num_episodes"),
            "eps_this_iter": r.metrics.get("env_runners/episodes_this_iter"),
            "timesteps_total": r.metrics.get("timesteps_total"),
            "time_s": r.metrics.get("time_total_s"),
        })
    best = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
    # print("Best mean reward:", best.metrics.get("episode_reward_mean"))
    # print("Best checkpoint:", best.checkpoint)

    