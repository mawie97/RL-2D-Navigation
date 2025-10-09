import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env_rllib_wrapper import register_sar_env
import cc_model
import numpy as np
from gymnasium.spaces import Dict as DictSpace, Box
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch

# class DebugBatches(DefaultCallbacks):
#     def on_learn_on_batch_begin(self, *, policy, train_batch, **kwargs):
#         # train_batch is for a single policy (shared policy).
#         obs = train_batch[SampleBatch.OBS]          # Dict: {"obs": ..., "state": ...}
#         agent_idx = train_batch["agent_index"]     # Per-row agent indices in this batch

#         local = obs["obs"]     # shape [B, 12]
#         global_ = obs["state"] # shape [B, 24]

#         # Print a few rows with which agent they came from.
#         n = min(4, len(local))
#         for i in range(n):
#             print(f"[BATCH] row {i} agent_idx={int(agent_idx[i])} "
#                   f"local[:4]={local[i][:4].tolist()} sum={float(local[i].sum()):.4f} "
#                   f"global_sum={float(global_[i].sum()):.4f}")

# Debug callback for reward!
class DebugBatches(DefaultCallbacks):
    def on_learn_on_batch_begin(self, *, policy, train_batch, **kwargs):
        rewards   = train_batch[SampleBatch.REWARDS]
        agent_idx = train_batch.get("agent_index", None)
        agent_ids = train_batch.get("agent_id", None)
        n = min(8, len(rewards))
        print("\n[BATCH] --- sample rows ---")
        for i in range(n):
            who = f"id={agent_ids[i]}" if agent_ids is not None else f"idx={int(agent_idx[i])}"
            print(f"  row {i}: {who}  reward={float(rewards[i]):.3f}")
def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"

if __name__ == "__main__":
    # Used to parallelise environment rollouts and training
    ray.init(ignore_reinit_error=True)

    # Register env
    env_name = register_sar_env(
        name="MultiAgentSAR-v0",
        csv="train_log.csv",
        xmls=["layouts/base_layout.xml"]
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
                "csv_log_path": "train_log.csv",
                "xml_paths": ["layouts/base_layout.xml"],
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
            train_batch_size=2048, minibatch_size=2048, num_epochs=10,
            clip_param=0.2, vf_clip_param=100.0, entropy_coeff=0.001,
        )
        # train_batch_size=32768
        .callbacks(DebugBatches)
        .resources(num_gpus=0)
        .env_runners(num_env_runners=1, rollout_fragment_length=200)
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_config={
                "render_env": True,
                "env_config": {"render_enabled": True},
            },
        )
        .api_stack(enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False)
    )


    algo = config.build_algo()

    for i in range(1):
        result = algo.train()
        print(f"iter {i}: mean ep reward={result['episode_reward_mean']:.2f}")
        if i % 10 == 0:
            checkpoint = algo.save()
            print("Checkpoint saved at", checkpoint)
