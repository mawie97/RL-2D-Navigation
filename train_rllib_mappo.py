import ray
from ray.rllib.algorithms.ppo import PPOConfig
from env_rllib_wrapper import register_sar_env
import cc_model
import numpy as np
from gymnasium.spaces import Box

def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # Register env
    env_name = register_sar_env(
        name="MultiAgentSAR-v0",
        csv="train_log.csv",
        xmls=["layouts/base_layout.xml"]
    )

    OBS_DIM, N_AGENTS = 12, 2
    GLOBAL_DIM = OBS_DIM * N_AGENTS


    ACT_DIM = 2
    single_obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
    single_act_space = Box(low=-0.1, high=0.1, shape=(ACT_DIM,), dtype=np.float32)


    config = (
        PPOConfig()
        .framework("torch")
        .environment(env=env_name,
                     env_config={"csv_log_path": "train_log.csv",
                                 "xml_paths": ["layouts/base_layout.xml"],"render_enabled": True })
        .multi_agent(
            policies={"shared_policy": (None, single_obs_space,     # <<< force a flat Box obs for each agent
            single_act_space, {"framework": "torch"})},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"],
        )
        .training(
            model={"custom_model": "torch_cc_model",
                   "custom_model_config": {"global_state_dim": GLOBAL_DIM},
                   "vf_share_layers": False},
            gamma=0.995, lr=3e-4, lambda_=0.95,
            train_batch_size=32768, minibatch_size=2048, num_epochs=10,
            clip_param=0.2, vf_clip_param=100.0, entropy_coeff=0.0,
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=3, rollout_fragment_length=200)
        .api_stack(enable_rl_module_and_learner=False,
           enable_env_runner_and_connector_v2=False)


    )

    algo = config.build_algo()


    for i in range(200):
        result = algo.train()
        print(f"iter {i}: mean ep reward={result['episode_reward_mean']:.2f}")
        if i % 10 == 0:
            checkpoint = algo.save()
            print("Checkpoint saved at", checkpoint)
