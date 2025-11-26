# eval_rllib.py
import ray
from pathlib import Path
from ray.rllib.algorithms.algorithm import Algorithm

# Your wrapper pieces
from env_rllib_wrapper import register_sar_env, make_env_creator
import cc_model  # ensure custom model is registered
import numpy as np

def pack_agent_obs(ob_agent):
    # ob_agent must be like {"obs": np.array([local...]), "state": np.array([global...])}
    return np.concatenate(
        [np.asarray(ob_agent["obs"], dtype=np.float32).ravel(),
         np.asarray(ob_agent["state"], dtype=np.float32).ravel()],
        axis=0
    ).astype(np.float32)

def policy_mapping_fn(agent_id, *a, **k):
    return "shared_policy"

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    HERE = Path(__file__).resolve().parent
    project_root = HERE.parent
    xmls = [str(project_root / "layouts")]
    csv_path = str(project_root / "eval_log.csv")

    # 1) Register RLlib env name BEFORE restoring (for worker creation)
    env_name = register_sar_env(name="MultiAgentSAR-v0", csv=csv_path, xmls=xmls)

    # 2) Restore the trained algo
    ckpt = "/Users/susu/Desktop/Thesis_Project/sarppo/ray_results/sar_ppo_run/PPO_MultiAgentSAR-v0_7e26c_00000_0_2025-11-13_13-58-47/checkpoint_000007"
    algo = Algorithm.from_checkpoint(ckpt)

    # 3) Build a local env instance for rendering (don’t use gym.make here)
    #    Use your creator (same logic as workers), pass render_enabled=True.
    local_env = make_env_creator(default_csv=csv_path, default_xmls=xmls)({
        "csv_log_path": csv_path,
        "xml_paths": xmls,
        "render_enabled": True,
        # "seed": 123,  # optional
    })

    policy = algo.get_policy("shared_policy")

    for ep in range(100):
        obs, _ = local_env.reset()     # obs: {agent_id: {"obs":..., "state":...}}
        ep_ret = {aid: 0.0 for aid in obs}

        while True:
            if not obs:   # defensive: can be empty at episode end
                break

            actions = {}
            for aid, ob_agent in obs.items():
                act, _, _ = policy.compute_single_action({"obs": ob_agent["obs"], "state": ob_agent["state"]}, explore=False, clip_action=True)
                actions[aid] = act

            obs, rewards, terminations, truncations, infos = local_env.step(actions)
            print(f"Actions: {actions}")
            for aid, r in rewards.items():
                ep_ret[aid] = ep_ret.get(aid, 0.0) + float(r)

            # ✅ stop when episode ended
            if terminations.get("__all__", False) or truncations.get("__all__", False):
                break

        # optional: show done reason if env provided one
        reason = None
        for info in (infos or {}).values():
            if isinstance(info, dict) and "done_reason" in info:
                reason = info["done_reason"]; break
        print(f"[EVAL] returns={ep_ret} | total={sum(ep_ret.values()):.3f} reason={reason}")
