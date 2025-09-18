# run_sar_mappo.py
from pathlib import Path

# --- imports from your repo ---
from onpolicy.config import get_config
from onpolicy.runner.shared.sar_runner import Runner as RunnerClass
from env.MultiAgentSAR import MultiAgentSAR
from sar_adapter.sar_env import SAROnPolicyWrapper

def make_env(xmls, agent_names, use_symseed=False):
    base = MultiAgentSAR(csv_log_path=None, xml_paths=xmls, agent_names=tuple(agent_names))
    if use_symseed:
        # base.symseed_sampler = your_sampler_callable
        pass
    return SAROnPolicyWrapper(base, agent_names)

if __name__ == "__main__":
    # get_config() returns an ArgumentParser in your version
    parser = get_config()
    # add/override CLI options you want
    parser.add_argument("--xmls", nargs="+", default=[str((Path.cwd() / "layouts" / "base_layout.xml").resolve())])
    parser.add_argument("--agents", nargs="+", default=["agent_0", "agent_1"])
    parser.add_argument("--symseed", action="store_true")

    args = parser.parse_args()
    args.num_env_steps = 200000
    args.episode_length = 300

    # ---- sensible defaults for MAPPO + our runner ----
    args.algorithm_name = "mappo"
    args.env_name = "sar_mujoco"
    args.use_wandb = False                # avoids wandb dependency
    args.use_tensorboard = True           # logs with tensorboardX
    args.use_centralized_V = True         # CTDE
    args.share_policy = True              # shared actor across agents
    args.use_feature_normalization = True
    args.use_recurrent_policy = False     # no RNN unless you add it
    args.n_rollout_threads = 1            # our sar_runner is single-threaded
    args.num_env_steps = args.num_env_steps
    args.episode_length = args.episode_length
    args.num_agents = len(args.agents)

    # build env (adapter returns (N, obs/state) arrays as expected by onpolicy)
    env = make_env(args.xmls, args.agents, use_symseed=args.symseed)

    # build config dict that BaseRunner expects
    cfg = {
        "all_args": args,
        "envs": env,             # training env
        "eval_envs": None,       # optional
        "device": "cpu",         # or "cuda"
        "num_agents": len(args.agents),
        "run_dir": Path("results/sar"),
    }

    runner = RunnerClass(cfg)
    runner.run()
