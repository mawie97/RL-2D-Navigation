# This is used to wrap the MuJoCo environment into 
# a creator function compatible with RLlib and registers 
# it with the RLlib registry.
# RLlib expects a factory (creator) function, not an environment instance.
from ray.tune.registry import register_env
from env.mappo_mujoco_env import MultiAgentSAR

# def make_env_creator(default_csv=None, default_xmls=None):
#     def _creator(cfg):
#         env = MultiAgentSAR(
#             csv_log_path=cfg.get("csv_log_path", default_csv),
#             xml_paths=cfg.get("xml_paths", default_xmls),
#         )
#         return env.with_agent_groups({"agent_group": ["agent_1","agent_2"]})
#     return _creator

def make_env_creator(default_csv=None, default_xmls=None):
    def _creator(cfg):
        # print(
        #     f"ENV INIT | worker_index={getattr(env_config, 'worker_index', None)} "
        #     f"vector_index={getattr(env_config, 'vector_index', None)} "
        #     f"in_eval={getattr(env_config, 'in_evaluation', False)}",
        #     flush=True,
        # )
        return MultiAgentSAR(
            csv_log_path=cfg.get("csv_log_path", default_csv),
            xml_paths=cfg.get("xml_paths", default_xmls),
        )
    return _creator

def register_sar_env(name="MultiAgentSAR-v0", csv=None, xmls=None):
    register_env(name, make_env_creator(csv, xmls))
    return name
