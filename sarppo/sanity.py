# from pathlib import Path
# import numpy as np
# from mappo_mujoco_env import MultiAgentSAR


# HERE = Path(__file__).resolve().parent
# project_root = HERE.parent
# xmls = [str(project_root / "layouts")]   # xml path
# env = MultiAgentSAR(csv_log_path=None, xml_paths=xmls, render_enabled=True)
# obs, info = env.reset()
# for t in range(10000):
#     # small constant action for both agents
#     action = {name: np.array([0.1, 0.1], dtype=np.float32) for name in env.agent_names}
#     obs, rew, term, trunc, info = env.step(action)
#     env.render()
# env.close()

from pathlib import Path
import numpy as np
import mujoco
from mappo_mujoco_env import MultiAgentSAR


def debug_move_agent_to_delta(env: MultiAgentSAR,
                              agent_name: str = "agent_1",
                              delta = (0.1, 0.1),
                              n_substeps: int = 30000):
    """
    Debug helper:
    - apply one action (delta x,y) to agent_name
    - then run MuJoCo for n_substeps
    - render every substep and print the agent's world position

    This lets you SEE how the PID (kp,ki,kd) moves the agent toward the target.
    """
    import time
    delta = np.asarray(delta, dtype=np.float32)

    # 1) Build an action dict for all agents: target delta for one, zeros for others
    action_dict = {
        name: (delta.copy() if name == agent_name else np.zeros(2, dtype=np.float32))
        for name in env.agent_names
    }

    # 2) Use your existing logic to set data.ctrl[...] via _apply_actions_for_agents
    env._apply_actions_for_agents(action_dict)

    # For sanity: print the world target the env computed
    tgt_world = env._step_targets[agent_name]
    print(f"{agent_name} target world pos (from delta {delta}): {tgt_world}")

    # 3) Figure out which body we track for world position
    body_id = env.agent_map[agent_name]["body_id"]

    # 4) Run substeps, render, and print position
    dt = env.model.opt.timestep
    for sub in range(n_substeps):
        mujoco.mj_step(env.model, env.data)

        # world position of this agent (x,y)
        pos_xy = env.data.xpos[body_id][:2].copy()

        # print every few substeps
        if sub % 1000 == 0:
            t = sub * dt
            print(f"sub={sub:4d}  t={t:5.3f}s  {agent_name} pos=({pos_xy[0]: .4f}, {pos_xy[1]: .4f})")

        # show each substep (this is what slows things down visually)
        env.render()
        # time.sleep(0.01)   # increase to 0.02 / 0.03 if still too fast


if __name__ == "__main__":

    HERE = Path(__file__).resolve().parent
    project_root = HERE.parent
    xmls = [str(project_root / "layouts")] 

    env = MultiAgentSAR(
        csv_log_path=None,
        xml_paths=xmls,
        render_enabled=True,
        seed=0,
    )

    obs, info = env.reset()

    # Move agent_1 by delta = (0.1, 0.1) once, and watch how it moves
    debug_move_agent_to_delta(env, agent_name="agent_1", delta=(0.1, 0.1), n_substeps=50000)

    env.close()
