from multi_ray_goal_env import MujocoGoalEnv
import numpy as np
import os
import glob

xml_root = "/Users/susu/Desktop/Thesis_Project/navppo/layouts/train/setup_1"
xml_paths = sorted(glob.glob(os.path.join(xml_root, "*.xml")))
csv_log_path = "debug_steps.csv"

env = MujocoGoalEnv(csv_log_path, xml_paths)

obs, info = env.reset()
print("Initial obs shape:", obs.shape)

for t in range(50):
    # random small action
    action = env.action_space.sample()
    print(f"Step {t}, action:", action)
    obs, reward, done, truncated, info = env.step(action)
    print("  reward:", reward, "done:", done)
    if done:
        print("Episode ended at step", t)
        break

env.close()
env.close_file()
print("Finished stepping.")
