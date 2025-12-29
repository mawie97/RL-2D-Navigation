import os
import sys
from calendar import c
from math import e
from train_env import train_ppo
from multi_ray_goal_env import MujocoGoalEnv
import os

env_class = MujocoGoalEnv

current_dir = os.path.dirname(__file__)  # scripts/train
xml_base_dir = os.path.abspath(os.path.join(current_dir, 'layouts', 'train', 'random_bresenham'))
base_dir = os.path.abspath(os.path.join(current_dir, 'runs', 'new_random_bresenham_noise1'))
xml_paths = sorted([os.path.join(xml_base_dir, f) for f in os.listdir(xml_base_dir) if f.endswith(".xml")])

log_dir = os.path.join(base_dir, "logs")
csv_log_path = os.path.join(base_dir, "self_log.csv")
model_dir = os.path.join(base_dir, "models")
env_dir = os.path.join(base_dir, "envs")
seed_value = 1234
num_episodes = 1500
headless = True

print("[INFO] Loaded XMLs:", xml_paths)

train_ppo(
    env_class,
    xml_paths,
    base_dir,
    log_dir,
    csv_log_path,
    model_dir,
    env_dir,
    seed_value,
    num_episodes,
    headless,
    total_timesteps=int(1e6)


)
