import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard.writer import SummaryWriter
from multi_ray_goal_env import MujocoGoalEnv

current_dir = os.path.dirname(__file__)
xml_base_dir = os.path.abspath(os.path.join(current_dir, 'layouts', 'eval', 'lvl_1_4'))
xml_paths = sorted([os.path.join(xml_base_dir, f) for f in os.listdir(xml_base_dir) if f.endswith(".xml")])

current_setup = "new_hybird_noise1"

base_dir = os.path.join(current_dir, "runs", current_setup)

csv_log_path = os.path.join(base_dir, "eval", "noise0", "lvl_1_4", "eval_log.csv")
os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)

tensorboard_eval_log_dir = os.path.join(
    base_dir, "eval", "noise0", "lvl_1_4", "tensorboard"
)
os.makedirs(tensorboard_eval_log_dir, exist_ok=True)
            
model_path = os.path.join(base_dir, "models", "model_ppo")
vecnorm_path = os.path.join(base_dir, "envs", "vecnormalize.pkl")

headless = True

eval_monitor_path = os.path.join(base_dir, "eval", "noise0", "lvl_1_4", "monitor.csv")
os.makedirs(os.path.dirname(eval_monitor_path), exist_ok=True)

def make_env():
    env = MujocoGoalEnv(csv_log_path, xml_paths, headless)
    return Monitor(env, filename=eval_monitor_path)

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize.load(vecnorm_path, eval_env)

eval_env.training = False
eval_env.norm_reward = False
eval_env.norm_obs = True

model = PPO.load(model_path)
writer = SummaryWriter(log_dir=tensorboard_eval_log_dir)

num_episodes = 0
max_episodes = 200
episode_rewards = []

obs = eval_env.reset()
while num_episodes < max_episodes:
    action, _ = model.predict(obs)  # type: ignore
    obs, reward, done, info = eval_env.step(action)

    eval_env.venv.envs[0].render()  # type: ignore

    if done[0]:
        num_episodes += 1
        ep_info = info[0].get("episode")
        if ep_info is not None:
            episode_rewards.append(ep_info["r"])
            writer.add_scalar("Eval/episode_reward", ep_info["r"], num_episodes)
            print(f"Episode {num_episodes} reward: {ep_info['r']}")
            

avg_reward = sum(episode_rewards) / len(episode_rewards)
print(f"Average reward over {max_episodes} episodes: {avg_reward}")
writer.add_scalar("Eval/average_reward", avg_reward, 0)

writer.close()
eval_env.close()
